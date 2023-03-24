
import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl
import utils.model

from copy import deepcopy
from matplotlib import cm


class FWIModel(pl.LightningModule):
    """
    General model framework for FWI.
    
    The `build_generators` method in the class will 
    build two models: one for forward, and one for inverse. 

    For invertable networks training, use the `InvertibleFWIModel`.
    """
    def __init__(self, config: dict):
        super(FWIModel, self).__init__()

        self.automatic_optimization = False

        self.parse_config(config)
        self.build_generators()
        self.build_discriminators()
        self.build_train_metrics()
        self.build_eval_metrics()
        self.safety_check()

    def parse_config(self, config: dict):
        # get model config
        model_config = config["model"]

        # Parse model config
        self.model_config = model_config
        self.name = model_config.get("name")
        self.amp_config = model_config.get('vel_to_amp', {})
        self.vel_config = model_config.get('amp_to_vel', {})
    
        # Parse training config
        self.training_config = config["training"]
        self.optimizer_config = self.training_config["optimizer"]
        self.scheduler_config = self.training_config.get("scheduler", {})

        # save the configs to the output folder
        self.save_hyperparameters()

    def build_generators(self):
        # Define generator models
        self.amp_gen = timm.create_model(
            self.amp_config['gen_config']['class'],
            **self.amp_config['gen_config']['params']
        )
        self.vel_gen = timm.create_model(
            self.vel_config['gen_config']['class'],
            **self.vel_config['gen_config']['params']
        )

    def build_discriminators(self):
        # Define discriminator models
        if "disc_config" in self.amp_config:
            self.amp_disc = timm.create_model(
                self.amp_config['disc_config']['class'],
                **self.amp_config['disc_config']['params']
            )
            self.use_amp_disc = True
        else:
            self.use_amp_disc = False

        if "disc_config" in self.vel_config:
            self.vel_disc = timm.create_model(
                self.vel_config['disc_config']['class'],
                **self.vel_config['disc_config']['params']
            )
            self.use_vel_disc = True
        else:
            self.use_vel_disc = False

    def safety_check(self):
        """ Do some safety check before running. """ 
        # check whether a GAN loss is needed for amplitude model
        if self.use_amp_disc:
            amp_metrics = self.train_metrics["amp_model"]
            if "gan_loss" not in amp_metrics:
                raise ValueError(
                    "Ampltitude model has been set to use discriminator." +
                    "However, `gan_loss` is not found in `train_metrics`." +
                    "Try to set `gan_loss` either under `model` or under `amp_to_vel`."
                )
        # check whether a GAN loss is needed for velocity model
        if self.use_vel_disc:
            vel_metrics = self.train_metrics["vel_model"]
            if "gan_loss" not in vel_metrics:
                raise ValueError(
                    "Velocity model has been set to use discriminator." +
                    "However, `gan_loss` is not found in `train_metrics`." +
                    "Try to set `gan_loss` either under `model` or under `vel_to_amp`."
                )

    def build_train_metrics(self):
        # Build training metrics
        self.build_metrics(name="train_metrics")

    def build_eval_metrics(self):
        # Build evaluation metrics
        self.build_metrics(name="eval_metrics")

    def build_metrics(self, name: str):
        output_metrics = {}

        # Define global training metrics
        if name in self.model_config:
            metric_config = self.model_config[name]
            global_metrics = self.build_metrics_from_config(
                metric_config, 
            )
        else:
            global_metrics = {}

        # metrics for amplitude model
        output_metrics["amp_model"] = deepcopy(global_metrics)
        if name in self.amp_config:
            metric_config = self.amp_config[name]
            metrics = self.build_metrics_from_config(
                metric_config, 
            )
            output_metrics["amp_model"].update(metrics)

        # metrics for velocity model
        output_metrics["vel_model"] = deepcopy(global_metrics)
        if name in self.vel_config:      
            metric_config = self.vel_config[name]
            metrics = self.build_metrics_from_config(
                metric_config, 
            )
            output_metrics["vel_model"].update(metrics)

        # create a member variable with the same name as `name`
        setattr(self, name, output_metrics)
        
    def build_metrics_from_config(self, metric_config: dict):
        # Define metrics parsing
        metrics = {}
        for name, item in metric_config.items():
            metric = eval(item["class"])(
                **item.get("params", {})
            )
            metric = {
                "weight": float(item.get("weight", 1.0)),
                "metric": metric,
            }
            metrics[name] = metric
        return metrics

    def forward_amp(self, amp):
        # Pass amplitude through amp-to-vel generator
        vel = self.vel_gen(amp)
        return vel

    def forward_vel(self, vel):
        # Pass velocity through vel-to-amp generator
        amp = self.amp_gen(vel)
        return amp

    def _make_optimizer(self, params, config):
        OPTIMIZER = eval(str(config["class"]))
        optimizer = OPTIMIZER(
            params, 
            **config.get("params", {})
        )
        return optimizer

    def _make_scheduler(self, optimizer, config):
        SCHEDULER = eval(str(config["class"]))
        scheduler = SCHEDULER(
            optimizer, 
            **config.get("params", {})
        )
        return scheduler

    def build_optimizers_and_schedulers(self, params, key):
        optimizers, schedulers = [], []
        # generators
        if key in self.optimizer_config:
            opt_config = self.optimizer_config[key]
        else:
            opt_config = self.optimizer_config
        if key in self.scheduler_config:
            sch_config = self.scheduler_config[key]
        else:
            sch_config = self.scheduler_config

        if opt_config != {}:  # shared optimizer
            gen_opt = self._make_optimizer(params, opt_config)
            optimizers.append(gen_opt)
            if sch_config != {}:
                gen_sch = self._make_scheduler(gen_opt, sch_config)
                schedulers.append(gen_sch)
        return optimizers, schedulers

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        # Create optimizers for generators
        params = list(self.amp_gen.parameters()) + \
                 list(self.vel_gen.parameters())
        opt, sch = self.build_optimizers_and_schedulers(
            params, key="generator"
        )
        optimizers.extend(opt)
        schedulers.extend(sch)

        # Create optimizers for amplitude discriminators
        if self.use_amp_disc:
            params = self.amp_disc.parameters()
            opt, sch = self.build_optimizers_and_schedulers(
                params, key="amp_disc"
            )
            optimizers.extend(opt)
            schedulers.extend(sch)

        # Create optimizers for velocity discriminators
        if self.use_vel_disc:
            params = self.vel_disc.parameters()
            opt, sch = self.build_optimizers_and_schedulers(
                params, key="vel_disc"
            )
            optimizers.extend(opt)
            schedulers.extend(sch)
        
        return optimizers, schedulers

    def discriminator_step(self, batch):
        # unpack batch
        vel, amp, fake_vel, fake_amp = batch
        stats = {}
        d_loss = 0

        # loss for amplitude prediction
        if self.use_amp_disc:
            prefix = "train/amplitude/discriminator/"

            # unpack optimizers
            amp_disc_opt = self.optimizers()[1]
            amp_metrics = self.train_metrics["amp_model"]
            if "disc_loss" in amp_metrics:   # use disc_loss
                gan_loss = amp_metrics["disc_loss"]["metric"]
                weight = amp_metrics["disc_loss"]["weight"]
            else:       # else use gan_loss
                gan_loss = amp_metrics["gan_loss"]["metric"]
                weight = amp_metrics["gan_loss"]["weight"]

            real_score = self.amp_disc(amp)
            fake_score = self.amp_disc(fake_amp.detach())
            d_loss_amp = weight * gan_loss(
                real_score, 
                torch.ones_like(real_score)
            ) + weight * gan_loss(
                fake_score, 
                torch.zeros_like(fake_score)
            )

            d_loss = d_loss + d_loss_amp
            stats[prefix + "loss"] = d_loss_amp.item()

        # loss for velocity prediction      
        if self.use_vel_disc:  
            prefix = "train/velocity/discriminator/"

            # unpack optimizers
            vel_disc_opt = self.optimizers()[2]
            vel_metrics = self.train_metrics["vel_model"]
            if "disc_loss" in vel_metrics:   # use disc_loss
                gan_loss = vel_metrics["disc_loss"]["metric"]
                weight = vel_metrics["disc_loss"]["weight"]
            else:      # else use gan_loss
                gan_loss = vel_metrics["gan_loss"]["metric"]
                weight = vel_metrics["gan_loss"]["weight"]

            real_score = self.vel_disc(vel)
            fake_score = self.vel_disc(fake_vel.detach())
            d_loss_vel = weight * gan_loss(
                real_score, 
                torch.ones_like(real_score)
            ) + weight * gan_loss(
                fake_score, 
                torch.zeros_like(fake_score)
            )

            d_loss = d_loss + d_loss_vel
            stats[prefix + "loss"] = d_loss_vel.item()

        # backward and update
        if self.use_amp_disc or self.use_vel_disc:
            if self.use_amp_disc:
                amp_disc_opt.zero_grad()
            if self.use_vel_disc:
                vel_disc_opt.zero_grad()
            self.manual_backward(d_loss)
            if self.use_amp_disc:
                amp_disc_opt.step()
            if self.use_vel_disc:
                vel_disc_opt.step()
            self.log("d_loss", d_loss.item(), prog_bar=True)
        return stats

    def calc_loss(self, pred, target, metrics: dict):
        # Get loss statistics
        stats = {}
        total_loss = 0
        for name, items in metrics.items():
            loss = items["metric"](pred, target)
            stats[name] = loss.item()
            weight = items.get("weight", 1.0)
            total_loss = total_loss + items["weight"] * loss
        stats["total_loss"] = total_loss.item()

        # This `loss` will be the loss used for backward
        stats["loss"] = total_loss
        return stats

    def training_step(self, batch, batch_idx):
        # unpack optimizers
        optimizers = self.optimizers()
        if isinstance(optimizers, list) or isinstance(optimizers, tuple):
            gen_opt = optimizers[0]
        else:
            gen_opt = optimizers

        # training statistics
        stats = {}

        # forward generators
        vel, amp = batch
        fake_amp = self.amp_gen(vel)
        fake_vel = self.vel_gen(amp)
        
        # discriminator backward
        if self.use_amp_disc or self.use_vel_disc:
            disc_stats = self.discriminator_step(
                batch=[vel, amp, fake_vel, fake_amp]
            )
            stats.update(disc_stats)

        # overall generator loss
        g_loss = 0

        # Calculate for `amp_model`:
        #   (special for `gan_loss`, `disc_loss` and `cycle_loss`)
        if "amp_model" in self.train_metrics:
            metrics = deepcopy(self.train_metrics["amp_model"])
            prefix = "train/amplitude/generator/"

            # gan loss
            if "gan_loss" in metrics:
                gan_loss = metrics.pop("gan_loss")
                weight = gan_loss["weight"]

                # calculate generator gan losses for amplitude model
                if self.use_amp_disc:
                    fake_score = self.amp_disc(fake_amp)
                    loss_gan = gan_loss["metric"](
                        fake_score, 
                        torch.ones_like(fake_score)
                    )
                    g_loss = g_loss + weight * loss_gan
                    stats[prefix + "gan_loss"] = loss_gan.item()

            # cycle consistency loss
            if "cycle_loss" in metrics:
                cycle_loss = metrics.pop("cycle_loss")
                weight = cycle_loss["weight"]
                cycle_loss = cycle_loss["metric"]

                cycle_amp = self.amp_gen(fake_vel)
                loss_cycle = cycle_loss(cycle_amp, amp)

                g_loss = g_loss + weight * loss_cycle
                stats[prefix + "cycle_loss"] = loss_cycle.item()

            # remove disc_loss
            if "disc_loss" in metrics:
                metrics.pop("disc_loss")
            
            # other losses
            g_losses_amp = self.calc_loss(
                pred=fake_amp, 
                target=amp, 
                metrics=metrics
            ) 
            g_loss = g_loss + g_losses_amp.pop("loss")
            stats.update({prefix + k: v for k, v in g_losses_amp.items()})

        # Calculate for `vel_model`:
        #   (special for `gan_loss`, `disc_loss` and `cycle_loss`)
        if "vel_model" in self.train_metrics:
            metrics = deepcopy(self.train_metrics["vel_model"])
            prefix = "train/velocity/generator/"

            # gan loss
            if "gan_loss" in metrics:
                gan_loss = metrics.pop("gan_loss")
                weight = gan_loss["weight"]

                # calculate generator gan losses for velocity model
                if self.use_vel_disc:
                    fake_score = self.vel_disc(fake_vel)
                    loss_gan_vel = gan_loss["metric"](
                        fake_score, 
                        torch.ones_like(fake_score)
                    )
                    g_loss = g_loss + weight * loss_gan_vel
                    stats[prefix + "gan_loss"] = loss_gan_vel.item()

            # cycle consistency loss
            if "cycle_loss" in metrics:
                cycle_loss = metrics.pop("cycle_loss")
                weight = cycle_loss["weight"]
                cycle_loss = cycle_loss["metric"]

                cycle_vel = self.vel_gen(fake_amp)
                loss_cycle = cycle_loss(cycle_vel, vel)

                g_loss = g_loss + weight * loss_cycle
                stats[prefix + "cycle_loss"] = loss_cycle.item()

            # remove disc_loss
            if "disc_loss" in metrics:
                metrics.pop("disc_loss")

            # other losses
            g_losses_vel = self.calc_loss(
                pred=fake_vel, 
                target=vel, 
                metrics=metrics
            ) 
            g_loss = g_loss + g_losses_vel.pop("loss")
            stats.update({prefix + k: v for k, v in g_losses_vel.items()})

        # backward velocity and amplitude model 
        gen_opt.zero_grad()
        self.manual_backward(g_loss)
        gen_opt.step()

        # log the metrics to tensorboard
        self.log("g_loss", g_loss.item(), prog_bar=True)
        self.log_dict(stats, prog_bar=False)

        step_output = {
            "pred_amp": fake_amp.detach(),
            "true_amp": amp,
            "pred_vel": fake_vel.detach(),
            "true_vel": vel,
        }
        return step_output

    def on_train_epoch_end(self):
        # learning rate scheduler update
        self.lr_scheduler_step(epoch=self.current_epoch)

        # log images
        for name, value in self.saved_output.items():
            self.log_image(name=f"train/{name}/{self.current_epoch}", tensor=value[0][0])

    def on_validation_epoch_end(self):
        # log images
        for name, value in self.saved_output.items():
            self.log_image(name=f"eval/{name}/{self.current_epoch}", tensor=value[0][0])
            
    def training_step_end(self, step_output):
        if step_output is not None:
            self.saved_output = step_output

        # log learning rate
        lr_schedulers = self.lr_schedulers()
        if isinstance(lr_schedulers, list) or isinstance(lr_schedulers, tuple):
            last_lr = lr_schedulers[0].get_last_lr()[0]
        else:
            last_lr = lr_schedulers.get_last_lr()[0]
        self.log('lr', last_lr)
        return super().training_step_end(step_output)

    def eval_with_metrics(self, pred, target, metrics):
        # evaluate performance
        stats = {}
        for name, metric in metrics.items():
            func = metric["metric"]
            value = func(pred, target)
            stats[name] = value.item()
        return stats

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        vel, amp = batch

        # Forward generators
        fake_amp = self.amp_gen(vel)
        fake_vel = self.vel_gen(amp)

        # Get evaluation performances
        amp_stats = self.eval_with_metrics(
            pred=fake_amp, 
            target=amp,
            metrics=deepcopy(self.eval_metrics["amp_model"])
        )

        vel_stats = amp_stats = self.eval_with_metrics(
            pred=fake_vel, 
            target=vel,
            metrics=deepcopy(self.eval_metrics["vel_model"])
        )

        # log all metrics
        stats = {}
        prefix = "evaluation/amplitude/generator/"
        for name, value in amp_stats.items():
            name = prefix + name
            stats[name] = value

        prefix = "evaluation/velocity/generator/"
        for name, value in vel_stats.items():
            name = prefix + name
            stats[name] = value

        self.log_dict(stats, prog_bar=False)

        # save output for logging
        self.saved_output = {
            "pred_amp": fake_amp.detach(),
            "true_amp": amp,
            "pred_vel": fake_vel.detach(),
            "true_vel": vel,
        }

    def lr_scheduler_step(self, epoch):
        # Step learning rate schedulers
        lr_schedulers = self.lr_schedulers()
        if isinstance(lr_schedulers, list) or isinstance(lr_schedulers, tuple):
            for scheduler in lr_schedulers:
                if scheduler is not None:
                    scheduler.step(epoch)
        else:
            lr_schedulers.step(epoch)

    def log_image(self, name, tensor):
        # assume tensor is a torch.Tensor with shape (height, width)
        # convert to 3 channels (assuming input tensor is grayscale)
        assert tensor.ndim == 2, "Image logging only work for 2D PyTorch tensors."
        tensor = tensor.detach().cpu()
        # tensor = torch.stack([tensor, tensor, tensor], dim=0)

        # normalize to range [0, 1]
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

        # convert to numpy array and apply colormap
        img_array = tensor.squeeze().cpu().numpy()
        img_cmap = cm.get_cmap('viridis')
        img_colored_array = img_cmap(img_array)

        # log the image to Tensorboard
        self.logger.experiment.add_image(name, img_colored_array, dataformats="HWC")


class InvertibleFWIModel(FWIModel):
    """
    Model framework for invertible architectures for FWI.
    
    The `build_generators` method in the class will 
    build one model that is invertible. The `self.amp_gen`
    and `self.vel_gen` should be two functions for forward
    and inverse.

    """
    def build_generators(self):
        model_name = self.model_config.get("class")
        assert "model_params" in self.model_config, \
            "For invertible architectures, put all the augments in `model_params`."
        self.invertible = timm.create_model(
            model_name,
            **self.model_config.get("model_params", {}),
        )
        self.invertible.train()
        self.amp_gen = self.invertible.vel_to_amp
        self.vel_gen = self.invertible.amp_to_vel

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        # Create optimizers for generators
        params = self.invertible.parameters()
        opt, sch = self.build_optimizers_and_schedulers(
            params, key="generator"
        )
        optimizers.extend(opt)
        schedulers.extend(sch)

        # Create optimizers for amplitude discriminators
        if self.use_amp_disc:
            params = self.amp_disc.parameters()
            opt, sch = self.build_optimizers_and_schedulers(
                params, key="amp_disc"
            )
            optimizers.extend(opt)
            schedulers.extend(sch)

        # Create optimizers for velocity discriminators
        if self.use_vel_disc:
            params = self.amp_disc.parameters()
            opt, sch = self.build_optimizers_and_schedulers(
                params, key="vel_disc"
            )
            optimizers.extend(opt)
            schedulers.extend(sch)
        
        return optimizers, schedulers
    