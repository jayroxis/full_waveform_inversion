
#   This file contains some IVT style models
#   and various presets.


import torch
import torch.nn as nn

import functools
from copy import deepcopy

import timm
from timm import create_model
import timm.models.registry as registry
from timm.models.registry import register_model

from .block import *


@register_model
def invertible_transformer(
    amp_chans: int, 
    vel_chans: int, 
    ivt_name: str = "IVT_medium",
    hidden_chans: int = 128,
    conv_hidden_chans: int = 256,
    **kwargs,
):
    # the encoders
    amp_encoder = DownsamplingBlock(
        in_chans=amp_chans, 
        out_chans=hidden_chans, 
        hidden_chans=conv_hidden_chans
    )
    vel_encoder = DownsamplingBlock(
        in_chans=vel_chans, 
        out_chans=hidden_chans, 
        hidden_chans=conv_hidden_chans
    )

    # the decoders
    amp_decoder = UpsamplingBlock(
        in_chans=hidden_chans, 
        out_chans=amp_chans, 
        hidden_chans=conv_hidden_chans
    )
    vel_decoder = UpsamplingBlock(
        in_chans=hidden_chans, 
        out_chans=vel_chans, 
        hidden_chans=conv_hidden_chans
    )

    # the iUNet backbone
    ivt_model = timm.create_model(
        ivt_name,
        input_dim=hidden_chans * 16, 
        output_dim=hidden_chans * 16, 
        mode_1_num_tokens=256,
        mode_2_num_tokens=256,
    )
    model = IVTModel(
        amp_encoder=amp_encoder, 
        vel_encoder=vel_encoder, 
        amp_decoder=amp_decoder, 
        vel_decoder=vel_decoder, 
        ivt_model=ivt_model,
    )
    return model



class IVTModel(nn.Module):
    def __init__(
        self, 
        amp_encoder, 
        vel_encoder, 
        amp_decoder, 
        vel_decoder, 
        ivt_model
    ):
        super(IVTModel, self).__init__()
        self.amp_encoder = amp_encoder
        self.vel_encoder = vel_encoder
        self.amp_decoder = amp_decoder
        self.vel_decoder = vel_decoder
        self.ivt_model = ivt_model
        self.downsample = nn.PixelUnshuffle(4)
        self.upsample = nn.PixelShuffle(4)

    def forward(self, amp, vel):
        amp_emb = self.amp_encoder(amp)
        vel_emb = self.vel_encoder(vel)
        amp_emb = self.downsample(amp_emb)
        vel_emb = self.downsample(vel_emb)
        B_a, L_a, W_a, H_a = amp_emb.shape
        B_v, L_v, W_v, H_v = vel_emb.shape
        amp_emb = amp_emb.flatten(-2).transpose(1, 2)
        vel_emb = vel_emb.flatten(-2).transpose(1, 2)
        token = torch.cat([amp_emb, vel_emb], dim=1)
        emb = self.ivt_model(token)
        amp_emb = emb[:, :self.ivt_model.mode_1_num_tokens]
        vel_emb = emb[:, self.ivt_model.mode_1_num_tokens:]
        amp_emb = amp_emb.transpose(1, 2).reshape(B_a, L_a, W_a, H_a)
        vel_emb = vel_emb.transpose(1, 2).reshape(B_v, L_v, W_v, H_v)
        amp_emb = self.upsample(amp_emb)
        vel_emb = self.upsample(vel_emb)
        vel = self.vel_decoder(amp_emb)
        amp = self.amp_decoder(vel_emb)
        return amp, vel

    def amp_to_vel(self, x):
        x = self.amp_encoder(x)
        x = self.downsample(x)
        B, L, W, H = x.shape
        x = x.flatten(-2).transpose(1, 2)
        x = self.ivt_model(x, full_attention=True)
        x = x.transpose(1, 2).reshape(B, L, W, H)
        x = self.upsample(x)
        x = self.vel_decoder(x)
        return x
    
    def vel_to_amp(self, x):
        x = self.vel_encoder(x)
        x = self.downsample(x)
        B, L, W, H = x.shape
        x = x.flatten(-2).transpose(1, 2)
        x = self.ivt_model(x, full_attention=True)
        x = x.transpose(1, 2).reshape(B, L, W, H)
        x = self.upsample(x)
        x = self.amp_decoder(x)
        return x 
    




# Custom IVT with maximum flexibility
@register_model
def custom_IVT(
    *args, **kwargs
):
    return IVT(*args, **kwargs)


# IVT-Nano Presets
@register_model
def IVT_nano(
    input_dim: int, 
    output_dim: int, 
    mode_1_num_tokens: int,
    mode_2_num_tokens: int,
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = IVT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        mode_1_num_tokens=mode_1_num_tokens,
        mode_2_num_tokens=mode_2_num_tokens,
        d_model=192, 
        nhead=8, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=384,
        num_layers=3,
    )
    return model


# IVT-Tiny Presets
@register_model
def IVT_tiny(
    input_dim: int, 
    output_dim: int, 
    mode_1_num_tokens: int,
    mode_2_num_tokens: int,
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = IVT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        mode_1_num_tokens=mode_1_num_tokens,
        mode_2_num_tokens=mode_2_num_tokens,
        d_model=256, 
        nhead=8, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=512,
        num_layers=6,
    )
    return model


# IVT-Small Presets
@register_model
def IVT_small(
    input_dim: int, 
    output_dim: int, 
    mode_1_num_tokens: int,
    mode_2_num_tokens: int,
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = IVT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        mode_1_num_tokens=mode_1_num_tokens,
        mode_2_num_tokens=mode_2_num_tokens,
        d_model=384, 
        nhead=8, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=768,
        num_layers=6,
    )
    return model



# IVT-Medium Presets
@register_model
def IVT_medium(
    input_dim: int, 
    output_dim: int, 
    mode_1_num_tokens: int,
    mode_2_num_tokens: int,
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = IVT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        mode_1_num_tokens=mode_1_num_tokens,
        mode_2_num_tokens=mode_2_num_tokens,
        d_model=512, 
        nhead=8, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=1024,
        num_layers=8,
    )
    return model


# IVT-Large Presets
@register_model
def IVT_large(
    input_dim: int, 
    output_dim: int, 
    mode_1_num_tokens: int,
    mode_2_num_tokens: int,
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = IVT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        mode_1_num_tokens=mode_1_num_tokens,
        mode_2_num_tokens=mode_2_num_tokens,
        d_model=768, 
        nhead=8, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=1536,
        num_layers=8,
    )
    return model


# IVT-Extra-Large Presets
@register_model
def IVT_xlarge(
    input_dim: int, 
    output_dim: int, 
    mode_1_num_tokens: int,
    mode_2_num_tokens: int,
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = IVT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        mode_1_num_tokens=mode_1_num_tokens,
        mode_2_num_tokens=mode_2_num_tokens,
        d_model=1024, 
        nhead=8, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=2048,
        num_layers=10,
    )
    return model


# IVT-Huge Presets
@register_model
def IVT_huge(
    input_dim: int, 
    output_dim: int, 
    mode_1_num_tokens: int,
    mode_2_num_tokens: int,
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = IVT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        mode_1_num_tokens=mode_1_num_tokens,
        mode_2_num_tokens=mode_2_num_tokens,
        d_model=1280, 
        nhead=16, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=2560,
        num_layers=12,
    )
    return model


# IVT-Gigantic Presets
@register_model
def IVT_gigantic(
    input_dim: int, 
    output_dim: int, 
    mode_1_num_tokens: int,
    mode_2_num_tokens: int,
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = IVT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        mode_1_num_tokens=mode_1_num_tokens,
        mode_2_num_tokens=mode_2_num_tokens,
        d_model=1280, 
        nhead=16, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=2560,
        num_layers=24,
    )
    return model


# Default model settings
_default_cfg = {
    "layerscale": {
        "class": "LayerScale",
    },
    "ff_layer": {
        "class": "nn.Linear",
    },
    "activation": {
        "class": "nn.GELU",
    },
    "dropout": {
        "class": "nn.Dropout",
    },
    "layernorm": {
        "class": "nn.LayerNorm",
    },
    "multihead_attn": {
        "class": "nn.MultiheadAttention",
        "params": {
            "batch_first": True,
        }
    },
    "attn_mask": {
        "class": "InvertibleAttentionMask",
    },
}

@register_model
class LayerScale(nn.Module):
    """
    A scaling layer that scales the output of another 
    layer by a learned scalar value.
    """
    def __init__(self, d_model: int, *args, **kwargs):
        super(LayerScale, self).__init__()

        self.alpha = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * self.alpha.unsqueeze(0).unsqueeze(0)


@register_model
class InvertibleAttentionMask(nn.Module):
    """
    Causal Attention Mask (generation) for GPT Architecture.
    """
    def __init__(self, dropout: float = 0.0, *args, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cpu")
        
    def to(self, device, **kwargs):
        """ Rewrite the behavior of `self.to(device)` """
        self.device = device
        super().to(device, **kwargs)

    def forward(self, size_1, size_2):
        """ Generate a invertible attention mask. """
        mask = torch.zeros(
            size_1 + size_2, 
            size_1 + size_2,
            device=self.device,
        )
        mask[:size_1, size_1:] = - torch.inf
        mask[size_1:, :size_1] = - torch.inf
        return mask



class InvertibleMultiheadAttentionLayer(nn.Module):
    """
    A single layer of the IVT decoder consisting of self-attention 
    and feedforward layers.
    """

    def __init__(
            self, 
            d_model, 
            nhead: int = 8, 
            dropout: float = 0.0, 
            ff_dim: int = None,
            module_config: dict = {},
            **kwargs,
        ):
        """
        Initializes the Invertible Multihead-Attention Layer.

        Args:
        - d_model (int): The number of hidden units in the layer.
        - nhead (int): The number of heads in the multi-head attention layer. 
          Default: 8.
        - dropout (float): The dropout rate to apply. Default: 0.1.
        """
        super(InvertibleMultiheadAttentionLayer, self).__init__()
        
        # register variables
        self.d_model = d_model
        if ff_dim is None:
            ff_dim = 2 * d_model
            self.ff_dim = ff_dim
            
        # init module registry
        self.module_registry = build_module_registry(
            config=module_config,
            default_cfg=_default_cfg,
        )
        LayerScale = self.module_registry["layerscale"]
        FeedForwardLayer = self.module_registry["ff_layer"]
        Activation = self.module_registry["activation"]
        Dropout = self.module_registry["dropout"]
        LayerNorm = self.module_registry["layernorm"]
        MultiheadAttention = self.module_registry["multihead_attn"]

        # self-attention layer
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.norm1 = LayerNorm(d_model)
        self.scale1 = LayerScale(d_model=d_model)

        # feedforward layer
        self.ff = nn.Sequential(
            FeedForwardLayer(d_model, ff_dim),
            Activation(),
            FeedForwardLayer(ff_dim, d_model),
            Dropout(dropout),
        )
        self.norm2 = LayerNorm(d_model)
        self.scale2 = LayerScale(d_model=d_model)

    def forward(self, x, mask=None, target=None):
        """
        Passes the input through the InvertibleMultiheadAttentionLayer.

        Args:
        - x (torch.Tensor): The input tensor of shape 
          (batch_size, sequence_length, d_model).
        - mask (torch.Tensor): An optional mask tensor to apply to 
          the self-attention layer. Default: None.

        Returns:
        - torch.Tensor: The output tensor of shape 
          (batch_size, sequence_length, d_model).
        """
        # if no target, then do self-attention
        if target is None:
            target = x

        # self attention
        residual = target
        x = self.norm1(x)
        x = target + self.scale1(
            self.self_attn(target, x, x, attn_mask=mask.to(x.device))[0]
        )

        # feedforward
        x = self.norm2(x)
        x = x + self.scale2(self.ff(x))

        # residual connection
        x = x + residual
        return x


    
class IVT(nn.Module):
    """
    The IVT model with compatibility to float output for regression.
    """
    def __init__(
            self, 
            output_dim: int, 
            input_dim: int, 
            mode_1_num_tokens: int,
            mode_2_num_tokens: int,
            d_model: int = 512, 
            nhead: int = 8, 
            dropout: float = 0.0, 
            attn_drop: float = 0.0,
            ff_dim: int = None,
            num_layers: int = 3,
            module_config: dict = {},
            **kwargs,
    ):
        """
        Initializes the IVT model.
        """
        super(IVT, self).__init__()
        
        # member variables
        self.mode_1_num_tokens = mode_1_num_tokens
        self.mode_2_num_tokens = mode_2_num_tokens

        # init module registry
        self.module_registry = build_module_registry(
            config=module_config,
            default_cfg=_default_cfg,
        )
        FeedForwardLayer = self.module_registry["ff_layer"]
        Activation = self.module_registry["activation"]
        LayerNorm = self.module_registry["layernorm"]
        AttentionMask = self.module_registry["attn_mask"]

        # embedding layer
        self.embedding = FeedForwardLayer(input_dim, d_model)

        # attention mask
        self.attn_mask = AttentionMask(dropout=attn_drop)

        # decoder layers
        decoder_config = module_config.get("decoder_layer", {})
        self.decoder = nn.ModuleList(
            [
                InvertibleMultiheadAttentionLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    ff_dim=ff_dim,
                    module_config=decoder_config,
                )
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        
        # output layer
        self.fc = nn.Sequential(
            FeedForwardLayer(d_model, d_model),
            Activation(),
            FeedForwardLayer(d_model, output_dim),
        )
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, full_attention=False):
        """
        Passes the input through the IVT float model.

        Args:
        - x (torch.Tensor): The input tensor of shape 
          (batch_size, sequence_length, input_dim).

        Returns:
        - torch.Tensor: The output tensor of shape 
          (batch_size, sequence_length, output_dim).
        """
        # embedding layer
        out = self.embedding(x)

        if full_attention:
            L = x.shape[1]
            attn_mask = torch.ones(L, L)
            attn_mask = (attn_mask != 1)  # False that is allowed to attend
        else:
            # generate attention mask
            attn_mask = self.attn_mask(
                self.mode_1_num_tokens, 
                self.mode_2_num_tokens
            )
            attn_mask = attn_mask.to(x.device).type(x.dtype)

        # decoder blocks
        for block in self.decoder:
            out = block(out, mask=attn_mask)

        # output layer
        out = self.norm(out)
        out = self.fc(out)
        return out
    
    @torch.jit.ignore
    def get_params_group(self, lr=1e-3, weight_decay=1e-4, **kwargs):
        """
        Get the optimizer parameters for training the model.

        Args:
            lr (float): Learning rate for the optimizer. Defaults to 1e-3.
                        weight_decay (float): Weight decay for the optimizer. 
                        Defaults to 1e-4.

        Returns:
            list: A list of dictionaries, where each dictionary specifies 
                  the parameters and optimizer settings for a different parameter group.
        """
        # define the parameter groups for the optimizer
        params_group = [{
            "params": self.parameters(), 
            "lr": float(lr), 
            "weight_decay": float(weight_decay),
            **kwargs
        }]
        return params_group
   





def build_model(model_name: str, *args, **kwargs):
    """
    Safely build a model from timm registry.
    """
    if "model_name" in kwargs:
        raise ValueError(f"Got multiple model_name = {kwargs}.")
    # If given model_name not in timm registry, use default name
    if model_name not in registry._model_entrypoints:
        msg = f"`model_name`: \"{model_name}\" and " + \
                " is not in `timm` model registry.\n" + \
                f"Please make sure you have \"{model_name}\" " + \
                f"registed using in timm registry " + \
                "\"timm.models.registry.register_model\"."
        raise ValueError(msg)
    if "lr" in kwargs:
        lr = float(kwargs.pop("lr"))
    else:
        lr = None
    if "weight_decay" in kwargs:
        weight_decay = float(kwargs.pop("weight_decay"))
    else:
        weight_decay = None
    model = create_model(model_name, *args, **kwargs)
    if lr is not None:
        model.lr = lr
    if weight_decay is not None:
        model.weight_decay = weight_decay
    return model



def build_partial_class(config):
    """
    Build a partial class from configs.

    Args:
        config (dict): it should have a "class" key for class name.
            Apart from that, additional class arguments are passed
            in the "params" key which is optional.
    """

    assert "class" in config, "The input config should be a key \"class\"."
    params = config.get("params", {})
    if config["class"] in registry._model_entrypoints:
        model_class = functools.partial(build_model, config["class"])
    else:
        model_class = eval(config["class"])
    partial_class = functools.partial(model_class, **params)
    return partial_class


def build_module_registry(config: dict, default_cfg: dict = {}):
    """
    Build module registry from a config dictionary.
    Note: it works for Python >= 3.8

    Args:
        config (dict): it should have the following structure:
        ```
        config = {
            "module1": {
                "class1": ModuleClass1,
                "params1": {
                    "param1": ***,
                    "param2": ***,
                    ...
                },
            },
            "module2": {
                "class2": ModuleClass2,
                "params2": {
                    "param1": ***,
                    ...
                },
            },
        }
        ```
        The `params` can be left empty of partially given.

    Returns:
        It returns the module registry which contains a dict
        of partial classes.
    """
    # module registry
    module_registry = {}
    model_cfg = deepcopy(default_cfg)
    model_cfg.update(config)
    for name, cfg in model_cfg.items():
        module_class = build_partial_class(cfg)
        module_registry[name] = module_class
    return module_registry


def get_params_group(model, lr, weight_decay=0.0, **kwargs):
    """
    ! DANGER: do not call this function in a member function 
    """
    # check whether the model has any trainable parameters
    has_grad = False
    if hasattr(model, "parameters"):
        if callable(getattr(model, "parameters")):
            for param in model.parameters():
                if param.requires_grad:
                    has_grad = True
                    break
    if not has_grad:
        return []
    
    # Get visual encoder parameters group
    if hasattr(model, "lr"):
        lr = model.lr
    if hasattr(model, "weight_decay"):
        weight_decay = model.weight_decay

    lr = float(lr)
    weight_decay = float(weight_decay)
    
    if hasattr(model, "get_params_group"):
        params_group = model.get_params_group(
            lr=lr, 
            weight_decay=weight_decay, 
            **kwargs
        )
    else:
        params_group = [{
            "params": model.parameters(), 
            "lr": lr, 
            "weight_decay": weight_decay
        }]
    return params_group
