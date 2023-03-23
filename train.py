
import os
import yaml
import json
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from engine.data import DataModule
from engine.arguments import ArgumentParserModule
from engine.model import FWIModel, InvertibleFWIModel


def main():
    # Parse Arguments
    arg_parser = ArgumentParserModule()
    args = arg_parser.parse_args()

    # Load configuration file
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)
    
    if "fno_2d" in json.dumps(configs["model"]):
        assert len(args.gpu) == 1, "FNO only works with one GPU."

    # Create FWI model from configs
    invertible = configs["model"].get("invertible", False)
    if invertible:
        model = InvertibleFWIModel(config=configs)
    else:
        model = FWIModel(config=configs)
    
    # Get data module
    data = DataModule(config=configs["data"])

    # Set up logger
    save_dir = configs["training"].get("save_dir", "./")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tb_logger = TensorBoardLogger(save_dir=save_dir)

    # Create PyTorch Lightning Trainer
    trainer = pl.Trainer(
        devices=args.gpu,
        logger=tb_logger,
        **configs["training"]["params"]
    )

    # Train the model
    trainer.fit(
        model, 
        train_dataloaders=data.train_loader, 
        val_dataloaders=data.val_loader
    )



if __name__ == "__main__":
    main()