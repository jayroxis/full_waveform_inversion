
import torch
from utils.data.datasets import *


class DataModule:
    def __init__(self, config: dict):
        super().__init__()

        self.train_set, self.train_loader = None, None
        self.val_set, self.val_loader = None, None
        self.test_set, self.test_loader = None, None

        if "train_set" in config:
            self.train_set = self._setup_dataset(config["train_set"])
            if "train_loader" in config:
                self.train_loader = self._setup_dataloader(
                    dataset=self.train_set,
                    config=config["train_loader"]
                )

        if "val_set" in config:
            self.val_set = self._setup_dataset(config["val_set"])
            if "val_loader" in config:
                self.val_loader = self._setup_dataloader(
                    dataset=self.val_set,
                    config=config["val_loader"]
                )

        if "test_set" in config:
            self.test_set = self._setup_dataset(config["test_set"])
            if "test_loader" in config:
                self.test_loader = self._setup_dataloader(
                    dataset=self.test_set,
                    config=config["test_loader"]
                )
        
    def _setup_dataset(self, config):
        DATASET = eval(str(config["class"]))
        dataset = DATASET(**config["params"])
        return dataset
    
    def _setup_dataloader(self, dataset, config):
        DATALOADER = eval(str(config["class"]))
        dataloader = DATALOADER(dataset, **config["params"])
        return dataloader