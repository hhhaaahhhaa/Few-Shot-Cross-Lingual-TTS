import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning.trainer.supporters import CombinedLoader

import Define
from lightning.collates import T2UCollate
from lightning.datasets.t2u import T2UDataset
from ..utils import EpisodicInfiniteWrapper
from .DADataModule import DADataModule


class T2UDataModule(pl.LightningDataModule):
    """
    Train: T2UDataset + T2UCollate.
    Val: T2UDataset + T2UCollate.
    """
    def __init__(self, data_configs, model_config, train_config, algorithm_config, log_dir, result_dir):
        super().__init__()
        self.data_configs = data_configs
        self.model_config = model_config
        self.train_config = train_config
        self.algorithm_config = algorithm_config

        self.log_dir = log_dir
        self.result_dir = result_dir
        self.val_step = self.train_config["step"]["val_step"]

        self.re_id = True
        self.collate = T2UCollate(data_configs)

    def setup(self, stage=None):
        if stage in (None, 'fit', 'validate'):
            self.train_datasets = [
                T2UDataset(
                    data_config['subsets']['train'],
                    Define.DATAPARSERS[data_config["name"]],
                    data_config
                ) for data_config in self.data_configs if 'train' in data_config['subsets']
            ]
            self.val_datasets = [
                T2UDataset(
                    data_config['subsets']['val'],
                    Define.DATAPARSERS[data_config["name"]],
                    data_config
                ) for data_config in self.data_configs if 'val' in data_config['subsets']
            ]
            self.train_dataset = ConcatDataset(self.train_datasets)
            self.val_dataset = ConcatDataset(self.val_datasets)
            self._train_setup()
            self._validation_setup()

        if stage in (None, 'test', 'predict'):
            self.test_datasets = [
                T2UDataset(
                    data_config['subsets']['test'],
                    Define.DATAPARSERS[data_config["name"]],
                    data_config
                ) for data_config in self.data_configs if 'test' in data_config['subsets']
            ]
            self.test_dataset = ConcatDataset(self.test_datasets)
            self._test_setup()

    def _train_setup(self):
        if not isinstance(self.train_dataset, EpisodicInfiniteWrapper):
            self.batch_size = self.train_config["optimizer"]["batch_size"]
            self.train_dataset = EpisodicInfiniteWrapper(self.train_dataset, self.val_step*self.batch_size)

    def _validation_setup(self):
        pass

    def _test_setup(self):
        pass

    def train_dataloader(self):
        """Training dataloader, not modified for multiple dataloaders."""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size//torch.cuda.device_count(),
            shuffle=True,
            drop_last=True,
            num_workers=Define.MAX_WORKERS,
            collate_fn=self.collate.collate_fn(sort=True, re_id=self.re_id),  # CAUTION: tune does not need re_id
        )
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader, not modified for multiple dataloaders."""
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size//torch.cuda.device_count(),
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=self.collate.collate_fn(sort=True, re_id=self.re_id),
        )
        return self.val_loader


class T2UDADataModule(pl.LightningDataModule):
    """
    One batch contains data from T2U and data from DA
    """
    def __init__(self, data_configs, model_config, train_config, algorithm_config, log_dir, result_dir):
        super().__init__()
        self.t2u_datamodule = T2UDataModule([data_configs[0]], model_config,
                                                train_config, algorithm_config, log_dir, result_dir)
        self.da_datamodule = DADataModule(data_configs[1:], model_config,
                                                train_config, algorithm_config, log_dir, result_dir)

    def setup(self, stage=None):
        self.t2u_datamodule.setup(stage)
        self.da_datamodule.setup(stage)

    def train_dataloader(self):
        return {
            "t2u": self.t2u_datamodule.train_dataloader(),
            "da": self.da_datamodule.train_dataloader()
        }

    def val_dataloader(self):
        loaders = {
            "t2u": self.t2u_datamodule.val_dataloader(),
            "da": self.da_datamodule.val_dataloader()
        }
        return CombinedLoader(loaders, mode="min_size")
