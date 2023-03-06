import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

import Define
from lightning.collates import DACollate
from lightning.datasets.t2u import DADataset
from ..utils import EpisodicInfiniteWrapper


class DADataModule(pl.LightningDataModule):
    def __init__(self, data_configs, model_config, train_config, algorithm_config, log_dir, result_dir):
        super().__init__()
        self.data_configs = data_configs
        self.model_config = model_config
        self.train_config = train_config
        self.algorithm_config = algorithm_config

        self.log_dir = log_dir
        self.result_dir = result_dir
        self.val_step = self.train_config["step"]["val_step"]

        self.collate = DACollate(data_configs)

    def setup(self, stage=None):
        if stage in (None, 'fit', 'validate'):
            self.train_datasets = [
                DADataset(
                    data_config['subsets']['train'],
                    Define.DATAPARSERS[data_config["name"]],
                    data_config
                ) for data_config in self.data_configs if 'train' in data_config['subsets']
            ]
            self.val_datasets = [
                DADataset(
                    data_config['subsets']['val'],
                    Define.DATAPARSERS[data_config["name"]],
                    data_config
                ) for data_config in self.data_configs if 'val' in data_config['subsets']
            ]
            self.train_dataset = ConcatDataset(self.train_datasets)
            self.val_dataset = ConcatDataset(self.val_datasets)
            self._train_setup()
            self._validation_setup()
    
    def _generate_sample_weights(self, dataset: ConcatDataset):
        w_tot = 10000
        res = []
        cum = 0
        for i, ds in enumerate(dataset.datasets):
            if i == 0:
                res.extend([w_tot / len(ds)] * len(ds))
            else:
                cum += len(ds)
        res.extend([w_tot / cum] * cum)
        return res
      
    def _train_setup(self):
        if not isinstance(self.train_dataset, EpisodicInfiniteWrapper):
            ws = self._generate_sample_weights(self.train_dataset)
            self.batch_size = 32  # fixed
            self.train_dataset = EpisodicInfiniteWrapper(self.train_dataset, self.val_step*self.batch_size, weights=ws)

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
            collate_fn=self.collate.collate_fn(),
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
            collate_fn=self.collate.collate_fn(),
        )
        return self.val_loader
