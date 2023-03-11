import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

import Define
from lightning.collates import GeneralFSCLCollate
from lightning.datasets.language import UnitFSCLDataset
from ..utils import EpisodicInfiniteWrapper


class UnitDataModule(pl.LightningDataModule):
    """
    DataModule that always use UnitFSCLDataset (including validation). Units can be unsupervised or pseudo labels.

    Train: UnitFSCLDataset + LanguageCollate.
    Val: UnitFSCLDataset + LanguageCollate.
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

        self.collate = GeneralFSCLCollate(data_configs)

    def setup(self, stage=None):
        spk_refer_wav = (self.model_config["speaker_emb"]
                     in ["dvec", "encoder", "scratch_encoder"])

        if stage in (None, 'fit', 'validate'):
            self.train_datasets, self.val_datasets = [], []
            for data_config in self.data_configs:
                if 'train' in data_config['subsets']:
                    self.train_datasets.append(
                        UnitFSCLDataset(
                            data_config['subsets']['train'],
                            Define.DATAPARSERS[data_config["name"]],
                            data_config, spk_refer_wav=spk_refer_wav
                        )
                    )
                
                if 'val' in data_config['subsets']:
                    self.val_datasets.append(
                        UnitFSCLDataset(
                            data_config['subsets']['val'],
                            Define.DATAPARSERS[data_config["name"]],
                            data_config, spk_refer_wav=spk_refer_wav
                        )
                    )
            
            self.train_dataset = ConcatDataset(self.train_datasets)
            self.val_dataset = ConcatDataset(self.val_datasets)
            self._train_setup()
            self._validation_setup()

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
            collate_fn=self.collate.collate_fn(False, re_id=True, mode="sup"), 
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
            collate_fn=self.collate.collate_fn(False, re_id=True, mode="sup"),
        )
        return self.val_loader
