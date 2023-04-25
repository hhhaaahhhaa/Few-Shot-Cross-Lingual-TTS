import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

import Define
from lightning.collates import TextCollate, SSLPRCollate
from lightning.datasets.language import TextDataset
from lightning.datasets.phoneme_recognition import SSLPRDataset, SSLUnitPseudoLabelDataset, SSLUnitFSCLDataset
from lightning.datasets.phoneme_recognition.MultiTaskSampler import MultiTaskSampler, CustomSamplerDataset
from lightning.datamodules.utils import EpisodicInfiniteWrapper


class SSLPRDataModule(pl.LightningDataModule):
    """
    Train: SSLPRDataset + PRCollate.
    Val: SSLPRDataset + PRCollate.
    Test: TextDataset.
    """
    def __init__(self, data_configs, model_config, train_config, algorithm_config, log_dir, result_dir, dataset_cls=SSLPRDataset):
        super().__init__()
        self.data_configs = data_configs
        self.model_config = model_config
        self.train_config = train_config
        self.algorithm_config = algorithm_config

        self.log_dir = log_dir
        self.result_dir = result_dir
        self.val_step = self.train_config["step"]["val_step"]

        self.collate = SSLPRCollate()
        # self.collate2 = TextCollate()

        self.dataset_cls = dataset_cls

    def setup(self, stage=None):
        if stage in (None, 'fit', 'validate'):
            self.train_datasets = [
                self.dataset_cls(
                # SSLUnitPseudoLabelDataset(
                # SSLUnitFSCLDataset(
                    data_config['subsets']['train'],
                    Define.DATAPARSERS[data_config["name"]],
                    data_config
                ) for data_config in self.data_configs if 'train' in data_config['subsets']
            ]
            self.val_datasets = [
                self.dataset_cls(
                    data_config['subsets']['val'],
                    Define.DATAPARSERS[data_config["name"]],
                    data_config
                ) for data_config in self.data_configs if 'val' in data_config['subsets']
            ]
            self.train_dataset = ConcatDataset(self.train_datasets)
            self.val_dataset = ConcatDataset(self.val_datasets)
            self._train_setup()
            self._validation_setup()

        # if stage in (None, 'test', 'predict'):
        #     self.test_datasets = [
        #         TextDataset(
        #             data_config['subsets']['test'],
        #             Define.DATAPARSERS[data_config["name"]],
        #             data_config
        #         ) for data_config in self.data_configs if 'test' in data_config['subsets']
        #     ]
        #     self.test_dataset = ConcatDataset(self.test_datasets)
        #     self._test_setup()

    def _train_setup(self):
        self.batch_size = self.train_config["optimizer"]["batch_size"]
        train_sampler = MultiTaskSampler(self.train_dataset, self.batch_size // torch.cuda.device_count())
        self.batched_train_dataset = CustomSamplerDataset(self.train_dataset, train_sampler)
        self.batched_train_dataset = EpisodicInfiniteWrapper(self.batched_train_dataset, self.val_step)
    
    def _validation_setup(self):
        self.batch_size = self.train_config["optimizer"]["batch_size"]
        val_sampler = MultiTaskSampler(self.val_dataset, self.batch_size // torch.cuda.device_count())
        self.batched_val_dataset = CustomSamplerDataset(self.val_dataset, val_sampler)

    def _test_setup(self):
        pass

    def train_dataloader(self):
        """Training dataloader, not modified for multiple dataloaders."""
        self.train_loader = DataLoader(
            self.batched_train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=Define.MAX_WORKERS,
            collate_fn=lambda batch: self.collate.collate_fn(False)(batch[0]),
        )
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader, not modified for multiple dataloaders."""
        self.val_loader = DataLoader(
            self.batched_val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: self.collate.collate_fn(False)(batch[0]),
        )
        return self.val_loader

    # def test_dataloader(self):
    #     """Test dataloader"""
    #     self.test_loader = DataLoader(
    #         self.test_dataset,
    #         batch_size=self.batch_size//torch.cuda.device_count(),
    #         shuffle=False,
    #         collate_fn=self.collate2.collate_fn(False, re_id=False),
    #     )
    #     return self.test_loader
