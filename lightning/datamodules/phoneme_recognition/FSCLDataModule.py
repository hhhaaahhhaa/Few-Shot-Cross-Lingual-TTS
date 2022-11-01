import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

import Define
from lightning.datasets.phoneme_recognition import FSCLDataset
from lightning.datasets.language import few_shot_task_dataset
from lightning.collates.PRFSCLCollate import FSCLCollate
from lightning.utils.tool import seed_all
from ..utils import prefetch_tasks


class FSCLDataModule(pl.LightningDataModule):
    """
    Train: SSLPRDataset + FSCLCollate.
    Val: FSCLDataset + FSCLCollate.
    Test: TextDataset.
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

        # few shot settings
        self.meta_type = self.algorithm_config["adapt"]["type"]

        self.train_ways = self.algorithm_config["adapt"]["train"]["ways"]
        self.train_shots = self.algorithm_config["adapt"]["train"]["shots"]
        self.train_queries = self.algorithm_config["adapt"]["train"]["queries"]

        self.test_ways = self.algorithm_config["adapt"]["test"]["ways"]
        self.test_shots = self.algorithm_config["adapt"]["test"]["shots"]
        self.test_queries = self.algorithm_config["adapt"]["test"]["queries"]

        self.meta_batch_size = self.algorithm_config["adapt"]["train"]["meta_batch_size"]

    def setup(self, stage=None):
        if stage in (None, 'fit', 'validate'):
            self.train_datasets = [
                FSCLDataset(
                    data_config['subsets']['train'],
                    Define.DATAPARSERS[data_config["name"]],
                    data_config
                ) for data_config in self.data_configs if 'train' in data_config['subsets']
            ]
            self.val_datasets = [
                FSCLDataset(
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
        #         TextDataset(data_config['subsets']['test'], data_config) 
        #         for data_config in self.data_configs if 'test' in data_config['subsets']
        #     ]
        #     self.test_dataset = ConcatDataset(self.test_datasets)
        #     self._test_setup()

    def _train_setup(self):
        epoch_length = self.meta_batch_size * self.val_step

        self.train_task_dataset = few_shot_task_dataset(
            self.train_dataset, self.train_ways, self.train_shots, self.train_queries,
            n_tasks_per_label=-1, epoch_length=epoch_length, type=self.meta_type, collate_cls=FSCLCollate
        )

    def _validation_setup(self):
        self.val_task_dataset = few_shot_task_dataset(
            self.val_dataset, self.test_ways, self.test_shots, self.test_queries,
            n_tasks_per_label=8, type=self.meta_type, collate_cls=FSCLCollate
        )
        with seed_all(43):
            self.val_SQids2Tid = prefetch_tasks(self.val_task_dataset, 'val', self.log_dir)

    # def _test_setup(self):
    #     self.test_task_dataset = few_shot_task_dataset(
    #         self.test_dataset, self.test_ways, self.test_shots, self.test_queries,
    #         n_tasks_per_label=1, type=self.meta_type
    #     )
    #     with seed_all(43):
    #         self.test_SQids2Tid = prefetch_tasks(self.test_task_dataset, 'test', self.result_dir)


    def train_dataloader(self):
        """Training dataloader"""
        self.train_loader = DataLoader(
            self.train_task_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=Define.MAX_WORKERS,
            collate_fn=lambda batch: batch,
        )
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        self.val_loader = DataLoader(
            self.val_task_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )
        return self.val_loader


    # def test_dataloader(self):
    #     """Test dataloader"""
    #     self.test_loader = DataLoader(
    #         self.test_task_dataset,
    #         batch_size=1,
    #         shuffle=False,
    #         collate_fn=lambda batch: batch,
    #     )
    #     return self.test_loader