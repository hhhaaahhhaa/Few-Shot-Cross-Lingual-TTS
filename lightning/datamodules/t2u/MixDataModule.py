import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning.trainer.supporters import CombinedLoader

import Define
from lightning.collates import MixCollate
from lightning.datasets.mix import T2U2SDataset
from ..utils import EpisodicInfiniteWrapper
from .DADataModule import DADataModule


class T2U2SDataModule(pl.LightningDataModule):
    """
    Combine UnitFSCL/T2UDatamodule, this is used for E2E tuning t2u2s.
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

        self.t2u_data_configs, self.u2s_data_configs = self.reparse_data_configs()
        self.collate = MixCollate(self.t2u_data_configs, self.u2s_data_configs)

    def reparse_data_configs(self):        
        u2s_data_configs = []
        for data_config in self.data_configs:
            u2s_config = {}
            for k, v in data_config.items():
                if k != "target":
                    u2s_config[k] = v
            u2s_config.update(data_config["target"])
            u2s_data_configs.append(u2s_config)
        if "pitch" in self.model_config["u2s"] and "energy" in self.model_config["u2s"]:
            for data_config in u2s_data_configs:
                data_config["pitch"] = self.model_config["u2s"]["pitch"]
                data_config["energy"] = self.model_config["u2s"]["energy"]
        return self.data_configs, u2s_data_configs
    
    def setup(self, stage=None):
        spk_refer_wav = (self.model_config["u2s"]["speaker_emb"]
                     in ["dvec", "encoder", "scratch_encoder"])

        if stage in (None, 'fit', 'validate'):
            self.train_datasets, self.val_datasets = [], []
            for t2u_data_config, u2s_data_config in zip(self.t2u_data_configs, self.u2s_data_configs):
                if 'train' in t2u_data_config['subsets']:
                    u2s_args = (
                        u2s_data_config['subsets']['train'],
                        Define.DATAPARSERS[u2s_data_config["name"]],
                        u2s_data_config
                    )
                    u2s_kwargs = {"spk_refer_wav": spk_refer_wav}
                    t2u_args = (
                        t2u_data_config['subsets']['train'],
                        Define.DATAPARSERS[t2u_data_config["name"]],
                        t2u_data_config
                    )
                    self.train_datasets.append(T2U2SDataset(t2u_args, {}, u2s_args, u2s_kwargs))
                
                # For unsupervised units use UnitFSCLDataset, otherwise FastSpeech2Dataset
                if 'val' in t2u_data_config['subsets']:
                    u2s_args = (
                        u2s_data_config['subsets']['val'],
                        Define.DATAPARSERS[u2s_data_config["name"]],
                        u2s_data_config
                    )
                    u2s_kwargs = {"spk_refer_wav": spk_refer_wav}
                    t2u_args = (
                        t2u_data_config['subsets']['val'],
                        Define.DATAPARSERS[t2u_data_config["name"]],
                        t2u_data_config
                    )
                    self.val_datasets.append(T2U2SDataset(t2u_args, {}, u2s_args, u2s_kwargs))
            
            self.train_dataset = ConcatDataset(self.train_datasets)
            self.val_dataset = ConcatDataset(self.val_datasets)
            self._train_setup()
            self._validation_setup()

        if stage in (None, 'test', 'predict'):
            pass

    def _train_setup(self):
        if not isinstance(self.train_dataset, EpisodicInfiniteWrapper):
            # self.batch_size = self.train_ways * (self.train_shots + self.train_queries) * self.meta_batch_size
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


class T2U2SDADataModule(pl.LightningDataModule):
    """
    One batch contains data from T2U2S and data from DA
    """
    def __init__(self, data_configs, model_config, train_config, algorithm_config, log_dir, result_dir):
        super().__init__()
        self.t2u_datamodule = T2U2SDataModule([data_configs[0]], model_config,
                                                train_config, algorithm_config, log_dir, result_dir)
        self.da_datamodule = DADataModule(data_configs[1:], model_config,
                                                train_config, algorithm_config, log_dir, result_dir)

    def setup(self, stage=None):
        self.t2u_datamodule.setup(stage)
        self.da_datamodule.setup(stage)

    def train_dataloader(self):
        return {
            "t2u2s": self.t2u_datamodule.train_dataloader(),
            "da": self.da_datamodule.train_dataloader()
        }

    def val_dataloader(self):
        loaders = {
            "t2u2s": self.t2u_datamodule.val_dataloader(),
            "da": self.da_datamodule.val_dataloader()
        }
        return CombinedLoader(loaders, mode="min_size")
