import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from text.define import LANG_NAME2ID
from lightning.build import build_all_speakers
from lightning.callbacks import GlobalProgressBar
from lightning.systems.interface import Tunable
from .filter import BaselineFilter
from ..Baseline import BaselineSystem


# pseudo label filter, for Baseline, sample level filtering will be implemented in dataset instead of run-time
class BaselineTuneSystem(BaselineSystem, Tunable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.pl_filter = BaselineFilter
    
    def tune_init(self, data_configs) -> None:
        self.target_lang_id = data_configs[0]["lang_id"]
        print(f"Target Language: {self.target_lang_id}.")

    def build_configs(self):
        super().build_configs()
        # self.pl_filter = BaselineFilter(threshold=self.algorithm_config["threshold"])

    def common_step(self, batch, batch_idx, train=True):
        emb_texts = self.embedding_model(batch[3])
        emb_texts = self.text_encoder(emb_texts, lengths=batch[4])
        if self.use_matching:
            emb_texts, _ = self.text_matching(emb_texts)
        output = self.model(batch[2], emb_texts, *(batch[4:]))

        # pseudo label filter, for Baseline, sample level filtering will be implemented in dataset instead of run-time
        # mask = self.pl_filter.calc(scores, lengths=batch[4])
        # loss = self.loss_func(batch[:-1], output, weights=mask.float())
        
        loss = self.loss_func(batch[:-1], output)
        loss_dict = {
            "Total Loss"       : loss[0],
            "Mel Loss"         : loss[1],
            "Mel-Postnet Loss" : loss[2],
            "Pitch Loss"       : loss[3],
            "Energy Loss"      : loss[4],
            "Duration Loss"    : loss[5],
        }
        return loss_dict, output
    
    def configure_callbacks(self):
        # Checkpoint saver
        save_step = self.train_config["step"]["save_step"]
        checkpoint = ModelCheckpoint(
            dirpath=self.ckpt_dir,
            monitor="Val/Total Loss", mode="min",
            every_n_train_steps=save_step, save_top_k=1,
            filename='best'
        )

        # Early stopping
        early_stopping = EarlyStopping(monitor="Val/Total Loss")

        # Progress bars (step/epoch)
        outer_bar = GlobalProgressBar(process_position=1)

        # Monitor learning rate / gpu stats
        lr_monitor = LearningRateMonitor()
        
        # Save figures/audios/csvs
        saver = self.build_saver()
        if isinstance(saver, list):
            callbacks = [checkpoint, outer_bar, lr_monitor, early_stopping, *saver]
        else:
            callbacks = [checkpoint, outer_bar, lr_monitor, early_stopping, saver]
        return callbacks
