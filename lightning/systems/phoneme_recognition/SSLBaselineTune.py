import torch
import torch.nn as nn
import torch.nn.functional as F

from dlhlp_lib.s3prl import S3PRLExtractor

import Define
from text.define import LANG_ID2SYMBOLS
from lightning.systems.system import System
from lightning.callbacks.phoneme_recognition.baseline_saver import Saver
from lightning.utils.tool import ssl_match_length
from .modules import PRFramewiseLoss
from .downstreams import *
from .heads import *
from .SSLBaseline import training_step_template, validation_step_template


class SSLBaselineTuneSystem(System):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # tests
        self.test_list = {}

    def build_model(self):
        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.downstream = Downstream1(
            self.model_config,
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            specific_layer=Define.LAYER_IDX
        )
        self.head = MultilingualPRHead(LANG_ID2SYMBOLS, d_in=self.model_config["transformer"]["d_model"])
        
        self.loss_func = PRFramewiseLoss()

    def build_optimized_model(self):
        return nn.ModuleList([self.head])

    def build_saver(self):
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
        return saver

    # Tune Interface
    def tune_init(self, *args, **kwargs):
        self.lang_id = self.preprocess_config["lang_id"]
        print("Current language: ", self.lang_id)
        # self.head.heads[f"head-{self.lang_id}"] = nn.Linear(self.head.d_in, len(LANG_ID2SYMBOLS[self.lang_id]))  # dirty

    def common_step(self, batch, batch_idx, train=True):
        labels, repr_info = batch

        self.upstream.eval()
        with torch.no_grad():
            ssl_repr, _ = self.upstream.extract(repr_info["wav"])  # B, L, n_layers, dim
            ssl_repr = ssl_match_length(ssl_repr, labels[5])
            ssl_repr = ssl_repr.detach()
        
        # print(ssl_repr.shape)
        # print(labels[3].shape)

        x = self.downstream(ssl_repr, labels[4].cpu())
       
        output = self.head(x, lang_id=repr_info["lang_id"])
        loss = self.loss_func(labels, output)
        loss_dict = {
            "Total Loss": loss,
        }

        return loss_dict, output

    def training_step(self, batch, batch_idx):
        labels, repr_info = batch
        return training_step_template(self, batch, batch_idx, labels, repr_info)
    
    def validation_step(self, batch, batch_idx):
        labels, repr_info = batch
        return validation_step_template(self, batch, batch_idx, labels, repr_info)


class SSLClusterTuneSystem(SSLBaselineTuneSystem):
    """
    This class only replace the MultilingualPRHead with MultilingualClusterHead.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def build_model(self):
        super().build_model()
        self.head = MultilingualClusterHead(LANG_ID2SYMBOLS, self.model_config["transformer"]["d_model"])
