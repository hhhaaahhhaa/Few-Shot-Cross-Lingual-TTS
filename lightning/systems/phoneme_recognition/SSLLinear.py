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


class SSLLinearTuneSystem(System):
    """
    Classical linear downstream evaluation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        self.upstream = S3PRLExtractor("hubert_large_ll60k")
        self.upstream.freeze()
        self.downstream = WeightedSumLayer(n_in_layers=Define.UPSTREAM_LAYER, specific_layer=Define.LAYER_IDX)
        self.head = MultilingualPRHead(LANG_ID2SYMBOLS, d_in=Define.UPSTREAM_DIM)
        
        self.loss_func = PRFramewiseLoss()

    def build_optimized_model(self):
        return nn.ModuleList([self.downstream, self.head])

    def build_saver(self):
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
        return saver

    # Tune Interface
    def tune_init(self, *args, **kwargs):
        self.lang_id = self.preprocess_config["lang_id"]
        print("Current language: ", self.lang_id)
        
    def common_step(self, batch, batch_idx, train=True):
        labels, repr_info = batch

        self.upstream.eval()
        with torch.no_grad():
            ssl_repr, _ = self.upstream.extract(repr_info["wav"])  # B, L, n_layers, dim
            ssl_repr = ssl_match_length(ssl_repr, labels[5])
            ssl_repr = ssl_repr.detach()

        # print(ssl_repr.shape)
        # print(labels[3].shape)

        x = self.downstream(ssl_repr, dim=2)
       
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
