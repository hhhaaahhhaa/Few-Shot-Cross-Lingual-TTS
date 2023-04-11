from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from dlhlp_lib.s3prl import S3PRLExtractor
from dlhlp_lib.common.layers import WeightedSumLayer

import Define
from lightning.build import build_id2symbols
from lightning.systems.system import System
from lightning.callbacks.phoneme_recognition.baseline_saver import Saver
from lightning.utils.tool import ssl_match_length
from .loss import PRFramewiseLoss
from .heads import MultilingualPRHead
from .SSLBaseline import training_step_template, validation_step_template


class SSLLinearSystem(System):
    """
    Classical linear downstream evaluation.
    """

    def __init__(self, *args, **kwargs):
        self.upstream_freeze = True
        super().__init__(*args, **kwargs)

    def build_model(self):
        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.downstream = WeightedSumLayer(
            n_in_layers=Define.UPSTREAM_LAYER, specific_layer=Define.LAYER_IDX)
        self.head = MultilingualPRHead(
            build_id2symbols(self.data_configs), d_in=Define.UPSTREAM_DIM)
        
        self.loss_func = PRFramewiseLoss()

    def build_optimized_model(self):
        return nn.ModuleList([self.downstream, self.head])

    def build_saver(self):
        saver = Saver(self.data_configs, self.log_dir, self.result_dir)
        return saver

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
        loss = self.loss_func(labels[3], output)
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

    def on_save_checkpoint(self, checkpoint):
        """ (Hacking!) Remove pretrained weights in checkpoint to save disk space. """
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k in state_dict:
            if k.split('.')[0] == "upstream":
                continue
            new_state_dict[k] = state_dict[k]
        checkpoint["state_dict"] = new_state_dict

        return checkpoint
