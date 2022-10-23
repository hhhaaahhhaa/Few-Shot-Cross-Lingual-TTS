import torch
import torch.nn as nn
import torch.nn.functional as F

from dlhlp_lib.s3prl import S3PRLExtractor

from lightning.systems.system import System
from lightning.utils.log import pr_loss2dict as loss2dict
from lightning.callbacks.phoneme_recognition.baseline_saver import Saver
import Define
from text.define import LANG_ID2SYMBOLS
from .modules import PRFramewiseLoss
from .downstreams import *
from .heads import *

from lightning.utils.tool import ssl_match_length


class SSLBaselineSystem(System):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.downstream = Downstream1(
            self.model_config,
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            specific_layer=Define.LAYER_IDX
        )
        self.head = MultilingualPRHead(LANG_ID2SYMBOLS, self.model_config["transformer"]["d_model"])
        self.loss_func = PRFramewiseLoss()

    def build_optimized_model(self):
        return nn.ModuleList([self.downstream, self.head])

    def build_saver(self):
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
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


class SSLClusterSystem(SSLBaselineSystem):
    """
    This class only replace the MultilingualPRHead with MultilingualClusterHead.
    I wish to prove that cluster head results in better representation and more compatible with DPDP.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def build_model(self):
        super().build_model()
        self.head = MultilingualClusterHead(LANG_ID2SYMBOLS, self.model_config["transformer"]["d_model"])


"""
Training/Validation step template for phoneme recognition systems.
"""
def training_step_template(pl_module, batch, batch_idx, labels, repr_info):
    train_loss_dict, predictions = pl_module.common_step(batch, batch_idx, train=True)

    mask = (labels[3] != 0)
    acc = ((labels[3] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
    pl_module.log_dict({"Train/Acc": acc.item()}, sync_dist=True)

    # Log metrics to CometLogger
    loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
    pl_module.log_dict(loss_dict, sync_dist=True)
    return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions, '_batch': labels, 'lang_id': repr_info["lang_id"]}


def validation_step_template(pl_module, batch, batch_idx, labels, repr_info):
    val_loss_dict, predictions = pl_module.common_step(batch, batch_idx)

    mask = (labels[3] != 0)
    acc = ((labels[3] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
    pl_module.log_dict({"Val/Acc": acc.item()}, sync_dist=True)

    # Log metrics to CometLogger
    loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
    pl_module.log_dict(loss_dict, sync_dist=True)
    return {'losses': val_loss_dict, 'output': predictions, '_batch': labels, 'lang_id': repr_info["lang_id"]}
