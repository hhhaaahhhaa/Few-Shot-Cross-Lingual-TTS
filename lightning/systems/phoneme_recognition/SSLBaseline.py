import torch
import torch.nn as nn
import torch.nn.functional as F

from dlhlp_lib.s3prl import S3PRLExtractor

from lightning.systems.system import System
from lightning.utils.log import pr_loss2dict as loss2dict
from lightning.callbacks.phoneme_recognition.ssl_baseline_saver import Saver
import Define
from text.define import LANG_ID2SYMBOLS
from .modules import BiLSTMDownstream, MultilingualPRHead, MultilingualClusterHead, PRFramewiseLoss
from lightning.utils.tool import ssl_match_length


class SSLBaselineSystem(System):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        self.upstream = S3PRLExtractor("hubert_large_ll60k")
        self.upstream.freeze()
        self.downstream = BiLSTMDownstream(n_in_layers=Define.UPSTREAM_LAYER, upstream_dim=Define.UPSTREAM_DIM, specific_layer=Define.LAYER_IDX)
        self.head = MultilingualPRHead(LANG_ID2SYMBOLS, 256)
        self.loss_func = PRFramewiseLoss()

        if Define.DEBUG:
            print(self)

    def build_optimized_model(self):
        return nn.ModuleList([self.downstream, self.head])

    def build_saver(self):
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
        return saver

    def common_step(self, batch, batch_idx, train=True):
        labels, repr_info = batch

        ssl_repr, _ = self.upstream.extract(repr_info["wav"])  # B, L, n_layers, dim
        ssl_repr = ssl_match_length(ssl_repr, labels[5])
        ssl_repr = ssl_repr.detach()

        if Define.DEBUG:
            print(ssl_repr.shape)
            print(labels[3].shape)

        x = self.downstream(ssl_repr, labels[4].cpu())
       
        output = self.head(x, lang_id=repr_info["lang_id"])
        loss = self.loss_func(labels, output)

        return loss, output

    def training_step(self, batch, batch_idx):
        labels, repr_info = batch
        train_loss, predictions = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss, 'losses': train_loss, 'output': predictions, '_batch': labels, 'lang_id': repr_info["lang_id"]}

    def validation_step(self, batch, batch_idx):
        labels, repr_info = batch
        val_loss, predictions = self.common_step(batch, batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': labels, 'lang_id': repr_info["lang_id"]}


class SSLClusterSystem(SSLBaselineSystem):
    """
    This class only replace the MultilingualPRHead with MultilingualClusterHead.
    I wish to prove that cluster head results in better representation and more compatible with DPDP.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def build_model(self):
        self.upstream = S3PRLExtractor("hubert_large_ll60k")
        self.upstream.freeze()
        self.downstream = BiLSTMDownstream(n_in_layers=Define.UPSTREAM_LAYER, upstream_dim=Define.UPSTREAM_DIM, specific_layer=Define.LAYER_IDX)
        self.head = MultilingualClusterHead(LANG_ID2SYMBOLS, 256)
        self.loss_func = PRFramewiseLoss()

        if Define.DEBUG:
            print(self)
