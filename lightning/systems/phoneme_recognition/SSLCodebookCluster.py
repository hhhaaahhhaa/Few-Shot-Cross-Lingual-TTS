#  Deprecated
import torch
import torch.nn as nn
import torch.nn.functional as F

from dlhlp_lib.s3prl import S3PRLExtractor

from lightning.systems.system import System
from lightning.callbacks.phoneme_recognition.baseline_saver import Saver
import Define
from text.define import LANG_ID2SYMBOLS
# from .modules import BiLSTMDownstream, MultiHeadAttentionCodebook, MultilingualClusterHead, PRFramewiseLoss, OrthoLoss
from lightning.utils.tool import ssl_match_length


class SSLCodebookClusterSystem(System):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        d_mid, codebook_size, d_word_vec, nh = 256, 32, 64, 4  # Fixed experminetal parameters
        self.upstream = S3PRLExtractor("hubert_large_ll60k")
        self.upstream.freeze()
        self.downstream = BiLSTMDownstream(
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            d_out=d_mid,
            specific_layer=Define.LAYER_IDX
        )
        self.codebook = MultiHeadAttentionCodebook(codebook_size, q_dim=d_mid, k_dim=d_mid, v_dim=d_word_vec, num_heads=nh)
        self.head = MultilingualClusterHead(LANG_ID2SYMBOLS, d_word_vec)
        self.loss_func = PRFramewiseLoss()
        self.loss_func2 = OrthoLoss()

        if Define.DEBUG:
            print(self)

    def ortho_loss(self):
        nh = 4
        k = self.codebook.k_banks.view(-1, nh, Define.UPSTREAM_DIM // nh)
        k = k.transpose(0, 1).contiguous()
        return self.loss_func2(k)

    def build_optimized_model(self):
        return nn.ModuleList([self.downstream, self.codebook, self.head])

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

        if Define.DEBUG:
            print(ssl_repr.shape)
            print(labels[3].shape)

        x = self.downstream(ssl_repr, labels[4].cpu())
        x, _ = self.codebook(x)
       
        output = self.head(x, lang_id=repr_info["lang_id"])
        loss = self.loss_func(labels, output)
        ortho_loss = self.ortho_loss()
        loss_dict = {
            "Total Loss": loss + ortho_loss,
            "CE Loss": loss,
            "Ortho Loss": ortho_loss,
        }
            
        return loss_dict, output

    def training_step(self, batch, batch_idx):
        labels, repr_info = batch
        train_loss_dict, predictions = self.common_step(batch, batch_idx, train=True)

        mask = (labels[3] != 0)
        acc = ((labels[3] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
        self.log_dict({"Train/Acc": acc.item()}, sync_dist=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions, '_batch': labels, 'lang_id': repr_info["lang_id"]}

    def validation_step(self, batch, batch_idx):
        labels, repr_info = batch
        val_loss_dict, predictions = self.common_step(batch, batch_idx)

        mask = (labels[3] != 0)
        acc = ((labels[3] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
        self.log_dict({"Val/Acc": acc.item()}, sync_dist=True)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss_dict["Total Loss"], 'output': predictions, '_batch': labels, 'lang_id': repr_info["lang_id"]}
