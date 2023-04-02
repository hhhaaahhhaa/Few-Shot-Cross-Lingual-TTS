"""
Reimplementation from paper "PHONEME SEGMENTATION USING SELF-SUPERVISED SPEECH MODELS, SLT 2022"
https://arxiv.org/pdf/2211.01461.pdf
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from dlhlp_lib.s3prl import S3PRLExtractor
from dlhlp_lib.utils.tool import get_mask_from_lengths

import Define
from lightning.model.boundary_classfier import Classifier
from lightning.systems.system import System
from lightning.callbacks.boundary_detection.saver import Saver
from lightning.utils.tool import ssl_match_length


class SSLConvSystem(System):
    """
    Use SSL + convolution downstream to perform phoneme segmentation task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_configs(self):
        self.bs = self.train_config["optimizer"]["batch_size"]

    def build_model(self):
        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.downstream = Classifier(
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            specific_layer=Define.LAYER_IDX,
        )

        self.loss_func = nn.BCEWithLogitsLoss(
            reduction="none", 
            pos_weight=torch.tensor([self.model_config["pos_weight"]]).to(self.device)
        )

    def build_optimized_model(self):
        return nn.ModuleList([self.downstream])

    def build_saver(self):
        self.saver = Saver(self.data_configs, self.model_config, self.log_dir, self.result_dir)
        return self.saver

    def common_step(self, batch, batch_idx, train=True):
        labels, repr_info = batch  # make labels in collate
        self.upstream.eval()
        with torch.no_grad():
            ssl_repr, _ = self.upstream.extract(repr_info["raw_feat"])  # B, L, n_layers, dim
            ssl_repr = ssl_match_length(ssl_repr, repr_info["max_len"].item())  # Unavoidable since we need to match label shape.
            ssl_repr = ssl_repr.detach()
        output = self.downstream(ssl_repr)
        mask = ~get_mask_from_lengths(repr_info["lens"])

        loss = self.loss_func(output, labels[2])
        loss = (loss * (mask)).sum() / (mask).sum()
        loss_dict = {
            "Total Loss": loss,
        }
            
        return loss_dict, output

    def training_step(self, batch, batch_idx):
        labels, repr_info = batch
        train_loss_dict, predictions = self.common_step(batch, batch_idx, train=True)

        # Calculate recall/precision
        mask = ~get_mask_from_lengths(repr_info["lens"])
        correct = ((labels[2] == (predictions >= 0)) * mask).sum()
        acc = correct / mask.sum()
        tp = torch.logical_and(labels[2] == 1, predictions >= 0).sum()
        recall = tp / (labels[2] == 1).sum()
        precision = tp / max(1, ((predictions >= 0) * mask).sum())
        bd = ((predictions >= 0) * mask).sum() / mask.sum()
        self.log_dict({"Train/Acc": acc.item()}, sync_dist=True, batch_size=self.bs)
        self.log_dict({"Train/recall": recall.item()}, sync_dist=True, batch_size=self.bs)
        self.log_dict({"Train/precision": precision.item()}, sync_dist=True, batch_size=self.bs)
        self.log_dict({"Train/bd ratio": bd.item()}, sync_dist=True, batch_size=self.bs)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions, '_batch': labels}
    
    def validation_step(self, batch, batch_idx):
        labels, repr_info = batch
        val_loss_dict, predictions = self.common_step(batch, batch_idx, train=True)

        # Calculate recall/precision
        mask = ~get_mask_from_lengths(repr_info["lens"])
        correct = ((labels[2] == (predictions >= 0)) * mask).sum()
        acc = correct / mask.sum()
        tp = torch.logical_and(labels[2] == 1, predictions >= 0).sum()
        recall = tp / (labels[2] == 1).sum()
        precision = tp / ((predictions >= 0) * mask).sum()
        self.log_dict({"Val/Acc": acc.item()}, sync_dist=True, batch_size=self.bs)
        self.log_dict({"Val/recall": recall.item()}, sync_dist=True, batch_size=self.bs)
        self.log_dict({"Val/precision": precision.item()}, sync_dist=True, batch_size=self.bs)

        if batch_idx == 0:
            layer_weights = F.softmax(self.downstream.weighted_sum.weight_raw, dim=0)
            self.saver.log_layer_weights(self.logger, layer_weights.data, self.global_step + 1, "val")

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': labels}

    def on_save_checkpoint(self, checkpoint):
        """ (Hacking!) Remove pretrained weights in checkpoint to save disk space. """
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k in state_dict:
            if k.split('.')[0] in ["upstream"]:
                continue
            new_state_dict[k] = state_dict[k]
        checkpoint["state_dict"] = new_state_dict

        return checkpoint
