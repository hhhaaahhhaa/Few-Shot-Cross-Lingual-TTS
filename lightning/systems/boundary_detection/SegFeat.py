"""
Reimplementation from paper "PHONEME BOUNDARY DETECTION USING LEARNABLE SEGMENTAL FEATURES, ICASSP 2020"
https://arxiv.org/pdf/2002.04992.pdf
"""
import json
from collections import OrderedDict
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from loguru import logger
# from pytorch_lightning import LightningModule
# from torch import optim
# from torch.utils.data import DataLoader

# from dataloader import (BuckeyeDataset, TimitDataset, collate_fn_padd,
#                         phoneme_lebels_to_frame_labels,
#                         segmentation_to_binary_mask)
# from model import Segmentor
# from utils import PrecisionRecallMetricMultiple, StatsMeter

from dlhlp_lib.utils.tool import get_mask_from_lengths

from lightning.systems.system import System
from lightning.callbacks.boundary_detection.saver import Saver
from lightning.model.boundary_classfier import Segmentor


class SegFeatSystem(System):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        self.segmentor = Segmentor(self.model_config)

    def build_optimized_model(self):
        return nn.ModuleList([self.segmentor])

    def build_saver(self):
        self.saver = Saver(self.data_configs, self.model_config, self.log_dir, self.result_dir)
        return self.saver

    def common_step(self, batch, batch_idx, train=True):
        """training_step
        forward 1 training step. calc ranking, phoneme classification
        and boundary classification losses.
        :param data_batch:
        :param batch_i:
        """
        # forward
        spect, dur, phonemes, length, max_len = batch[4], batch[7], batch[1], batch[5], batch[6]
        seg = dur
        out       = self.segmentor(spect, length, seg)
        loss      = F.relu(1 + out['pred_scores'] - out['gt_scores']).mean()

        out["boundary"] = torch.zeros((spect.shape[0], batch[6])).to(self.device)
        for i in range(spect.shape[0]):
            for x in out["pred"][i][1:]:
                out["boundary"][i][x - 1] = 1

        if self.model_config.use_cls:
            phn_loss, phn_acc = self.cls_loss(seg, phonemes, out['cls_out'])
            loss += self.config.phn_cls * phn_loss
            self.phn_acc['train'].update(phn_acc)

        if self.model_config.use_bin:
            bin_loss, bin_acc = self.bin_loss(seg, out['bin_out'])
            loss += self.config.bin_cls * bin_loss
            self.bin_acc['train'].update(bin_acc)
    
        loss_dict = {
            "Total Loss": loss,
        }
            
        return loss_dict, out

    def training_step(self, batch, batch_idx):
        labels = batch
        train_loss_dict, predictions = self.common_step(batch, batch_idx, train=True)

        pred_boundaries = predictions["boundary"]
        gt_boundaries = labels[-1]
        mask = ~get_mask_from_lengths(labels[5])
        self.log_boundary_detection_metric(self, gt_boundaries, pred_boundaries, mask, stage="Train")

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions, '_batch': labels}
    
    def validation_step(self, batch, batch_idx):
        labels = batch
        val_loss_dict, predictions = self.common_step(batch, batch_idx, train=False)

        pred_boundaries = predictions["boundary"]
        gt_boundaries = labels[-1]
        mask = ~get_mask_from_lengths(labels[5])
        self.log_boundary_detection_metric(self, gt_boundaries, pred_boundaries, mask, stage="Train")

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': labels}

    def log_boundary_detection_metric(self, gt_boundaries, pred_boundaries, mask, stage="Train"):
        correct = ((gt_boundaries == (pred_boundaries >= 0)) * mask).sum()
        acc = correct / mask.sum()
        tp = torch.logical_and(gt_boundaries == 1, pred_boundaries >= 0).sum()
        recall = tp / (gt_boundaries == 1).sum()
        precision = tp / max(1, ((pred_boundaries >= 0) * mask).sum())
        bd = ((pred_boundaries >= 0) * mask).sum() / mask.sum()
        self.log_dict({f"{stage}/Acc": acc.item()}, sync_dist=True, batch_size=self.bs)
        self.log_dict({f"{stage}/recall": recall.item()}, sync_dist=True, batch_size=self.bs)
        self.log_dict({f"{stage}/precision": precision.item()}, sync_dist=True, batch_size=self.bs)
        self.log_dict({f"{stage}/bd ratio": bd.item()}, sync_dist=True, batch_size=self.bs)
