"""
Reimplementation from paper "PHONEME BOUNDARY DETECTION USING LEARNABLE SEGMENTAL FEATURES, ICASSP 2020"
https://arxiv.org/pdf/2002.04992.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from dlhlp_lib.utils.tool import get_mask_from_lengths

from lightning.systems.system import System
from lightning.callbacks.boundary_detection.saver import Saver
from lightning.model.boundary_classfier import Segmentor


class SegFeatSystem(System):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_rec = 0.0

    def build_configs(self):
        self.bs = self.train_config["optimizer"]["batch_size"]

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
        # print("previous step end to current step start: ", time.time() - self.time_rec)
        # st = time.time()
        spect, seg, phonemes, length = batch[4], batch[7], batch[1], batch[5]
        out = self.segmentor(spect, length, seg)
        # print("Forward time: ", time.time() - st)
        # print(out['pred_scores'])
        # print(out['gt_scores'])
        loss = F.relu(1 + out['pred_scores'] - out['gt_scores']).mean()

        out["boundary"] = torch.zeros((spect.shape[0], batch[6])).to(self.device)
        for i in range(spect.shape[0]):
            for x in out["pred"][i][1:]:
                out["boundary"][i][x - 1] = 1
        # self.time_rec = time.time()

        # Currently not used
        # if self.model_config["use_cls"]:
        #     phn_loss, phn_acc = self.cls_loss(seg, phonemes, out['cls_out'])
        #     loss += self.config.phn_cls * phn_loss
        #     self.phn_acc['train'].update(phn_acc)

        # if self.model_config["use_bin"]:
        #     bin_loss, bin_acc = self.bin_loss(seg, out['bin_out'])
        #     loss += self.config.bin_cls * bin_loss
        #     self.bin_acc['train'].update(bin_acc)
    
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
        self.log_boundary_detection_metric(gt_boundaries, pred_boundaries, mask, stage="Train")

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
        self.log_boundary_detection_metric(gt_boundaries, pred_boundaries, mask, stage="Train")

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
