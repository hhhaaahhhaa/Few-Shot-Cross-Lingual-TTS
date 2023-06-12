import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dlhlp_lib.utils.tool import get_mask_from_lengths

import Define
from lightning.systems.interface import Tunable
from .filter import BaselineFilter, FramewiseFilter
from lightning.utils.tool import flat_merge_dict, pad
from ..Semi import SemiSystem


class SemiTuneSystem(SemiSystem, Tunable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pl_filter = FramewiseFilter(threshold=Define.PL_CONF)
        # self.pl_filter = BaselineFilter(threshold=Define.PL_CONF)
    
    def tune_init(self, data_configs) -> None:
        self.target_lang_id = data_configs[0]["lang_id"]
        print(f"Target Language: {self.target_lang_id}.")

    def build_optimized_model(self):
        opt_modules = [self.text_encoder, self.model, self.embedding_model]
        if self.use_matching:
            opt_modules += [self.text_matching]
        return nn.ModuleList(opt_modules)
    
    def _expand(self, seq, d):
        out = list()
        for i, v in enumerate(seq):
            expand_size = d[i].item()
            out.extend([v] * max(int(expand_size), 0))
        out = torch.Tensor(out).bool().to(self.device)

        return out

    def common_r2s_step_ablation(self, repr, batch, batch_idx, train=True, loss_mask=None):
        labels, _ = batch
        output = self.model(labels[2], repr, *(labels[4:]))

        # this is the ablation of FramewiseFilter, use this block only in ablation experiment
        if loss_mask is not None:
            mel_mask = output[7]
            max_len = mel_mask.shape[-1]
            temp = list()
            for mask_seq, expand_target in zip(loss_mask, labels[-2]):
                temp.append(self._expand(mask_seq, expand_target))
            loss_mel_mask = pad(temp, max_len)
            assert loss_mel_mask.shape == mel_mask.shape
            updated_mel_mask = torch.logical_or(loss_mel_mask, mel_mask)
            output = (*output[:7], updated_mel_mask, *output[8:])
        
        loss = self.loss_func(labels[:-1], output)
        loss_dict = {
            "Total Loss"       : loss[0],
            "Mel Loss"         : loss[1],
            "Mel-Postnet Loss" : loss[2],
            "Pitch Loss"       : loss[3],
            "Energy Loss"      : loss[4],
            "Duration Loss"    : loss[5],
        }
        return loss_dict, output
    
    def common_t2r_step(self, batch, batch_idx, train=True, mask=None):
        labels, _ = batch
        emb_texts = self.embedding_model(labels[3])
        output = self.text_encoder(emb_texts, lengths=labels[4], mask=mask)

        loss_dict = {}

        return loss_dict, output

    def common_step(self, batch, batch_idx, train=True):
        labels, ref_info = batch
        with torch.no_grad():
            seg_repr = self.fscl.build_segmental_representation([ref_info])
            mask = self.pl_filter.mask_gen(ref_info["phoneme_score"], lengths=labels[4], durations=labels[-2]).to(self.device)

        if not train:
            t2r_loss_dict, t2r_output = self.common_t2r_step(batch, batch_idx, train, mask=None)  # be careful that seg_repr collapse
        else:
            t2r_loss_dict, t2r_output = self.common_t2r_step(batch, batch_idx, train)

        if self.use_matching:  # Close the gap of distributions by restricting domain
            seg_repr, _ = self.repr_matching(seg_repr)
            t2r_output, _ = self.text_matching(t2r_output)

        if not train:
            u2s_loss_dict, u2s_output = self.common_r2s_step(t2r_output, batch, batch_idx, train)
        else:  # here we use framewise mixing
            mixed_repr = torch.where(mask.unsqueeze(-1), seg_repr, t2r_output)
            u2s_loss_dict, u2s_output = self.common_r2s_step(mixed_repr, batch, batch_idx, train)

        loss_dict = flat_merge_dict({
            "U2S": u2s_loss_dict,
            "TM": t2r_loss_dict
        })

        loss_dict["Total Loss"] = u2s_loss_dict["Total Loss"]

        # Calculate unsup ratio
        length_mask = get_mask_from_lengths(labels[4])
        pl_ratio = torch.logical_and(~length_mask, ~mask).sum() / (~length_mask).sum()
        self.log("PL ratio", pl_ratio.item(), sync_dist=True, batch_size=self.bs)

        return loss_dict, u2s_output
    
    # this is the ablation of FramewiseFilter, use this block only in ablation experiment
    def common_step_ablation(self, batch, batch_idx, train=True):
        labels, ref_info = batch
        with torch.no_grad():
            mask = self.pl_filter.mask_gen(ref_info["phoneme_score"], lengths=labels[4], durations=labels[-2]).to(self.device)

        t2r_loss_dict, t2r_output = self.common_t2r_step(batch, batch_idx, train)
        if not train:
            u2s_loss_dict, u2s_output = self.common_r2s_step(t2r_output, batch, batch_idx, train)
        else:
            u2s_loss_dict, u2s_output = self.common_r2s_step(t2r_output, batch, batch_idx, train, loss_mask=mask)

        loss_dict = flat_merge_dict({
            "U2S": u2s_loss_dict,
            "TM": t2r_loss_dict
        })

        loss_dict["Total Loss"] = u2s_loss_dict["Total Loss"]

        # Calculate unsup ratio
        length_mask = get_mask_from_lengths(labels[4])
        pl_ratio = torch.logical_and(~length_mask, ~mask).sum() / (~length_mask).sum()
        self.log("PL ratio", pl_ratio.item(), sync_dist=True, batch_size=self.bs)

        return loss_dict, u2s_output

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.test_global_step = checkpoint["global_step"]
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        state_dict_pop_keys = []
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    if self.local_rank == 0:
                        print(f"Skip loading parameter: {k}, "
                                    f"required shape: {model_state_dict[k].shape}, "
                                    f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                if self.local_rank == 0:
                    print(f"Dropping parameter {k}")
                state_dict_pop_keys.append(k)
                is_changed = True

        # modify state_dict format to model_state_dict format
        for k in state_dict_pop_keys:
            state_dict.pop(k)
        for k in model_state_dict:
            if k not in state_dict:
                if "fscl.upstream" not in k:
                    print("Reinitialize: ", k)
                state_dict[k] = model_state_dict[k]

        if is_changed:
            checkpoint.pop("optimizer_states", None)
