import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

from lightning.build import build_all_speakers
from lightning.systems import System
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.callbacks.language.baseline_saver import Saver
from ..plugin.fscl import IFSCLPlugIn, OrigFSCLPlugIn, TransformerFSCLPlugIn
from ..plugin.tm import ITextMatchingPlugIn, TMPlugIn
from lightning.utils.tool import flat_merge_dict
from ..t2u.schedules import mix_schedule, no_schedule, zero_schedule
from text.define import LANG_ID2NAME


def _dual_fastspeech2_class_factory(FSCLPlugInClass: Type[IFSCLPlugIn], TMPlugInClass: Type[ITextMatchingPlugIn], variation="base"):
    class DualSystem(System):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def build_configs(self):
            self.spk_config = {
                "emb_type": self.model_config["speaker_emb"],
                "speakers": build_all_speakers(self.data_configs)
            }
            self.bs = self.train_config["optimizer"]["batch_size"]
        
        def setup(self, stage=None):
            self.tm.codebook_bind(self.fscl.codebook_attention)
            
        def build_model(self):
            self.model = FastSpeech2(self.model_config, spk_config=self.spk_config)
            self.loss_func = FastSpeech2Loss(self.model_config)
            self.fscl = FSCLPlugInClass(self.model_config)
            # this requires fscl class to equip a codebook module
            self.tm = TMPlugInClass(self.data_configs, self.model_config["text_matching"])

        def build_optimized_model(self):
            # Currently optimize tm only for experimental check
            if variation == "base":
                return nn.ModuleList([self.model, self.tm.build_optimized_model()]) + self.fscl.build_optimized_model()
            elif variation == "fix_fscl":
                return nn.ModuleList([self.model, self.tm.build_optimized_model()])
            elif variation == "unsup_tune":
                return nn.ModuleList([self.model, self.tm.build_optimized_model()])
            else:
                raise NotImplementedError

        def build_saver(self):
            self.saver = Saver(self.data_configs, self.model_config, self.log_dir, self.result_dir)
            return self.saver

        def common_u2s_step(self, seg_repr, batch, batch_idx, train=True):
            labels, _ = batch
            output = self.model(labels[2], seg_repr, *(labels[4:]))
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
        
        def common_tm_step(self, seg_repr, batch, batch_idx, train=True):
            labels, _ = batch
            # seg_repr_clustered = self.tm.cluster(seg_repr, lang_args=labels[-1])
            # print(seg_repr.shape, seg_repr_clustered.shape)
            # c_loss = self.tm.cluster_loss_func(seg_repr, seg_repr_clustered, labels[4])
            output = self.tm(labels[3], labels[4])
            # print(seg_repr.shape, output.shape)
            seg_repr = seg_repr.clone().detach()
            m_loss = self.tm.match_loss_func(seg_repr, output, labels[4])

            loss_dict = {
                "Total Loss": m_loss,  # Currently close c_loss (unstable)
                # "Cluster Loss": c_loss,
                "Match Loss": m_loss,
            }
            return loss_dict, output
        
        def mix_ratio(self):
            return zero_schedule(self.global_step + 1)
        
        def mix_aug(self, seg_repr, tm_output, temp=1.0):
            alpha = torch.randn(seg_repr.shape[:-1]).to(self.device)
            alpha = torch.sigmoid(alpha / temp).unsqueeze(-1)
            # print(alpha.shape, seg_repr.shape, tm_output.shape)
            mixed = (1 - alpha * seg_repr) + alpha * tm_output
            return self.mix_ratio() * seg_repr + (1 - self.mix_ratio()) * mixed

        def common_step(self, batch, batch_idx, train=True):
            _, ref_info = batch
            if variation == "fix_fscl":
                with torch.no_grad():
                    seg_repr, attn_fscl = self.fscl.build_segmental_representation([ref_info])
                    seg_repr = seg_repr.detach()
            else:
                seg_repr, _ = self.fscl.build_segmental_representation([ref_info])
            
            tm_loss_dict, tm_output = self.common_tm_step(seg_repr, batch, batch_idx, train)  # be careful that seg_repr collapse
            if not train:
                u2s_loss_dict, u2s_output = self.common_u2s_step(tm_output, batch, batch_idx, train)
            else:
                # Unsup Tune
                # temp = 0.01 + mix_schedule(self.global_step + 1)
                temp = 0.01  # Means that 68.27% of mix ratio is between sigmoid(-1 / temp) and sigmoid(1 / temp)

                seg_repr_aug = self.mix_aug(seg_repr, tm_output, temp=temp)
                u2s_loss_dict, u2s_output = self.common_u2s_step(seg_repr_aug, batch, batch_idx, train)
           
            loss_dict = flat_merge_dict({
                "U2S": u2s_loss_dict,
                "TM": tm_loss_dict
            })

            loss_dict["Total Loss"] = self.mix_ratio() * tm_loss_dict["Total Loss"] + u2s_loss_dict["Total Loss"]
            return loss_dict, u2s_output

        def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
            assert len(batch[0]) == 13, f"data with 13 elements, but get {len(batch)}"
        
        def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
            assert len(batch[0]) == 13, f"data with 13 elements, but get {len(batch)}"
        
        def training_step(self, batch, batch_idx):
            train_loss_dict, output = self.common_step(batch, batch_idx, train=True)

            # Log metrics to CometLogger
            loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
            self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
            self.log("MixAug ratio", self.mix_ratio(), sync_dist=True)
            return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': output, '_batch': batch[0]}

        def validation_step(self, batch, batch_idx):
            val_loss_dict, predictions = self.common_step(batch, batch_idx, train=False)

            if batch_idx == 0:
                self.saver.log_2D_tensor(
                    self.logger, torch.sigmoid(self.tm.alpha[:10]).data, self.global_step + 1, "alpha",
                    x_labels=[str(i) for i in range(self.tm.alpha.shape[1])],
                    y_labels=[LANG_ID2NAME[i] for i in range(10)], 
                    stage="val"
                )
           
            # Log metrics to CometLogger
            loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
            self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
            return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': batch[0]}

        def on_save_checkpoint(self, checkpoint):
            checkpoint = self.fscl.on_save_checkpoint(checkpoint, prefix="fscl")
            return checkpoint
        
        def on_load_checkpoint(self, checkpoint: dict) -> None:
            self.test_global_step = checkpoint["global_step"]
            state_dict = checkpoint["state_dict"]
            model_state_dict = self.state_dict()
            is_changed = False
            state_dict_pop_keys = []
            state_dict_remap_keys = []
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
                    if k.startswith("codebook_attention"):  # Checkpoints before PlugIn classes are created
                        k_new = "fscl." + k
                        state_dict_remap_keys.append((k_new, k))
                    
                    if self.local_rank == 0:
                        print(f"Dropping parameter {k}")
                    state_dict_pop_keys.append(k)
                    is_changed = True

            if len(state_dict_remap_keys) > 0:
                for (k_new, k) in state_dict_remap_keys:
                    print(f"Remap parameters from old to new ({k} => {k_new}).")
                    state_dict[k_new] = state_dict[k]

            # modify state_dict format to model_state_dict format
            for k in state_dict_pop_keys:
                state_dict.pop(k)
            for k in model_state_dict:
                if k not in state_dict:
                    if k.startswith("fscl.upstream"):
                        pass
                    else:
                        print("Reinitialized: ", k)
                    state_dict[k] = model_state_dict[k]

            if is_changed:
                checkpoint.pop("optimizer_states", None)

    return DualSystem


def dual_fastspeech2_class_factory(name):
    if name == "orig":
        return _dual_fastspeech2_class_factory(OrigFSCLPlugIn, TMPlugIn)
    elif name == "transformer":
        return _dual_fastspeech2_class_factory(TransformerFSCLPlugIn, TMPlugIn)
    else:
        raise NotImplementedError
