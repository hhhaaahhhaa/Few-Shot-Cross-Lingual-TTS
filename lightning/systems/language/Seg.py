import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

from lightning.build import build_all_speakers
from lightning.systems import System
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.callbacks.language.baseline_saver import Saver
from ..plugin.fscl import IFSCLPlugIn, OrigFSCLPlugIn, LinearFSCLPlugIn, TransformerFSCLPlugIn


def _seg_fastspeech2_class_factory(FSCLPlugInClass: Type[IFSCLPlugIn]):
    class SegTuneSystem(System):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def build_configs(self):
            self.spk_config = {
                "emb_type": self.model_config["speaker_emb"],
                "speakers": build_all_speakers(self.data_configs)
            }
            self.bs = self.train_config["optimizer"]["batch_size"]
        
        def build_model(self):
            self.model = FastSpeech2(self.model_config, spk_config=self.spk_config)
            self.loss_func = FastSpeech2Loss(self.model_config)
            self.fscl = FSCLPlugInClass(self.model_config)

        def build_optimized_model(self):
            return nn.ModuleList([self.model])

        def build_saver(self):
            self.saver = Saver(self.data_configs, self.model_config, self.log_dir, self.result_dir)
            return self.saver

        def common_step(self, batch, batch_idx, train=True):
            labels, ref_info = batch
            emb_texts, _ = self.fscl.build_segmental_representation([ref_info])
            output = self.model(labels[2], emb_texts, *(labels[4:]))
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

        def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
            assert len(batch[0]) == 13, f"data with 13 elements, but get {len(batch)}"
        
        def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
            assert len(batch[0]) == 13, f"data with 13 elements, but get {len(batch)}"
        
        def training_step(self, batch, batch_idx):
            train_loss_dict, output = self.common_step(batch, batch_idx, train=True)

            # Log metrics to CometLogger
            loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
            self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
            return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': output, '_batch': batch[0]}

        def validation_step(self, batch, batch_idx):
            val_loss_dict, predictions = self.common_step(batch, batch_idx, train=False)

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

    return SegTuneSystem


def seg_fastspeech2_class_factory(name):
    if name == "orig":
        return _seg_fastspeech2_class_factory(OrigFSCLPlugIn)
    elif name == "linear":
        return _seg_fastspeech2_class_factory(LinearFSCLPlugIn)
    elif name == "transformer":
        return _seg_fastspeech2_class_factory(TransformerFSCLPlugIn)
    else:
        raise NotImplementedError
