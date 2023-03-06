import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Type

from ..FastSpeech2 import BaselineSystem
from ...plugin.fscl import IFSCLPlugIn, OrigFSCLPlugIn
from .utils import generate_reference_info


class BaselineTuneSystem(BaselineSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tune_init(self, data_configs) -> None:
        assert len(data_configs) == 1, f"Currently only support adapting to one language"
        self.target_lang_id = data_configs[0]["lang_id"]
        print(f"Target Language: {self.target_lang_id}.")


def fscl_tune_fastspeech2_class_factory(FSCLPlugInClass: Type[IFSCLPlugIn]):
    class TransEmbTuneSystem(BaselineSystem):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def build_model(self):
            super().build_model()
            self.fscl = FSCLPlugInClass(self.model_config)

        def tune_init(self, data_configs):
            assert len(data_configs) == 1, f"Currently only support adapting to one language"
            print("Generate reference...")
            ref_infos = generate_reference_info(data_configs[0])
            self.target_lang_id = ref_infos[0]["lang_id"]
            print(f"Target Language: {self.target_lang_id}.")

            print("Embedding initialization...")
            self.cuda()
            table, attn = self.fscl.build_embedding_table(ref_infos, return_attn=True)
            self.attn = attn
            self.embedding_model.tables[f"table-{ref_infos[0]['symbol_id']}"].copy_(table)
            for p in self.embedding_model.parameters():
                p.requires_grad = True
            self.cpu()
        
        def validation_step(self, batch, batch_idx):
            val_loss_dict, predictions = self.common_step(batch, batch_idx, train=False)
            synth_predictions = self.synth_step(batch, batch_idx)
            
            # TODO: abstraction of logging part
            if batch_idx == 0:
                layer_weights = F.softmax(self.fscl.codebook_attention.weight_raw.squeeze(0).squeeze(-1), dim=0)
                self.saver.log_layer_weights(self.logger, layer_weights.data, self.global_step + 1, "val")
                self.saver.log_codebook_attention(self.logger, self.attn, batch[-1][0].item(), batch_idx, self.global_step + 1, "val")

            # Log metrics to CometLogger
            loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
            self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
            return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': batch, 'synth': synth_predictions}
        
        def on_save_checkpoint(self, checkpoint):
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k in state_dict:
                if k.split('.')[0] in ["fscl"]:
                    continue
                new_state_dict[k] = state_dict[k]
            checkpoint["state_dict"] = new_state_dict

            return checkpoint
    
    return TransEmbTuneSystem


TransEmbOrigTuneSystem = fscl_tune_fastspeech2_class_factory(OrigFSCLPlugIn)
        