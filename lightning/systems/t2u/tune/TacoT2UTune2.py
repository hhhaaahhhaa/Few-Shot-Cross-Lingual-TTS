import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ..TacoT2U import TacoT2USystem
from ...plugin.fscl import FSCLPlugIn


class TacoT2UTuneSystem(TacoT2USystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tune_init(self, data_configs) -> None:
        assert len(data_configs) == 1, f"Currently only support adapting to one language"
        self.target_lang_id = data_configs[0]["lang_id"]
        print(f"Target Language: {self.target_lang_id}.")



def fscl_class_factory(BaseClass, PlugInClass):  # Class factory
    class TransEmbTuneSystem(BaseClass, PlugInClass): 
        def __init__(self, *args, **kwargs):
            self.fscl_cls = PlugInClass
            super(TransEmbTuneSystem, self).__init__(*args, **kwargs)
    
        def build_model(self):
            super().build_model()
            self.fscl = self.fscl_cls(self.model_config)
            self.fscl.build_model()

        def tune_init(self, data_configs):
            table = self.fscl.generate_embedding_table(data_configs)
            self.embedding_model.tables[f"table-{data_configs[0]['lang_id']}"].copy_(table)
            for p in self.embedding_model.parameters():
                p.requires_grad = True

        def validation_step(self, batch, batch_idx):
            val_loss_dict, predictions, alignment = self.common_step(batch, batch_idx)

            mask = (batch[6] != 0)
            acc = ((batch[6] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
            self.log_dict({"Val/Acc": acc.item()}, sync_dist=True)

            # visualization
            if batch_idx == 0:
                layer_weights = F.softmax(self.codebook_attention.weight_raw.squeeze(0).squeeze(-1), dim=0)
                self.saver.log_layer_weights(self.logger, layer_weights.data, self.global_step + 1, "val")
                self.saver.log_codebook_attention(self.logger, self.attn, batch[9][0].item(), batch_idx, self.global_step + 1, "val")
            
            # Log metrics to CometLogger
            loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
            self.log_dict(loss_dict, sync_dist=True)
            return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, 
                    '_batch': batch, 'symbol_id': batch[10][0], 'alignment': alignment}

        def on_save_checkpoint(self, checkpoint):
            """ (Hacking!) Remove pretrained weights in checkpoint to save disk space. """
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k in state_dict:
                if k.split('.')[0] in ["fscl"]:
                    continue
                new_state_dict[k] = state_dict[k]
            checkpoint["state_dict"] = new_state_dict

            return checkpoint
    
    return TransEmbTuneSystem


TransEmbOrigTuneSystem = fscl_class_factory(TacoT2USystem, FSCLPlugIn)
