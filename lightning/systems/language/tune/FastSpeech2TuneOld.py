import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from dlhlp_lib.s3prl import S3PRLExtractor

import Define
from text.define import LANG_ID2SYMBOLS
from lightning.model.reduction import PhonemeQueryExtractor
from ..embeddings import SoftMultiAttCodebook2
from .interface import IFastSpeech2TuneSystem
from .utils import generate_reference_info


class BaselineTuneSystem(IFastSpeech2TuneSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tune_init(self, data_configs) -> None:
        assert len(data_configs) == 1, f"Currently only support adapting to one language"
        self.target_lang_id = data_configs[0]["lang_id"]
        print(f"Target Language: {self.target_lang_id}.")


class TransEmbOrigTuneSystem(IFastSpeech2TuneSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        super().build_model()
        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)
        self.codebook_attention = SoftMultiAttCodebook2(
            codebook_size=self.model_config["codebook_size"],
            embed_dim=self.model_config["transformer"]["encoder_hidden"],
            num_heads=self.model_config["downstream"]["transformer"]["nhead"],
        )

    def tune_init(self, data_configs):
        assert len(data_configs) == 1, f"Currently only support adapting to one language"
        print("Generate reference...")
        ref_infos = generate_reference_info(data_configs[0])
        self.target_lang_id = ref_infos[0]["lang_id"]
        print(f"Target Language: {self.target_lang_id}.")

        # Merge all information and perform embedding layer initialization
        print("Embedding initialization...")
        self.cuda()
        self.upstream.eval()
        with torch.no_grad():
            hiddens, avg_frames_list, phonemes_list = [], [], []
            for info in ref_infos:
                ssl_repr, _ = self.upstream.extract(info["raw_feat"])  # B, L, n_layers, dim
                hiddens.extend([x1 for x1 in ssl_repr])

                avg_frames_list.extend(info["avg_frames"])
                phonemes_list.extend(info["phonemes"])
            
            table_pre = self.phoneme_query_extractor(hiddens, avg_frames_list, 
                                len(LANG_ID2SYMBOLS[ref_infos[0]["lang_id"]]), phonemes_list)  # 1, n_symbols, dim
            
            table, attn = self.codebook_attention(table_pre, need_weights=True)
            self.attn = attn

            table = table.squeeze(0)  # n_symbols, dim
            table[0].fill_(0)
            # print(table.shape)
            self.embedding_model.tables[f"table-{ref_infos[0]['symbol_id']}"].copy_(table)
        for p in self.embedding_model.parameters():
            p.requires_grad = True
        self.cpu()

        # # tune part
        # if Define.ADAPART:
        #     for p in self.model.decoder.parameters():
        #         p.requires_grad = False
        #     for p in self.model.mel_linear.parameters():
        #         p.requires_grad = False
        #     for p in self.model.postnet.parameters():
        #         p.requires_grad = False
    
    def validation_step(self, batch, batch_idx):
        val_loss_dict, predictions = self.common_step(batch, batch_idx, train=False)
        # print("valid end")
        # input()
        synth_predictions = self.synth_step(batch, batch_idx)
        
        if batch_idx == 0:
            layer_weights = F.softmax(self.codebook_attention.weight_raw.squeeze(0).squeeze(-1), dim=0)
            self.saver.log_layer_weights(self.logger, layer_weights.data, self.global_step + 1, "val")
            self.saver.log_codebook_attention(self.logger, self.attn, batch[-1][0].item(), batch_idx, self.global_step + 1, "val")

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': batch, 'synth': synth_predictions}
    
    def on_save_checkpoint(self, checkpoint):
        """ (Hacking!) Remove pretrained weights in checkpoint to save disk space. """
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k in state_dict:
            if k.split('.')[0] in ["upstream", "codebook_attention"]:
                continue
            new_state_dict[k] = state_dict[k]
        checkpoint["state_dict"] = new_state_dict

        return checkpoint
    