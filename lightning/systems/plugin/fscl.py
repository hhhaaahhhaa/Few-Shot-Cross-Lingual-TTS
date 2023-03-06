import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from dlhlp_lib.s3prl import S3PRLExtractor

import Define
from text.define import LANG_ID2SYMBOLS
from lightning.model.reduction import PhonemeQueryExtractor
from ..language.embeddings import SoftMultiAttCodebook2
from lightning.utils.tool import generate_reference_info


class FSCLPlugIn(pl.LightningModule):
    def __init__(self, model_config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_config = model_config

    def build_model(self):
        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)
        self.codebook_attention = SoftMultiAttCodebook2(
            codebook_size=self.model_config["codebook_size"],
            embed_dim=self.model_config["transformer"]["encoder_hidden"],
            num_heads=self.model_config["downstream"]["transformer"]["nhead"],
        )
    
    def generate_embedding_table(self, data_configs):
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
        self.cpu()

        return table.cpu()
