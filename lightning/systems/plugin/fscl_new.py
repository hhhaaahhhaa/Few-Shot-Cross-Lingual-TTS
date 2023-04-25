from typing import Type
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict

from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.s3prl import S3PRLExtractor
from dlhlp_lib.common.layers import WeightedSumLayer

import Define
from lightning.systems.interface import FSCLPlugIn, Hookable
from text.define import LANG_ID2SYMBOLS
from lightning.model.codebook import SoftMultiAttCodebook
from lightning.model.upstream import Padding
from lightning.model.downstream import TransformerDownstream
from lightning.model.reduction import PhonemeQueryExtractor, segmental_average
from lightning.utils.tool import ssl_match_length


class SemiFSCLPlugIn(FSCLPlugIn, Hookable):
    def __init__(self, model_config, *args, **kwargs) -> None:
        super().__init__()
        self.model_config = model_config
        self._build_model()
        self._custom_hooks = {}

    def _build_model(self):
        if Define.UPSTREAM == "mel":
            self.upstream = Padding()
            self.featurizer = None
            upstream_dim = AUDIO_CONFIG["mel"]["n_mel_channels"]
        else:
            self.upstream = S3PRLExtractor(Define.UPSTREAM)
            self.featurizer = WeightedSumLayer(
                n_in_layers=self.upstream.n_layers,
                specific_layer=Define.LAYER_IDX
            )
            upstream_dim = self.upstream.dim
            self.upstream.freeze()
        self.downstream1 = TransformerDownstream(  # Extract segmental representation
            self.model_config["downstream"],
            upstream_dim=upstream_dim,
        )
        self.downstream2 = TransformerDownstream(  # Extract phoneme query (deeper)
            self.model_config["downstream"],
            upstream_dim=upstream_dim,
            use_proj=False,
        )

        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)
        self.codebook_attention = SoftMultiAttCodebook(
            codebook_size=self.model_config["codebook_size"],
            embed_dim=self.model_config["dim"],
            num_heads=self.model_config["nhead"],
        )
        self.checkpoint_remove_list = ["upstream"]
    
    def build_optimized_model(self):
        opt_modules = [self.downstream1, self.downstream2, self.codebook_attention]
        if self.featurizer is not None:
            opt_modules += [self.featurizer]
        return nn.ModuleList(opt_modules)

    def extract_hiddens(self, ref_infos):
        hiddens1, hiddens2 = [], []
        for info in ref_infos:
            with torch.no_grad():
                repr, _ = self.upstream.extract(info["raw_feat"])  # B, L, n_layers, dim
                repr = ssl_match_length(repr, info["max_len"].item())
                repr = repr.detach()
            if self.featurizer is not None:
                repr = self.featurizer(repr, dim=2)
            repr1 = self.downstream1(repr, info["lens"].cpu())
            repr2 = self.downstream2(repr1, info["lens"].cpu())
            hiddens1.extend([x1 for x1 in repr1])
            hiddens2.extend([x2 for x2 in repr2])
        return hiddens1, hiddens2

    def build_segmental_representation(self, ref_infos):
        avg_frames_list = []
        for info in ref_infos:
            avg_frames_list.extend(info["avg_frames"])
        hiddens, _ = self.extract_hiddens(ref_infos)
        seg_repr = segmental_average(hiddens, avg_frames_list)

        return seg_repr

    def build_embedding_table(self, ref_infos, return_attn=False):
        avg_frames_list, phonemes_list = [], []
        for info in ref_infos:
            avg_frames_list.extend(info["avg_frames"])
            phonemes_list.extend(info["phonemes"])
        _, hiddens = self.extract_hiddens(ref_infos)
        table_pre = self.phoneme_query_extractor(hiddens, avg_frames_list, 
                            len(LANG_ID2SYMBOLS[ref_infos[0]["lang_id"]]), phonemes_list)  # 1, n_symbols, dim
        table, attn = self.codebook_attention(table_pre, need_weights=return_attn)
        table = table.squeeze(0)  # n_symbols, dim
        table[0].fill_(0)

        return table, attn
    
    def build_both(self, ref_infos, return_attn=False):
        avg_frames_list, phonemes_list = [], []
        for info in ref_infos:
            avg_frames_list.extend(info["avg_frames"])
            phonemes_list.extend(info["phonemes"])
        hiddens1, hiddens2 = self.extract_hiddens(ref_infos)
        seg_repr = segmental_average(hiddens1, avg_frames_list)    

        table_pre = self.phoneme_query_extractor(hiddens2, avg_frames_list, 
                            len(LANG_ID2SYMBOLS[ref_infos[0]["lang_id"]]), phonemes_list)  # 1, n_symbols, dim
        table, attn = self.codebook_attention(table_pre, need_weights=return_attn)
        table = table.squeeze(0)  # n_symbols, dim
        table[0].fill_(0)

        return seg_repr, (table, attn)

    def build_hook(self, key: str):
        pass

    def get_hook(self, key: str):
        if key == "layer_weights":
            return F.softmax(self.featurizer.weight_raw, dim=0)
        else:
            raise NotImplementedError

    def on_save_checkpoint(self, checkpoint, prefix=""):
        """ Remove pretrained weights in checkpoint to save disk space. """
        if prefix != "":
            remove_list = [f"{prefix}.{name}" for name in self.checkpoint_remove_list]
        else:
            remove_list = self.checkpoint_remove_list
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k in state_dict:
            if k.startswith(tuple(remove_list)):
                continue
            new_state_dict[k] = state_dict[k]
        checkpoint["state_dict"] = new_state_dict

        return checkpoint 
