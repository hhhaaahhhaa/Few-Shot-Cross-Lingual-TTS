import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict

from dlhlp_lib.s3prl import S3PRLExtractor

import Define
from text.define import LANG_ID2SYMBOLS
from lightning.model.reduction import PhonemeQueryExtractor, segmental_average
from ..language.embeddings import SoftMultiAttCodebook2
from lightning.utils.tool import ssl_match_length


class IFSCLPlugIn(pl.LightningModule):
    """ Interface for FSCLPlugIn """
    def build_model(self, *args, **kwargs):
        raise NotImplementedError
    
    def build_optimized_model(self, *args, **kwargs):
        raise NotImplementedError
    
    def build_embedding_table(self, ref_infos, return_attn=False, *args, **kwargs):
        raise NotImplementedError
    
    def build_segmental_representation(self, ref_infos, *args, **kwargs):
        raise NotImplementedError

    def on_save_checkpoint(self, checkpoint, prefix="fscl", *args, **kwargs):
        raise NotImplementedError    


class OrigFSCLPlugIn(IFSCLPlugIn):
    def __init__(self, model_config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_config = model_config
        self.build_model()

    def build_model(self):
        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)
        self.codebook_attention = SoftMultiAttCodebook2(
            codebook_size=self.model_config["codebook_size"],
            embed_dim=self.model_config["transformer"]["encoder_hidden"],
            num_heads=self.model_config["downstream"]["transformer"]["nhead"],
        )
        self.checkpoint_remove_list = ["upstream"]

    def build_optimized_model(self):
        return nn.ModuleList([self.codebook_attention])

    def build_embedding_table(self, ref_infos, return_attn=False):
        self.upstream.eval()
        hiddens, avg_frames_list, phonemes_list = [], [], []
        for info in ref_infos:
            with torch.no_grad():
                ssl_repr, _ = self.upstream.extract(info["raw_feat"])  # B, L, n_layers, dim
                ssl_repr = ssl_repr.detach()
            hiddens.extend([x1 for x1 in ssl_repr])
            avg_frames_list.extend(info["avg_frames"])
            phonemes_list.extend(info["phonemes"])
        
        table_pre = self.phoneme_query_extractor(hiddens, avg_frames_list, 
                            len(LANG_ID2SYMBOLS[ref_infos[0]["lang_id"]]), phonemes_list)  # 1, n_symbols, dim
        
        table, attn = self.codebook_attention(table_pre, need_weights=return_attn)
        table = table.squeeze(0)  # n_symbols, dim
        table[0].fill_(0)
        return table, attn
    
    def build_segmental_representation(self, ref_infos):
        self.upstream.eval()
        hiddens, avg_frames_list = [], []
        for info in ref_infos:
            with torch.no_grad():
                ssl_repr, _ = self.upstream.extract(info["raw_feat"])  # B, L, n_layers, dim
                ssl_repr = ssl_match_length(ssl_repr, info["max_len"].item())  # Unavoidable since we need to match shape when using segmental forward.
                ssl_repr = ssl_repr.detach()
            hiddens.extend([x1 for x1 in ssl_repr])
            avg_frames_list.extend(info["avg_frames"])
        
        seg_repr = segmental_average(hiddens, avg_frames_list)        
        seg_repr, attn = self.codebook_attention(seg_repr, need_weights=False)  # B, L, dim

        return seg_repr, attn
        
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

    def get_layer_weights(self):
        return F.softmax(self.codebook_attention.weight_raw.squeeze(0).squeeze(-1), dim=0)


from ..language.embeddings import SoftMultiAttCodebook
from ..t2u.downstreams import LinearDownstream
class LinearFSCLPlugIn(IFSCLPlugIn):
    """
    Weighted sum + Linear, which should be identical with OrigFSCLPlugIn
    """
    def __init__(self, model_config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_config = model_config
        self.build_model()

    def build_model(self):
        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)
        self.downstream = LinearDownstream(
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            d_out=self.model_config["transformer"]["encoder_hidden"],
            specific_layer=Define.LAYER_IDX
        )
        self.codebook_attention = SoftMultiAttCodebook(
            codebook_size=self.model_config["codebook_size"],
            embed_dim=self.model_config["transformer"]["encoder_hidden"],
            num_heads=self.model_config["downstream"]["transformer"]["nhead"],
        )
        self.checkpoint_remove_list = ["upstream"]

    def build_optimized_model(self):
        return nn.ModuleList([self.codebook_attention, self.downstream])

    def build_embedding_table(self, ref_infos, return_attn=False):
        self.upstream.eval()
        hiddens, avg_frames_list, phonemes_list = [], [], []
        for info in ref_infos:
            with torch.no_grad():
                ssl_repr, _ = self.upstream.extract(info["raw_feat"])  # B, L, n_layers, dim
                ssl_repr = ssl_repr.detach()
            ssl_repr = self.downstream(ssl_repr)
            hiddens.extend([x1 for x1 in ssl_repr])
            avg_frames_list.extend(info["avg_frames"])
            phonemes_list.extend(info["phonemes"])
            
        table_pre = self.phoneme_query_extractor(hiddens, avg_frames_list, 
                            len(LANG_ID2SYMBOLS[ref_infos[0]["lang_id"]]), phonemes_list)  # 1, n_symbols, dim
        
        table, attn = self.codebook_attention(table_pre, need_weights=return_attn)
        table = table.squeeze(0)  # n_symbols, dim
        table[0].fill_(0)
        return table, attn
    
    def build_segmental_representation(self, ref_infos):
        self.upstream.eval()
        hiddens, avg_frames_list = [], []
        for info in ref_infos:
            with torch.no_grad():
                ssl_repr, _ = self.upstream.extract(info["raw_feat"])  # B, L, n_layers, dim
                ssl_repr = ssl_match_length(ssl_repr, info["max_len"].item())  # Unavoidable since we need to match shape when using segmental forward.
                ssl_repr = ssl_repr.detach()
            ssl_repr = self.downstream(ssl_repr)
            hiddens.extend([x1 for x1 in ssl_repr])
            avg_frames_list.extend(info["avg_frames"])
        
        seg_repr = segmental_average(hiddens, avg_frames_list)        
        seg_repr, attn = self.codebook_attention(seg_repr, need_weights=False)  # B, L, dim

        return seg_repr, attn
        
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
    
    def get_layer_weights(self):
        return F.softmax(self.downstream.weighted_sum.weight_raw, dim=0)


from ..t2u.downstreams import Downstream1
class TransformerFSCLPlugIn(LinearFSCLPlugIn):
    def __init__(self, model_config, *args, **kwargs) -> None:
        super().__init__(model_config, *args, **kwargs)

    def build_model(self):
        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)
        self.downstream = Downstream1(
            self.model_config["downstream"],
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            specific_layer=Define.LAYER_IDX
        )
        self.codebook_attention = SoftMultiAttCodebook(
            codebook_size=self.model_config["codebook_size"],
            embed_dim=self.model_config["transformer"]["encoder_hidden"],
            num_heads=self.model_config["downstream"]["transformer"]["nhead"],
        )
        self.checkpoint_remove_list = ["upstream"]

    def build_embedding_table(self, ref_infos, return_attn=False):
        self.upstream.eval()
        hiddens, avg_frames_list, phonemes_list = [], [], []
        for info in ref_infos:
            with torch.no_grad():
                ssl_repr, _ = self.upstream.extract(info["raw_feat"])  # B, L, n_layers, dim
                ssl_repr = ssl_match_length(ssl_repr, info["max_len"].item())  # Unavoidable since we need to match shape in transformer forward.
                ssl_repr = ssl_repr.detach()
            ssl_repr = self.downstream(ssl_repr, info["lens"].cpu())
            hiddens.extend([x1 for x1 in ssl_repr])
            avg_frames_list.extend(info["avg_frames"])
            phonemes_list.extend(info["phonemes"])
            
        table_pre = self.phoneme_query_extractor(hiddens, avg_frames_list, 
                            len(LANG_ID2SYMBOLS[ref_infos[0]["lang_id"]]), phonemes_list)  # 1, n_symbols, dim
        
        table, attn = self.codebook_attention(table_pre, need_weights=return_attn)
        table = table.squeeze(0)  # n_symbols, dim
        table[0].fill_(0)
        return table, attn
    
    def build_segmental_representation(self, ref_infos):
        self.upstream.eval()
        hiddens, avg_frames_list = [], []
        for info in ref_infos:
            with torch.no_grad():
                ssl_repr, _ = self.upstream.extract(info["raw_feat"])  # B, L, n_layers, dim
                ssl_repr = ssl_match_length(ssl_repr, info["max_len"].item())  # Unavoidable since we need to match shape when using segmental forward.
                ssl_repr = ssl_repr.detach()
            ssl_repr = self.downstream(ssl_repr, info["lens"].cpu())
            hiddens.extend([x1 for x1 in ssl_repr])
            avg_frames_list.extend(info["avg_frames"])
        
        seg_repr = segmental_average(hiddens, avg_frames_list)        
        seg_repr, attn = self.codebook_attention(seg_repr, need_weights=False)  # B, L, dim

        return seg_repr, attn
