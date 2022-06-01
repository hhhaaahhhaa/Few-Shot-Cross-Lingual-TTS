import os
import numpy as np
import json
from functools import partial

import torch
from torch import nn

import pytorch_lightning as pl

from Objects.visualization import MatchingGraphInfo
import transformer.Constants as Constants
from transformer import Encoder, Decoder, PostNet
from transformer.Modules import ScaledDotProductAttention, MultiheadAttention
from text.symbols import symbols
from text.define import LANG_ID2SYMBOLS
import Define


class MultilingualTablePhonemeEmbedding(pl.LightningModule):
    """
    Multilingual embedding table for phonemized input. Concatenate monolingual tables on the fly, 
    therefore one can utilize it when transfer learning (maybe different number of languages).
    """
    def __init__(self, lang_id2symbols, d_word_vec):
        super().__init__()
        self.lang_id2symbols = lang_id2symbols
        self.d_word_vec = d_word_vec

        self.tables = nn.ParameterDict()
        for lang_id, v in lang_id2symbols.items():
            if len(v) > 0:
                w_init = torch.randn(len(v), d_word_vec)
                w_init[Constants.PAD].fill_(0)
                self.tables[f"table-{lang_id}"] = nn.Parameter(w_init)
    
    def get_new_embedding(self, lang_id=None, init=False):
        """
        Return corresponding monolingual table, if lang_id is None, return concatenated table; if init is true, 
        return a random initialized embedding table.
        """
        if lang_id is None:
            tables = []
            for lid, v in self.lang_id2symbols.items():
                if len(v) > 0:
                    if init:
                        table = torch.randn(len(self.lang_id2symbols[lid]), self.d_word_vec).to(self.device)
                        table[Constants.PAD].fill_(0)
                    else:
                        table = self.tables[f"table-{lid}"].clone()
                    tables.append(table)
            return torch.cat(tables, dim=0)
        else:
            if init:
                table = torch.randn(len(self.lang_id2symbols[lid]), self.d_word_vec).to(self.device)
                table[Constants.PAD].fill_(0)
            else:
                table = self.tables[f"table-{lang_id}"].clone()
            return table


class MultiHeadAttentionCodebook(pl.LightningModule):
    """
    Multihead Attention with learnable weights and keys, queries are from input.
    """
    def __init__(self, q_dim, v_dim):
        pass


# class MultiHeadSelfAttention(pl.LightningModule):
#     """
#     Standard MHSA module.
#     """
#     def __init__(self, dim, head):
#         super().__init__()
#         self.codebook_config = algorithm_config["adapt"]["phoneme_emb"]
#         self.codebook_size = self.codebook_config["size"]
#         self.d_word_vec = model_config["transformer"]["encoder_hidden"]
#         self.num_heads = 4
#         assert self.d_word_vec % self.num_heads == 0

#         self.emb_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

#         if Define.UPSTREAM != "mel" and Define.UPSTREAM is not None:
#             self.weight_raw = nn.Parameter(torch.zeros(1, 1, 25, 1))
            
#             # specific layer
#             if Define.LAYER_IDX is not None:
#                 last_hidden = torch.ones(1, 1, 25, 1) * float('-inf')
#                 last_hidden[0][0][Define.LAYER_IDX][0] = 10.0
#                 self.weight_raw = nn.Parameter(last_hidden)
#                 self.weight_raw.requires_grad = False

#         self.d_feat = Define.UPSTREAM_DIM
#         self.q_linear = nn.Linear(self.d_feat, self.d_word_vec)

#         # att(feats, att_banks) -> token_id weights -> emb_banks
#         self.att_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

#         self.attention = MultiheadAttention(temperature=(self.d_word_vec // self.num_heads) ** 0.5)

#     def get_new_embedding(self, ref, *args, **kwargs):
#         """
#         ref: 
#             Sup: Tensor with size (B=1, vocab_size, 25, representation_dim).
#             Unsup: Tensor with size (B, L, 25, representation_dim).
#         """
#         B = ref.shape[0]
#         try:
#             assert ref.device == self.device
#         except:
#             ref = ref.to(device=self.device)
#         ref[ref != ref] = 0

#         if Define.UPSTREAM != "mel" and Define.UPSTREAM is not None:
#             weighted_sum = torch.nn.functional.softmax(self.weight_raw, dim=2) * ref
#             ref = weighted_sum.sum(dim=2)
#         q = self.q_linear(ref).view(B, -1, self.num_heads, self.d_word_vec // self.num_heads)
#         q = q.transpose(1, 2).contiguous()  # B x nH x vocab_size x dword // nH
#         k = self.att_banks.view(-1, self.num_heads, self.d_word_vec // self.num_heads)
#         k = k.transpose(0, 1).unsqueeze(0).contiguous()  # 1 x nH x codebook_size x dword // nH
#         v = self.emb_banks.view(-1, self.num_heads, self.d_word_vec // self.num_heads)
#         v = v.transpose(0, 1).unsqueeze(0).contiguous()
#         weighted_embedding, attn = self.attention(q, k, v)
#         weighted_embedding = weighted_embedding.transpose(1, 2).contiguous().view(B, -1, self.d_word_vec)
#         # print(torch.sum(self.att_banks), torch.sum(self.emb_banks))
        
#         return weighted_embedding



class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Tensor with shape B, N_q, dim_q.
            k: Tensor with shape B, N_k, dim_k=dim_q.
            v: Tensor with shape B, N_v=N_k, dim_v.
        
        Return:
            output: Tensor with shape B, N_q, dim_v.
            attn: Tensor with shape B, N_q, N_k. 
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiheadAttention(nn.Module):
    """ Multihead Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=3)

    def forward(self, q, k, v, mask=None):
        # input shape is B, nH, N, dim
        attn = q @ k.transpose(2, 3)
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -torch.inf)

        attn = self.softmax(attn)
        output = attn @ v

        return output, attn


class SoftMultiAttCodebook(pl.LightningModule):
    def __init__(self, model_config, algorithm_config):
        super().__init__()
        # Multihead Attention layer (4 heads)
        #   key: att_banks
        #   value: emb_banks
        #   query: refs
        self.codebook_config = algorithm_config["adapt"]["phoneme_emb"]
        self.codebook_size = self.codebook_config["size"]
        self.d_word_vec = model_config["transformer"]["encoder_hidden"]
        self.num_heads = 4
        assert self.d_word_vec % self.num_heads == 0

        self.emb_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

        if Define.UPSTREAM != "mel" and Define.UPSTREAM is not None:
            self.weight_raw = nn.Parameter(torch.zeros(1, 1, 25, 1))
            
            # specific layer
            if Define.LAYER_IDX is not None:
                last_hidden = torch.ones(1, 1, 25, 1) * float('-inf')
                last_hidden[0][0][Define.LAYER_IDX][0] = 10.0
                self.weight_raw = nn.Parameter(last_hidden)
                self.weight_raw.requires_grad = False

        self.d_feat = Define.UPSTREAM_DIM
        self.q_linear = nn.Linear(self.d_feat, self.d_word_vec)

        # att(feats, att_banks) -> token_id weights -> emb_banks
        self.att_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

        self.attention = MultiheadAttention(temperature=(self.d_word_vec // self.num_heads) ** 0.5)

    def get_new_embedding(self, ref, *args, **kwargs):
        """
        ref: 
            Sup: Tensor with size (B=1, vocab_size, 25, representation_dim).
            Unsup: Tensor with size (B, L, 25, representation_dim).
        """
        B = ref.shape[0]
        try:
            assert ref.device == self.device
        except:
            ref = ref.to(device=self.device)
        ref[ref != ref] = 0

        if Define.UPSTREAM != "mel" and Define.UPSTREAM is not None:
            weighted_sum = torch.nn.functional.softmax(self.weight_raw, dim=2) * ref
            ref = weighted_sum.sum(dim=2)
        q = self.q_linear(ref).view(B, -1, self.num_heads, self.d_word_vec // self.num_heads)
        q = q.transpose(1, 2).contiguous()  # B x nH x vocab_size x dword // nH
        k = self.att_banks.view(-1, self.num_heads, self.d_word_vec // self.num_heads)
        k = k.transpose(0, 1).unsqueeze(0).contiguous()  # 1 x nH x codebook_size x dword // nH
        v = self.emb_banks.view(-1, self.num_heads, self.d_word_vec // self.num_heads)
        v = v.transpose(0, 1).unsqueeze(0).contiguous()
        weighted_embedding, attn = self.attention(q, k, v)
        weighted_embedding = weighted_embedding.transpose(1, 2).contiguous().view(B, -1, self.d_word_vec)
        # print(torch.sum(self.att_banks), torch.sum(self.emb_banks))
        
        return weighted_embedding

    def get_matching(self, ref, lang_id, *args, **kwargs):
        try:
            assert ref.device == self.device
        except:
            ref = ref.to(device=self.device)
        ref[ref != ref] = 0

        if Define.UPSTREAM != "mel" and Define.UPSTREAM is not None:
            weighted_sum = torch.nn.functional.softmax(self.weight_raw, dim=2) * ref
            ref = weighted_sum.sum(dim=2)  # 1 x vocab_size x dword
        q = self.q_linear(ref).view(-1, self.num_heads, self.d_word_vec // self.num_heads)
        q = q.transpose(0, 1).unsqueeze(0).contiguous()  # 1 x nH x vocab_size x dword // nH
        k = self.att_banks.view(-1, self.num_heads, self.d_word_vec // self.num_heads)
        k = k.transpose(0, 1).unsqueeze(0).contiguous()  # 1 x nH x codebook_size x dword // nH
        v = self.emb_banks.view(-1, self.num_heads, self.d_word_vec // self.num_heads)
        v = v.transpose(0, 1).unsqueeze(0).contiguous()
        _, attn = self.attention(q, k, v)

        mask = torch.nonzero(ref.sum(dim=2).squeeze(0), as_tuple=True)  # vocab_size
        attn = attn.squeeze(0)
        infos = []
        for i in range(self.num_heads):
            info = MatchingGraphInfo({
                "title": f"Head-{i}",
                "y_labels": [LANG_ID2SYMBOLS[lang_id][int(m)] for m in mask[0]],
                "x_labels": [str(i) for i in range(1, self.codebook_size + 1)],
                "attn": attn[i][mask].detach().cpu().numpy(),
                "quantized": False,
            })
            infos.append(info)
        
        if Define.UPSTREAM != "mel" and Define.UPSTREAM is not None:
            weights = torch.nn.functional.softmax(self.weight_raw.data, dim=2)
            weight_info = MatchingGraphInfo({
                "title": "Layer Weight",
                "y_labels": ["w"],
                "x_labels": [str(i) for i in range(25)],
                "attn": weights.squeeze(0).squeeze(2).detach().cpu().numpy(),
                "quantized": False,
            })
            infos.append(weight_info)
        return infos


def load_hubert_centroids(paths, size):
    from sklearn.cluster import KMeans
    repr = [np.load(path) for path in paths]
    repr = np.concatenate(repr, axis=0)
    kmeans = KMeans(n_clusters=size, random_state=0).fit(repr)
    centers = kmeans.cluster_centers_
    return torch.from_numpy(centers)


def load_centroids(key, size):
    return _load_centroids(
        [
            f"preprocessed_data/LibriTTS/train-clean-100-clean_{key}-pf.npy",
            f"preprocessed_data/AISHELL-3/train-clean_{key}-pf.npy",
            f"preprocessed_data/GlobalPhone/fr/train-clean_{key}-pf.npy",
            f"preprocessed_data/GlobalPhone/es/train-clean_{key}-pf.npy",
            f"preprocessed_data/GlobalPhone/cz/train-clean_{key}-pf.npy",
            f"preprocessed_data/kss/train-clean_{key}-pf.npy",
        ], size)


def _load_centroids(paths, size):
    from sklearn.cluster import KMeans
    repr = [np.load(path) for path in paths]
    repr = np.concatenate(repr, axis=0)
    kmeans = KMeans(n_clusters=size, random_state=0).fit(repr)
    centers = kmeans.cluster_centers_
    return torch.from_numpy(centers)
