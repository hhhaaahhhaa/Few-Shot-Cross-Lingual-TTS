from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List

import Define
from ...model.reference_extractor import AverageReprModule
from text.define import LANG_ID2SYMBOLS
from transformer.Modules import MultiheadAttention
from Objects.visualization import MatchingGraphInfo
from lightning.utils.tool import torch_exist_nan


class BiLSTMDownstream(nn.Module):
    """
    Weighted sum + BiLSTM.
    """
    def __init__(self, n_in_layers: int, upstream_dim: int, specific_layer: int=None) -> None:
        super().__init__()
        self.weight_raw = nn.Parameter(torch.zeros(1, 1, n_in_layers, 1))
            
        # specific layer, fix weight_raw during training.
        if specific_layer is not None:
            weights = torch.ones(1, 1, n_in_layers, 1) * float('-inf')
            weights[0][0][specific_layer][0] = 10.0
            self.weight_raw = nn.Parameter(weights)
            self.weight_raw.requires_grad = False

        self.d_out = 256
        self.proj = nn.Linear(upstream_dim, self.d_out)
        self.lstm = nn.LSTM(input_size=self.d_out, hidden_size=self.d_out // 2, 
                                num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, repr):
        """
        Args:
            repr: SSL representation with shape (B, L, n_layers, d_in).
        Return:
            Return tensor with shape (B, L, d_out)
        """
        weighted_sum = F.softmax(self.weight_raw, dim=2) * repr  # B, L, d_in
        x = self.proj(weighted_sum.sum(dim=2))  # B, L, d_out
        x, _ = self.lstm(x)  # B, L, d_out

        return x


class MultilingualPRHead(nn.Module):
    def __init__(self, lang_id2symbols, d_in):
        super().__init__()
        self.lang_id2symbols = lang_id2symbols
        self.d_in = d_in

        self.heads = nn.ModuleDict()
        for lang_id, v in lang_id2symbols.items():
            if len(v) > 0:
                self.heads[f"head-{lang_id}"] = nn.Linear(d_in, len(v))

    def forward(self, x, lang_id: int):
        return self.heads[f"head-{lang_id}"](x)


class MultiHeadAttentionCodebook(nn.Module):
    """
    Multihead Attention with learnable weights and keys, queries are from input.
    """
    def __init__(self, codebook_size:int, q_dim: int, v_dim: int, num_heads: int, temperature: float=None):
        super().__init__()
        self.codebook_size = codebook_size
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.temperature = (self.v_dim // self.num_heads) ** 0.5 if temperature is None else temperature
        assert self.v_dim % self.num_heads == 0

        self.q_linear = nn.Linear(self.q_dim, self.v_dim)
        self.k_banks = nn.Parameter(torch.randn(self.codebook_size, self.v_dim))
        self.v_banks = nn.Parameter(torch.randn(self.codebook_size, self.v_dim))
        self.attention = MultiheadAttention(temperature=self.temperature)

    def forward(self, query, attn_mask=None):
        """
        Args:
            query: Tensor with shape (B, L_q, q_dim).
        Return:
            attn_output: Tensor with shape (B, nH, L_q, q_dim).
            attn_weights: Tensor with shape (B, nH, L_q, v_dim // nH).
        """
        B = query.shape[0]
        q = self.q_linear(query).view(B, -1, self.num_heads, self.v_dim // self.num_heads)
        q = q.transpose(1, 2).contiguous()  # B, nH, L_q, v_dim // nH
        k = self.k_banks.view(-1, self.num_heads, self.v_dim // self.num_heads)
        k = k.transpose(0, 1).unsqueeze(0).contiguous()  # 1, nH, codebook_size, v_dim // nH
        v = self.v_banks.view(-1, self.num_heads, self.v_dim // self.num_heads)
        v = v.transpose(0, 1).unsqueeze(0).contiguous()  # 1, nH, codebook_size, v_dim // nH
        attn_output, attn_weights = self.attention(q, k, v, mask=attn_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, self.v_dim)  # B, L_q, v_dim

        return attn_output, attn_weights


class SoftAttCodebook(pl.LightningModule):
    """
    Weighted sum + MultiHeadAttentionCodebook.
    Use 4 attention heads.
    """
    def __init__(self, model_config, algorithm_config, n_in_layers: int, upstream_dim: int, specific_layer: int=None) -> None:
        super().__init__()
        self.codebook_config = algorithm_config["adapt"]["phoneme_emb"]
        self.codebook_size = self.codebook_config["size"]
        self.d_word_vec = model_config["transformer"]["encoder_hidden"]
        self.num_heads = 4
        self.d_feat = Define.UPSTREAM_DIM

        self.emb_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

        # specific layer, fix weight_raw during training.
        if specific_layer is not None:
            weights = torch.ones(1, 1, n_in_layers, 1) * float('-inf')
            weights[0][0][specific_layer][0] = 10.0
            self.weight_raw = nn.Parameter(weights)
            self.weight_raw.requires_grad = False

        self.attention = MultiHeadAttentionCodebook(self.codebook_size, q_dim=self.d_feat, v_dim=self.d_word_vec, num_heads=self.num_heads)

    def forward(self, repr):
        """
        Args:
            repr: SSL representation with shape (B, L, n_layers, d_in).
        Return:
            Return tensor with shape (B, L, d_out)
        """
        weighted_sum = F.softmax(self.weight_raw, dim=2) * repr
        weighted_sum = weighted_sum.sum(dim=2)  # B, L, d_in
        x, _ = self.attention(weighted_sum)  # B, L, d_out

        return x

    def get_matching(self, repr, lang_id, *args, **kwargs):
        # TODO: Not yet tested, exist tensor shape issue
        weighted_sum = F.softmax(self.weight_raw, dim=2) * repr  # B, L, d_in
        _, attn_weights = self.attention(weighted_sum)

        mask = torch.nonzero(repr.sum(dim=2).squeeze(0), as_tuple=True)  # vocab_size
        attn = attn_weights.squeeze(0)
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

