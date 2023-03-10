import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from typing import Optional


class MultilingualEmbedding(nn.Module):
    def __init__(self, id2symbols, dim: int, padding_idx: int=0):
        super().__init__()
        self.id2symbols = id2symbols
        self.dim = dim
        self.padding_idx = padding_idx

        self.tables = nn.ParameterDict()
        for symbol_id, v in id2symbols.items():
            if len(v) > 0:
                w_init = torch.randn(len(v), dim)
                std = sqrt(2.0 / (len(v) + dim))
                val = sqrt(3.0) * std  # uniform bounds for std
                w_init.uniform_(-val, val)
                w_init[padding_idx].fill_(0)
                self.tables[f"table-{symbol_id}"] = nn.Parameter(w_init)

    def forward(self, x, symbol_id: Optional[str]=None):
        if symbol_id is None:
            # for k, p in self.tables.items():
            #     print(k, p.shape)
            concat_tables = torch.cat([p for p in self.tables.values()], dim=0)
            return F.embedding(x, concat_tables, padding_idx=self.padding_idx)
        return F.embedding(x, self.tables[f"table-{symbol_id}"], padding_idx=self.padding_idx)


# Matching old one
import Define
from transformer.Modules import MultiheadAttention
class SoftMultiAttCodebook(nn.Module):
    def __init__(self, codebook_size, embed_dim, num_heads):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_word_vec = embed_dim
        self.num_heads = num_heads
        assert self.d_word_vec % self.num_heads == 0

        self.emb_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

        # att(feats, att_banks) -> token_id weights -> emb_banks
        self.att_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

        self.attention = MultiheadAttention(temperature=(self.d_word_vec // self.num_heads) ** 0.5)

    def forward(self, ref, need_weights=False):
        """
        ref: Tensor with size (B, L, representation_dim) or (1, vocab_size, representation_dim).
        """
        ref[ref != ref] = 0
        B = ref.shape[0]

        q = ref.view(B, -1, self.num_heads, self.d_word_vec // self.num_heads)
        q = q.transpose(1, 2).contiguous()  # 1 x nH x vocab_size x dword // nH
        k = self.att_banks.view(-1, self.num_heads, self.d_word_vec // self.num_heads)
        k = k.transpose(0, 1).unsqueeze(0).contiguous()  # 1 x nH x codebook_size x dword // nH
        v = self.emb_banks.view(-1, self.num_heads, self.d_word_vec // self.num_heads)
        v = v.transpose(0, 1).unsqueeze(0).contiguous()
        weighted_embedding, attn = self.attention(q, k, v)
        weighted_embedding = weighted_embedding.transpose(1, 2).contiguous().view(B, -1, self.d_word_vec)

        # print(torch.sum(self.att_banks), torch.sum(self.emb_banks))
        
        if need_weights:
            return weighted_embedding, attn
        else:
            return weighted_embedding, None


class SoftMultiAttCodebook2(nn.Module):
    def __init__(self, codebook_size, embed_dim, num_heads):
        super().__init__()
        super().__init__()
        self.codebook_size = codebook_size
        self.d_word_vec = embed_dim
        self.num_heads = num_heads
        assert self.d_word_vec % self.num_heads == 0

        self.emb_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

        # att(feats, att_banks) -> token_id weights -> emb_banks
        self.att_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

        self.attention = MultiheadAttention(temperature=(self.d_word_vec // self.num_heads) ** 0.5)

        if Define.UPSTREAM != "mel" and Define.UPSTREAM is not None:            
            if Define.LAYER_IDX is not None:
                weights = torch.ones(1, Define.UPSTREAM_LAYER, 1) * float('-inf')
                weights[0][Define.LAYER_IDX][0] = 10.0
                self.weight_raw = nn.Parameter(weights)
                self.weight_raw.requires_grad = False
            else:
                self.weight_raw = nn.Parameter(torch.ones(1, Define.UPSTREAM_LAYER, 1))

        self.q_linear = nn.Linear(Define.UPSTREAM_DIM, self.d_word_vec)

    def forward(self, ref, need_weights=False):
        """
        ref: Tensor with size (B, L, n_layer, representation_dim) or (1, vocab_size, n_layer, representation_dim).
        """
        ref[ref != ref] = 0
        B = ref.shape[0]
        # print("codebook")
        # print(ref.shape)

        # print("Phoneme Query Sum1")
        # print(ref.shape, ref.sum())

        if Define.UPSTREAM != "mel" and Define.UPSTREAM is not None:
            weighted_sum = torch.nn.functional.softmax(self.weight_raw.unsqueeze(0), dim=2) * ref
            ref = weighted_sum.sum(dim=2)

        # print("Phoneme Query Sum2")
        # print(ref.shape, ref.sum())
        # input()

        q = self.q_linear(ref).view(B, -1, self.num_heads, self.d_word_vec // self.num_heads)
        q = q.transpose(1, 2).contiguous()  # B x nH x (L or vocab_size) x dword // nH
        k = self.att_banks.view(-1, self.num_heads, self.d_word_vec // self.num_heads)
        k = k.transpose(0, 1).unsqueeze(0).contiguous()  # 1 x nH x codebook_size x dword // nH
        v = self.emb_banks.view(-1, self.num_heads, self.d_word_vec // self.num_heads)
        v = v.transpose(0, 1).unsqueeze(0).contiguous()
        weighted_embedding, attn = self.attention(q, k, v)
        weighted_embedding = weighted_embedding.transpose(1, 2).contiguous().view(B, -1, self.d_word_vec)

        # print(weighted_embedding.shape)

        # print("Check banks")
        # print(torch.sum(self.att_banks), torch.sum(self.emb_banks))
        # input()
        
        if need_weights:
            return weighted_embedding, attn
        else:
            return weighted_embedding, None
