import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


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

    
class MultilingualClusterHead(nn.Module):
    def __init__(self, lang_id2symbols: Dict[str, int], d_in: int, temperature=0.1, mode="cos"):
        super().__init__()
        self.lang_id2symbols = lang_id2symbols
        self.d_in = d_in
        self.temperature = temperature
        self.mode = mode

        self.clusters = nn.ParameterDict()
        for lang_id, v in lang_id2symbols.items():
            if len(v) > 0:
                self.clusters[f"head-{lang_id}"] = nn.Parameter(torch.randn(len(v), d_in))

    def forward(self, x, lang_id: int):
        """
        Args:
            x: Tensor with shape (B, L, d_in), which should be a time sequence.
        Return:
            Tensor with shape (B, L, n_c), calculate cosine similarity between centers and input as Hubert and wav2vec2.0.
        """
        y = self.clusters[f"head-{lang_id}"].unsqueeze(0).unsqueeze(0)
        if self.mode == "cos":
            sim = F.cosine_similarity(y, x.unsqueeze(2), dim=3)  # B, L, n_c
            return sim / self.temperature
        elif self.mode == "l2":
            sim = torch.linalg.norm(y - x, dim=3)
            return sim
        else:
            raise NotImplementedError
