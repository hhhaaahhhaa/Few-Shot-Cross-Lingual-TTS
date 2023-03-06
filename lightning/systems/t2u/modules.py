import torch
import torch.nn as nn
import torch.nn.functional as F

from dlhlp_lib.utils.tool import get_mask_from_lengths
from dlhlp_lib.common.layers import GradientReversalLayer
from dlhlp_lib.common.wav2vec2U import Discriminator


class DA(nn.Module):
    def __init__(self, d_in: int, mixture: float=0.0) -> None:
        super().__init__()
        self.mixture = mixture
        self.rev = GradientReversalLayer()
        self.D = Discriminator(d_in)
        self.d_in = d_in

    def embed_discrete(self, x):
        x1 = F.one_hot(x, self.d_in).float()
        return x1 * (1 - self.mixture) + self.mixture * torch.ones_like(x1)
    
    def check_entropy(self, x, lengths):
        # x: Tensor with size B, L, d_enc
        x = x * torch.log(x + 1e-8)
        x_sz = x.size(1)
        padding_mask = get_mask_from_lengths(lengths)
        padding_mask = padding_mask[:, : x.size(1)]
        x[padding_mask] = 0
        x_sz = x_sz - padding_mask.sum(dim=-1)
        x = x.sum(dim=[1, 2])
        x = x / x_sz
        return x.mean()

    def forward(self, x, lengths, gradient_reverse: bool=False):
        padding_mask = get_mask_from_lengths(lengths)
        if gradient_reverse:
            x = self.rev(x)
        score = self.D(x, padding_mask)
        
        return score
