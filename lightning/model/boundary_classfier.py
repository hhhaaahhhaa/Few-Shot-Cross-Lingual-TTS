from typing import Optional
import torch.nn as nn
import torch

from dlhlp_lib.common.wav2vec2U import SamePad
from dlhlp_lib.common.layers import WeightedSumLayer


class Classifier(nn.Module):
    def __init__(
        self, 
        n_in_layers: int,
        upstream_dim: int,
        specific_layer: Optional[int]=None,
        mode="readout",
    ):
        super(Classifier, self).__init__()
        self.mode = mode

        if self.mode == "readout":
            self.n_in_layers = n_in_layers
            self.weighted_sum = WeightedSumLayer(n_in_layers, specific_layer)

            self.layerwise_convolutions = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(upstream_dim, 768, 9, 1, 8),
                    SamePad(kernel_size=9, causal=True),
                    nn.ReLU(),
                ) for _ in range(self.n_in_layers)
            ])
            self.network = nn.Sequential(
                nn.Conv1d(768, 512, kernel_size=3, stride=1, padding=2),
                SamePad(kernel_size=3, causal=True),
                nn.ReLU(),
                nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=2),
                SamePad(kernel_size=3, causal=True),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=2),
                SamePad(kernel_size=3, causal=True),
                nn.ReLU(),
                nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=2),
                SamePad(kernel_size=3, causal=True),
                nn.ReLU(),
                nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=2),
                SamePad(kernel_size=3, causal=True),
                nn.ReLU(),
            )
            self.out = nn.Linear(32, 1)
        elif self.mode == "finetune":
            self.out = nn.Linear(upstream_dim, 1)
        else:
            raise NotImplementedError
    
    def forward(self, x):
        """
        Args:
            x: S3PRL output tensor with shape (B, L, n_layer, dim)
        """
        if self.mode == "readout":
            layers = []
            for i in range(self.n_in_layers):
                x_slc = x[:, :, i, :].permute(0, 2, 1).contiguous()
                layers.append(self.layerwise_convolutions[i](x_slc))
            x = torch.stack(layers, dim=0)  # n_in_layer, B, 768, L
            x = self.weighted_sum(x, dim=0)  # B, 768, L
            x = self.network(x)
            x = x.permute(0, 2, 1).contiguous()  # B, L, 32
        elif self.mode == "finetune":
            x = x[:, :, -1, :]  # B, L, upstream_dim
        out = self.out(x).squeeze(-1)  # B, L

        return out
