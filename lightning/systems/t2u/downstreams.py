import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from dlhlp_lib.transformers import TransformerEncoderBlock, CodeformerBlock
from dlhlp_lib.utils.tool import get_mask_from_lengths


class WeightedSumLayer(nn.Module):
    def __init__(self, n_in_layers: int, specific_layer: Optional[int]=None) -> None:
        super().__init__()
        self.n_in_layers = n_in_layers
            
        # specific layer, fix weight_raw during training.
        if specific_layer is not None:
            weights = torch.ones(n_in_layers) * float('-inf')
            weights[specific_layer] = 10.0
            self.weight_raw = nn.Parameter(weights)
            self.weight_raw.requires_grad = False
        else:
            self.weight_raw = nn.Parameter(torch.randn(n_in_layers))

    def forward(self, x, dim: int):
        weight_shape = [1] * x.dim()
        weight_shape[dim] = self.n_in_layers
        weighted_sum = torch.reshape(F.softmax(self.weight_raw, dim=0), tuple(weight_shape)) * x  # B, L, d_in
        weighted_sum = weighted_sum.sum(dim=dim)
        
        return weighted_sum


class LinearDownstream(nn.Module):
    """
    Weighted sum + Linear.
    """
    def __init__(self, n_in_layers: int, upstream_dim: int, d_out: int, specific_layer: Optional[int]=None) -> None:
        super().__init__()
        self.weighted_sum = WeightedSumLayer(n_in_layers, specific_layer)

        self.d_out = d_out
        self.proj = nn.Linear(upstream_dim, self.d_out)

    def forward(self, repr):
        """
        Args:
            repr: SSL representation with shape (B, L, n_layers, d_in).
        Return:
            Return tensor with shape (B, L, d_out)
        """
        x = self.weighted_sum(repr, dim=2)
        x = self.proj(x)  # B, L, d_out

        return x


class BiLSTMDownstream(nn.Module):
    """
    Weighted sum + BiLSTM.
    """
    def __init__(self, n_in_layers: int, upstream_dim: int, d_out: int, specific_layer: Optional[int]=None) -> None:
        super().__init__()

        self.weighted_sum = WeightedSumLayer(n_in_layers, specific_layer)
        self.d_out = d_out
        self.proj = nn.Linear(upstream_dim, self.d_out)
        self.lstm = nn.LSTM(input_size=self.d_out, hidden_size=self.d_out // 2, 
                                num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, repr, lengths):
        """
        Args:
            repr: SSL representation with shape (B, L, n_layers, d_in).
            lengths: Handle padding for LSTM.
        Return:
            Return tensor with shape (B, L, d_out)
        """
        x = self.weighted_sum(repr, dim=2)
        x = self.proj(x)  # B, L, d_out

        # total length should be record due to data parallelism issue (https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism)
        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)  # B, L, d_out
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=total_length)

        return x


class Downstream1(nn.Module):
    """
    Weighted sum + TransformerEncoderBlock * n.
    """
    def __init__(self,
        model_config,
        n_in_layers: int,
        upstream_dim: int,
        specific_layer: Optional[int]=None
    ) -> None:
        super().__init__()
        self.weighted_sum = WeightedSumLayer(n_in_layers, specific_layer)

        self.d_out = model_config["transformer"]["d_model"]
        self.proj = nn.Linear(upstream_dim, self.d_out)

        self.layers = nn.ModuleList()
        for i in range(model_config["transformer"]["layer"]):
            self.layers.append(
                TransformerEncoderBlock(
                    d_model=model_config["transformer"]["d_model"],
                    nhead=model_config["transformer"]["nhead"],
                    dim_feedforward=model_config["transformer"]["dim_feedforward"][i],
                    dropout=model_config["transformer"]["dropout"]
                )
            )

    def forward(self, repr, lengths):
        """
        Args:
            repr: SSL representation with shape (B, L, n_layers, d_in).
            lengths: Handle padding.
        Return:
            Return tensor with shape (B, L, d_out)
        """
        x = self.weighted_sum(repr, dim=2)
        x = self.proj(x)  # B, L, d_out

        padding_mask = get_mask_from_lengths(lengths).to(x.device)
        for layer in self.layers:
            x, _ = layer(x, src_key_padding_mask=padding_mask)

        return x


class Downstream2(nn.Module):
    """
    Weighted sum + TransformerEncoderBlock * (n-1) + CodeformerBlock.
    """
    def __init__(self,
        model_config,
        n_in_layers: int,
        upstream_dim: int,
        specific_layer: Optional[int]=None
    ) -> None:
        super().__init__()
        self.weighted_sum = WeightedSumLayer(n_in_layers, specific_layer)

        self.d_out = model_config["transformer"]["d_model"]
        self.proj = nn.Linear(upstream_dim, self.d_out)

        self.layers = nn.ModuleList()
        for i in range(model_config["transformer"]["layer"] - 1):
            self.layers.append(
                TransformerEncoderBlock(
                    d_model=model_config["transformer"]["d_model"],
                    nhead=model_config["transformer"]["nhead"],
                    dim_feedforward=model_config["transformer"]["dim_feedforward"][i],
                    dropout=model_config["transformer"]["dropout"]
                )
            )
        self.layers.append(
            CodeformerBlock(
                codebook_size=model_config["codebook_size"],
                d_model=model_config["transformer"]["d_model"],
                nhead=model_config["transformer"]["nhead"],
                dim_feedforward=model_config["transformer"]["dim_feedforward"][-1],
                dropout=model_config["transformer"]["dropout"]
            )
        )

    def forward(self, repr, lengths, need_weights=False):
        """
        Args:
            repr: SSL representation with shape (B, L, n_layers, d_in).
            lengths: Handle padding.
        Return:
            Return tensor with shape (B, L, d_out)
        """
        x = self.weighted_sum(repr, dim=2)
        x = self.proj(x)  # B, L, d_out

        padding_mask = get_mask_from_lengths(lengths).to(x.device)
        for layer in self.layers[:-1]:
            x, _ = layer(x, src_key_padding_mask=padding_mask)
        x, attn_weights = self.layers[-1](x, need_weights=need_weights)
        if need_weights:
            return x, attn_weights
        else:
            return x
