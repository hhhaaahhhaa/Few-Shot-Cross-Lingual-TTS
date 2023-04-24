import torch
import torch.nn as nn
import torch.nn.functional as F

from dlhlp_lib.transformers import TransformerEncoderBlock, CodeformerBlock
from dlhlp_lib.utils.tool import get_mask_from_lengths


class BiLSTMDownstream(nn.Module):
    """ BiLSTM """
    def __init__(self,
        upstream_dim: int,
        d_out: int,
        use_proj: bool=True,
    ) -> None:
        super().__init__()
        self.d_out = d_out
        self.proj = nn.Linear(upstream_dim, self.d_out) if use_proj else None
        
        self.lstm = nn.LSTM(input_size=self.d_out, hidden_size=self.d_out // 2, 
                                num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, x, lengths):
        """
        Args:
            x: Representation with shape (B, L, n_layers, d_in).
            lengths: Handle padding for LSTM.
        Return:
            Return tensor with shape (B, L, d_out)
        """
        if self.proj is not None:
            x = self.proj(x)  # B, L, d_out
        # total length should be record due to data parallelism issue (https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism)
        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)  # B, L, d_out
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=total_length)

        return x


class TransformerDownstream(nn.Module):
    """ TransformerEncoderBlock * n """
    def __init__(self,
        model_config,
        upstream_dim: int,
        use_proj: bool=True,
    ) -> None:
        super().__init__()
        self.d_out = model_config["transformer"]["d_model"]
        self.proj = nn.Linear(upstream_dim, self.d_out) if use_proj else None

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

    def forward(self, x, lengths):
        """
        Args:
            x: Representation with shape (B, L, d_in).
            lengths: Handle padding.
        Return:
            Return tensor with shape (B, L, d_out)
        """
        if self.proj is not None:
            x = self.proj(x)  # B, L, d_out
        padding_mask = get_mask_from_lengths(lengths).to(x.device)
        for layer in self.layers:
            x, _ = layer(x, src_key_padding_mask=padding_mask)

        return x
