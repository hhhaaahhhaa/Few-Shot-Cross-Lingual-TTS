import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMDownstream(nn.Module):
    """
    Weighted sum + BiLSTM.
    """
    def __init__(self, n_in_layers: int, upstream_dim: int, specific_layer: int=None) -> None:
        super().__init__()
        self.weight_raw = nn.Parameter(torch.zeros(1, 1, n_in_layers, 1))
            
        # specific layer, fix weight_raw during training.
        if specific_layer is not None:
            last_hidden = torch.ones(1, 1, n_in_layers, 1) * float('-inf')
            last_hidden[0][0][specific_layer][0] = 10.0
            self.weight_raw = nn.Parameter(last_hidden)
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
