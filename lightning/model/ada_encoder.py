import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.utils.tool import get_mask_from_lengths

from transformer import Encoder2


class ADAEncoder(pl.LightningModule):
    """
    Use mel for default since mel is a universal feature
    """
    def __init__(self, config) -> None:
        super().__init__()
        encoder_dim = config["transformer"]["encoder_hidden"]
        self.embedding = nn.Linear(AUDIO_CONFIG["mel"]["n_mel_channels"], encoder_dim)
        self.encoder = Encoder2(config)

    def forward(self, x, lengths):
        x = self.embedding(x)
        mask = get_mask_from_lengths(lengths).to(self.device)
        return self.encoder(x, mask)
