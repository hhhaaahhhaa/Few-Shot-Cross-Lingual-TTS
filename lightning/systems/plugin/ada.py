import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from lightning.model import FastSpeech2ADALoss, ADAEncoder


class FastSpeech2ADAPlugIn(pl.LightningModule):
    def __init__(self, d_in, model_config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_in = d_in
        self.model_config = model_config

    def build_model(self):
        self.model = ADAEncoder(self.d_in, self.model_config)
        self.recon_loss_func = FastSpeech2ADALoss()
        self.match_loss_func = nn.MSELoss()

    def build_optimized_model(self):
        return nn.ModuleList([self.model])
    
    def forward(self, x, lengths, embed=True):
        return self.model(x, lengths, embed=embed)
