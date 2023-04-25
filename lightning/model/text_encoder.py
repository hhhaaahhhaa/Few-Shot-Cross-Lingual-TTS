from typing import Type
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from dlhlp_lib.utils.tool import get_mask_from_lengths
from dlhlp_lib.utils.numeric import torch_exist_nan

from transformer import Encoder2
from lightning.model.codebook import SoftMultiAttCodebook


class TextEncoder(nn.Module):
    def __init__(self, model_config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_config = model_config
        self.build_model()

    def build_model(self):
        self.encoder = Encoder2(self.model_config)

    def build_optimized_model(self):
        return self

    def forward(self, x, lengths, mask=None):
        final_mask = get_mask_from_lengths(lengths).to(x.device)
        if mask is not None:
            final_mask = torch.logical_or(final_mask, mask)
            # Avoid to pass full mask into transformer, which will cause NaN gradient.
            B, L = mask.shape
            check_full_mask = final_mask.sum(dim=1)
            for i in range(B):
                if check_full_mask[i] == L:
                    final_mask[i][0] = 0
        x = self.encoder(x, final_mask)
        return x


class TextEncoderOld(nn.Module):
    def __init__(self, model_config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_config = model_config
        self.build_model()

    def codebook_bind(self, module: SoftMultiAttCodebook):
        print("Codebook bind...")
        self.codebook_attention.emb_banks = module.emb_banks  # bind
        # self.codebook_attention.emb_banks.requires_grad = False

    def build_model(self):
        self.use_matching = self.model_config.get("use_matching", True)
        self.encoder = Encoder2(self.model_config)
        self.codebook_attention = SoftMultiAttCodebook(
            codebook_size=self.model_config["codebook"]["codebook_size"],
            embed_dim=self.model_config["transformer"]["encoder_hidden"],
            num_heads=self.model_config["codebook"]["nhead"],
        )

    def build_optimized_model(self):
        return self
    
    def forward(self, x, lengths, mask=None):
        final_mask = get_mask_from_lengths(lengths).to(self.device)
        if mask is not None:
            final_mask = torch.logical_or(final_mask, mask)
            # Avoid to pass full mask into transformer, which will cause NaN gradient.
            B, L = mask.shape
            check_full_mask = final_mask.sum(dim=1)
            for i in range(B):
                if check_full_mask[i] == L:
                    final_mask[i][0] = 0
        x = self.encoder(x, final_mask)
        if self.use_matching:
            output, _ = self.codebook_attention(x)
            return output
        return x

    # def loss_func(self, x, y, lengths):
    #     mask = get_mask_from_lengths(lengths).to(self.device)
    #     return F.mse_loss(x.masked_select((~mask).unsqueeze(-1)), y.masked_select((~mask).unsqueeze(-1)))
    
    # def match_loss_func(self, x, y, lengths):
    #     return self.loss_func(x, y, lengths)
