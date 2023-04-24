import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.utils.tool import get_mask_from_lengths

import Define
from transformer import Decoder, PostNet, Encoder2
from .modules import VarianceAdaptor
from .speaker_encoder import SpeakerEncoder, LanguageEncoder


class FastSpeech2(pl.LightningModule):
    """ Headless FastSpeech2 """

    def __init__(self, model_config, **kwargs):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder2(model_config)
        self.variance_adaptor = VarianceAdaptor(model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            AUDIO_CONFIG["mel"]["n_mel_channels"]
        )
        self.postnet = PostNet()

        # If not using multi-speaker, would return None
        if not model_config.get("multi_speaker", False):
            self.speaker_emb = None
        else:
            self.speaker_emb = SpeakerEncoder(model_config, kwargs["spk_config"])

        # If not using multi-lingual, would return None
        if not model_config.get("multi_lingual", False):
            self.language_emb = None
        else:
            # currently fixed, enable up to 100 languages
            self.language_emb = LanguageEncoder(model_config, {"emb_type": "table"})

    def forward(
        self,
        speaker_args,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        lang_args=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        average_spk_emb=False,
    ):
        # print("Ch1-1")
        # print(e_targets, e_targets.sum(-1))
        
        src_masks = get_mask_from_lengths(src_lens, max_src_len).to(self.device)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len).to(self.device)
            if mel_lens is not None
            else None
        )

        # print("FastSpeech2m input shape: ", texts.shape)
        # print("FastSpeech2m mask shape: ", src_masks.shape)
        output = self.encoder(texts, src_masks)
        # print("FastSpeech2m encoder output shape: ", output.shape)

        # print("Check encoder output")
        # print(output.sum(-1))

        if self.speaker_emb is not None:
            spk_emb = self.speaker_emb(speaker_args)
            # print("FastSpeech2m spk_emb shape: ", spk_emb.shape)
            if average_spk_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True).expand(output.shape[0], -1)
            output += spk_emb.unsqueeze(1).expand(-1, max_src_len, -1)
            # output = output + self.speaker_emb(speaker_args).unsqueeze(1).expand(
            #     -1, max_src_len, -1
            # )

        # print("Check add spkemb output")
        # print(output.sum(-1))
        # input()

        # if not Define.NOLID:
        #     if self.language_emb is not None and lang_args is not None:
        #         lang_emb = self.language_emb(lang_args)
        #         output += lang_emb.unsqueeze(1).expand(-1, max_src_len, -1)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        # print("Check va output")
        # print(output.sum(-1))
        # print(p_predictions.sum())
        # print(e_predictions.sum())
        # input()

        # print("FastSpeech2m variance adaptor output shape: ", output.shape)

        if self.speaker_emb is not None:
            spk_emb = self.speaker_emb(speaker_args)
            if average_spk_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True).expand(output.shape[0], -1)
            output += spk_emb.unsqueeze(1).expand(-1, max(mel_lens), -1)
            # output = output + self.speaker_emb(speaker_args).unsqueeze(1).expand(
            #     -1, max(mel_lens), -1
            # )

        output, mel_masks = self.decoder(output, mel_masks)
        # print(output.shape)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        # print("Check decoder output")
        # print(output.sum())
        # print(postnet_output.sum())
        # input()

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
