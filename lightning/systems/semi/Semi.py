import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Type

from lightning.build import build_all_speakers, build_id2symbols
from lightning.systems import System
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.model.embeddings import MultilingualEmbedding
from lightning.model.codebook import SoftMultiAttCodebook
from lightning.model.text_encoder import TextEncoder
from lightning.callbacks.language.baseline_saver import Saver
from lightning.utils.tool import flat_merge_dict
from ..plugin.fscl_new import SemiFSCLPlugIn


class SemiSystem(System):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_configs(self):
        self.spk_config = {
            "emb_type": self.model_config["speaker_emb"],
            "speakers": build_all_speakers(self.data_configs)
        }
        self.bs = self.train_config["optimizer"]["batch_size"]

    def build_model(self):
        self.use_matching = self.model_config["use_matching"]
        encoder_dim = self.model_config["transformer"]["encoder_hidden"]
        self.embedding_model = MultilingualEmbedding(
            id2symbols=build_id2symbols(self.data_configs), dim=encoder_dim)
        self.text_encoder = TextEncoder(self.model_config["text_encoder"])
        self.model = FastSpeech2(self.model_config, spk_config=self.spk_config)
        self.loss_func = FastSpeech2Loss(self.model_config)
        self.fscl = SemiFSCLPlugIn(self.model_config["fscl"])

        if self.use_matching:
            self.shared_emb_banks = nn.Parameter(
                torch.randn(self.model_config["matching"]["codebook_size"], self.model_config["matching"]["dim"]))
            self.text_matching = SoftMultiAttCodebook(
                codebook_size=self.model_config["matching"]["codebook_size"],
                embed_dim=self.model_config["matching"]["dim"],
                num_heads=self.model_config["matching"]["nhead"],
            )
            self.repr_matching = SoftMultiAttCodebook(
                codebook_size=self.model_config["matching"]["codebook_size"],
                embed_dim=self.model_config["matching"]["dim"],
                num_heads=self.model_config["matching"]["nhead"],
            )
            self.text_matching.emb_banks = self.shared_emb_banks
            self.repr_matching.emb_banks = self.shared_emb_banks

    def build_optimized_model(self):
        opt_modules = [self.text_encoder, self.model, self.embedding_model]
        if self.use_matching:
            opt_modules += [self.text_matching, self.repr_matching]
        return nn.ModuleList(opt_modules) + self.fscl.build_optimized_model()
    
    def build_saver(self):
        self.saver = Saver(self.data_configs, self.model_config, self.log_dir, self.result_dir)
        return self.saver
    
    def common_r2s_step(self, repr, batch, batch_idx, train=True):
        labels, _ = batch
        output = self.model(labels[2], repr, *(labels[4:]))
        loss = self.loss_func(labels[:-1], output)
        loss_dict = {
            "Total Loss"       : loss[0],
            "Mel Loss"         : loss[1],
            "Mel-Postnet Loss" : loss[2],
            "Pitch Loss"       : loss[3],
            "Energy Loss"      : loss[4],
            "Duration Loss"    : loss[5],
        }
        return loss_dict, output
    
    def common_t2r_step(self, batch, batch_idx, train=True):
        labels, _ = batch
        emb_texts = self.embedding_model(labels[3])
        output = self.text_encoder(emb_texts, lengths=labels[4])

        loss_dict = {}

        return loss_dict, output
   
    def common_step(self, batch, batch_idx, train=True):
        _, ref_info = batch
        seg_repr, _ = self.fscl.build_segmental_representation([ref_info])
        t2r_loss_dict, t2r_output = self.common_t2r_step(batch, batch_idx, train)  # be careful that seg_repr collapse

        if self.use_matching:  # Close the gap of distributions by restricting domain
            seg_repr = self.repr_matching(seg_repr)
            t2r_output = self.text_matching(t2r_output)

        if not train:
            u2s_loss_dict, u2s_output = self.common_r2s_step(t2r_output, batch, batch_idx, train)
        else:  # here we use pure switch strategy instead of mixing
            B, L, dim = seg_repr.shape
            mask = torch.zeros((B, L)) if torch.rand() < 0.5 else torch.ones((B, L))
            mixed_repr = torch.where(mask.to(self.device).unsqueeze(-1), seg_repr, t2r_output)
            u2s_loss_dict, u2s_output = self.common_r2s_step(mixed_repr, batch, batch_idx, train)

        loss_dict = flat_merge_dict({
            "U2S": u2s_loss_dict,
            "TM": t2r_loss_dict
        })

        loss_dict["Total Loss"] = u2s_loss_dict["Total Loss"]
        return loss_dict, u2s_output

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch[0]) == 13, f"data with 13 elements, but get {len(batch)}"
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch[0]) == 13, f"data with 13 elements, but get {len(batch)}"
    
    def training_step(self, batch, batch_idx):
        train_loss_dict, output = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': output, '_batch': batch[0]}
    
    def validation_step(self, batch, batch_idx):
        val_loss_dict, predictions = self.common_step(batch, batch_idx, train=False)
        
        if batch_idx == 0:
            layer_weights = self.fscl.get_hook("layer_weights")
            self.saver.log_layer_weights(self.logger, layer_weights.data, self.global_step + 1, "val")

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': batch[0]}

    def on_save_checkpoint(self, checkpoint):
        checkpoint = self.fscl.on_save_checkpoint(checkpoint, prefix="fscl")
        return checkpoint
