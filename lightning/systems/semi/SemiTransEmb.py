import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from dlhlp_lib.utils.tool import get_mask_from_lengths

from lightning.build import build_all_speakers
from lightning.systems import AdaptorSystem
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.model.codebook import SoftMultiAttCodebook
from lightning.model.text_encoder import TextEncoder
from lightning.callbacks.language.baseline_saver import Saver
from lightning.utils.tool import flat_merge_dict
from ..plugin.fscl_new import SemiFSCLPlugIn


class SemiTransEmbSystem(AdaptorSystem):
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
        opt_modules = [self.text_encoder, self.model]
        if self.use_matching:
            opt_modules += [self.text_matching, self.repr_matching]
        return nn.ModuleList(opt_modules) + self.fscl.build_optimized_model()

    def build_saver(self):
        self.saver = Saver(self.data_configs, self.model_config, self.log_dir, self.result_dir)
        return self.saver
    
    def _on_meta_batch_start(self, batch):
        """ Check meta-batch data """
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 4, "sup + qry + sup_info + qry_info"
        assert len(batch[0][0]) == 1, "n_batch == 1"
        assert len(batch[0][0][0]) == 13, "data with 13 elements"
    
    def common_r2s_step(self, repr, batch, batch_idx, train=True):
        qry_batch = batch[0][1][0]
        output = self.model(qry_batch[2], repr, *(qry_batch[4:]))
        loss = self.loss_func(qry_batch[:-1], output)
        loss_dict = {
            "Total Loss"       : loss[0],
            "Mel Loss"         : loss[1],
            "Mel-Postnet Loss" : loss[2],
            "Pitch Loss"       : loss[3],
            "Energy Loss"      : loss[4],
            "Duration Loss"    : loss[5],
        }
        return loss_dict, output
    
    def common_t2r_step(self, emb_table, batch, batch_idx, train=True):
        qry_batch = batch[0][1][0]
        emb_texts = F.embedding(qry_batch[3], emb_table, padding_idx=0)
        output = self.text_encoder(emb_texts, lengths=qry_batch[4])

        loss_dict = {}

        return loss_dict, output
   
    def common_step(self, batch, batch_idx, train=True):
        seg_repr = self.fscl.build_segmental_representation([batch[0][3]])
        emb_table, attn = self.fscl.build_embedding_table([batch[0][2]], return_attn=~train)
        t2r_loss_dict, t2r_output = self.common_t2r_step(emb_table, batch, batch_idx, train)  # be careful that seg_repr collapse

        if self.use_matching:  # Close the gap of distributions by restricting domain
            seg_repr, _ = self.repr_matching(seg_repr)
            t2r_output, _ = self.text_matching(t2r_output)

        if not train:
            u2s_loss_dict, u2s_output = self.common_r2s_step(t2r_output, batch, batch_idx, train)
        else:
            B, L, dim = seg_repr.shape
            mask = torch.rand(B) < 0.5
            mask = mask.unsqueeze(-1).expand(-1, L)
            # mask = torch.zeros((B, L), dtype=torch.bool) if random.uniform(0, 1) < 0.5 else torch.ones((B, L), dtype=torch.bool)
            mixed_repr = torch.where(mask.to(self.device).unsqueeze(-1), seg_repr, t2r_output)
            u2s_loss_dict, u2s_output = self.common_r2s_step(mixed_repr, batch, batch_idx, train)

        loss_dict = flat_merge_dict({
            "U2S": u2s_loss_dict,
            "TM": t2r_loss_dict
        })

        loss_dict["Total Loss"] = u2s_loss_dict["Total Loss"]

        # Calculate unsup ratio
        # length_mask = get_mask_from_lengths(qry_batch[4])
        # pl_ratio = torch.logical_and(~length_mask, ~mask).sum() / (~length_mask).sum()
        # self.log("PL ratio", pl_ratio.item(), sync_dist=True, batch_size=self.bs)
        
        return loss_dict, u2s_output, attn

    def training_step(self, batch, batch_idx):
        train_loss_dict, output, _ = self.common_step(batch, batch_idx, train=True)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': output, '_batch': qry_batch}

    def validation_step(self, batch, batch_idx):
        val_loss_dict, predictions, attn = self.common_step(batch, batch_idx, train=False)
        qry_batch = batch[0][1][0]

        if batch_idx == 0:
            layer_weights = self.fscl.get_hook("layer_weights")
            self.saver.log_layer_weights(self.logger, layer_weights.data, self.global_step + 1, "val")
        if batch_idx % 4 == 0:
            lang_id = qry_batch[-1][0].item()  # all batch belongs to the same language
            self.saver.log_codebook_attention(self.logger, attn, lang_id, batch_idx, self.global_step + 1, "val")
        
        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': qry_batch}
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint = self.fscl.on_save_checkpoint(checkpoint, prefix="fscl")
        return checkpoint

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.test_global_step = checkpoint["global_step"]
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        state_dict_pop_keys = []
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    if self.local_rank == 0:
                        print(f"Skip loading parameter: {k}, "
                                    f"required shape: {model_state_dict[k].shape}, "
                                    f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                if self.local_rank == 0:
                    print(f"Dropping parameter {k}")
                state_dict_pop_keys.append(k)
                is_changed = True

        # modify state_dict format to model_state_dict format
        for k in state_dict_pop_keys:
            state_dict.pop(k)
        for k in model_state_dict:
            if k not in state_dict:
                if "fscl.upstream" not in k:
                    print("Reinitialize: ", k)
                state_dict[k] = model_state_dict[k]

        if is_changed:
            checkpoint.pop("optimizer_states", None)
