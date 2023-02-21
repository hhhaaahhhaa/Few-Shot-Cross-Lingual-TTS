import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from dlhlp_lib.s3prl import S3PRLExtractor
from dlhlp_lib.transformers import CodebookAttention

import Define
from lightning.build import build_all_speakers, build_id2symbols
from lightning.systems.adaptor import AdaptorSystem
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.model.reduction import PhonemeQueryExtractor
from lightning.callbacks.language.baseline_saver import Saver
from .embeddings import *
from ..t2u.downstreams import LinearDownstream


class TransEmbSystem(AdaptorSystem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def build_configs(self):
        self.spk_config = {
            "emb_type": self.model_config["speaker_emb"],
            "speakers": build_all_speakers(self.data_configs)
        }
        self.bs = self.train_config["optimizer"]["batch_size"]
    
    def build_model(self):
        encoder_dim = self.model_config["transformer"]["encoder_hidden"]
        self.model = FastSpeech2(self.model_config, spk_config=self.spk_config)
        self.loss_func = FastSpeech2Loss(self.model_config)
        
        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.embedding_generator = LinearDownstream(
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            d_out=encoder_dim,
            specific_layer=Define.LAYER_IDX
        )
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)
        self.codebook_attention = CodebookAttention(
            codebook_size=self.model_config["codebook_size"],
            embed_dim=encoder_dim,
            num_heads=self.model_config["nhead"],
        )

        # Although the vocoder is only used in callbacks, we need it to be
        # moved to cuda for faster inference, so it is initialized here with the
        # model, and let pl.Trainer handle the DDP devices.
        # self.vocoder = LightningMelGAN()
        # self.vocoder.freeze()

    def build_optimized_model(self):
        return nn.ModuleList([self.codebook_attention, self.embedding_generator, self.model])

    def build_saver(self):
        self.saver = Saver(self.data_configs, self.model_config, self.log_dir, self.result_dir)
        return self.saver

    def build_embedding_table(self, batch, return_attn=False):  
        _, _, sup_info = batch[0]

        # TODO: Mel version
        self.upstream.eval()
        with torch.no_grad():
            ssl_repr, _ = self.upstream.extract(sup_info["raw_feat"])  # B, L, n_layers, dim
            ssl_repr = ssl_repr.detach()

        x = self.embedding_generator(ssl_repr)
        table_pre = self.phoneme_query_extractor(x, sup_info["avg_frames"], 
                            sup_info["n_symbols"], sup_info["phonemes"])  # 1, n_symbols, n_layers, dim

        table, attn = self.codebook_attention(table_pre, need_weights=return_attn)
        table = table.squeeze(0)  # n_symbols, dim
        
        # print("Table shape and gradient required: ", table.shape)
        # print(table.requires_grad)
        
        if return_attn:
            return table, attn
        else:
            return table

    def _on_meta_batch_start(self, batch):
        """ Check meta-batch data """
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 3, "sup + qry + sup_info"
        assert len(batch[0][0]) == 1, "n_batch == 1"
        assert len(batch[0][0][0]) == 13, "data with 13 elements"
    
    def common_step(self, batch, batch_idx, train=True):
        if not train:
            emb_table, attn = self.build_embedding_table(batch, return_attn=True)
        else:
            emb_table = self.build_embedding_table(batch)
        qry_batch = batch[0][1][0]
        emb_texts = F.embedding(qry_batch[3], emb_table, padding_idx=0)
        output = self.model(qry_batch[2], emb_texts, *(qry_batch[4:]))
        loss = self.loss_func(qry_batch[:-1], output)
        loss_dict = {
            "Total Loss"       : loss[0],
            "Mel Loss"         : loss[1],
            "Mel-Postnet Loss" : loss[2],
            "Pitch Loss"       : loss[3],
            "Energy Loss"      : loss[4],
            "Duration Loss"    : loss[5],
        }
        if not train:
            return loss_dict, output, attn
        else:
            return loss_dict, output

    def training_step(self, batch, batch_idx):
        train_loss_dict, output = self.common_step(batch, batch_idx, train=True)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': output, '_batch': qry_batch}

    def validation_step(self, batch, batch_idx):
        val_loss_dict, predictions, attn = self.common_step(batch, batch_idx, train=False)
        qry_batch = batch[0][1][0]

        # visualization
        if batch_idx == 0:
            layer_weights = F.softmax(self.embedding_generator.weighted_sum.weight_raw, dim=0)
            self.saver.log_layer_weights(self.logger, layer_weights.data, self.global_step + 1, "val")
        if batch_idx in [0, 8, 16]:
            lang_id = qry_batch[-1][0].item()  # all batch belongs to the same language
            self.saver.log_codebook_attention(self.logger, attn, lang_id, batch_idx, self.global_step + 1, "val")

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': qry_batch}
    
    # def visualize_matching(self, batch, batch_idx):
    #     if self.codebook_type != "table-sep":
    #         _, _, repr_info, lang_id = batch[0]
    #         with torch.no_grad():
    #             ref_phn_feats = self.reference_extractor.extract(repr_info, norm=False)
    #         matching = self.embedding_model.get_matching(self.codebook_type, ref_phn_feats=ref_phn_feats, lang_id=lang_id)
    #         self.codebook_analyzer.visualize_matching(batch_idx, matching)
    #     return None

    # def log_matching(self, batch, batch_idx, stage="val"):
    #     step = self.global_step + 1
    #     _, _, repr_info, lang_id = batch[0]
    #     with torch.no_grad():
    #         ref_phn_feats = self.reference_extractor.extract(repr_info, norm=False)
        
    #     matchings = self.embedding_model.get_matching(self.codebook_type, ref_phn_feats=ref_phn_feats, lang_id=lang_id)
    #     for matching in matchings:
    #         fig = self.codebook_analyzer.plot_matching(matching, quantized=False)
    #         figure_name = f"{stage}/step_{step}_{batch_idx:03d}_{matching['title']}"
    #         self.logger.experiment.log_figure(
    #             figure_name=figure_name,
    #             figure=fig,
    #             step=step,
    #         )
    #         plt.close(fig)

    def on_save_checkpoint(self, checkpoint):
        """ (Hacking!) Remove pretrained weights in checkpoint to save disk space. """
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k in state_dict:
            if k.split('.')[0] == "upstream":
                continue
            new_state_dict[k] = state_dict[k]
        checkpoint["state_dict"] = new_state_dict

        return checkpoint
