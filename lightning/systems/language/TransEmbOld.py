# Deprecated
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from dlhlp_lib.s3prl import S3PRLExtractor

import Define
from lightning.build import build_all_speakers
from lightning.systems.adaptor import AdaptorSystem
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.model.reduction import PhonemeQueryExtractor
from lightning.utils.tool import ssl_match_length
from lightning.callbacks.language.baseline_saver import Saver
from .embeddings import *
from ..t2u.downstreams import Downstream1


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
        self.model = FastSpeech2(self.model_config, spk_config=self.spk_config)
        self.loss_func = FastSpeech2Loss(self.model_config)
        
        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.embedding_generator = Downstream1(
            self.model_config["downstream"],
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            specific_layer=Define.LAYER_IDX
        )
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)

    def build_optimized_model(self):
        return nn.ModuleList([self.embedding_generator, self.model])

    def build_saver(self):
        self.saver = Saver(self.data_configs, self.model_config, self.log_dir, self.result_dir)
        return self.saver

    def build_embedding_table(self, batch):  
        _, _, sup_info = batch[0]

        # TODO: Mel version
        self.upstream.eval()
        with torch.no_grad():
            ssl_repr, _ = self.upstream.extract(sup_info["raw_feat"])  # B, L, n_layers, dim
            ssl_repr = ssl_match_length(ssl_repr, sup_info["max_len"].item())
            ssl_repr = ssl_repr.detach()

        x = self.embedding_generator(ssl_repr, sup_info["lens"].cpu())
        table = self.phoneme_query_extractor(x, sup_info["avg_frames"], 
                            sup_info["n_symbols"], sup_info["phonemes"])  # 1, n_symbols, n_layers, dim
        table = table.squeeze(0)  # n_symbols, dim
        
        # print("Table shape and gradient required: ", table.shape)
        # print(table.requires_grad)
        
        return table

    def _on_meta_batch_start(self, batch):
        """ Check meta-batch data """
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 3, "sup + qry + sup_info"
        assert len(batch[0][0]) == 1, "n_batch == 1"
        assert len(batch[0][0][0]) == 13, "data with 13 elements"
    
    def common_step(self, batch, batch_idx, train=True):
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
        
        return loss_dict, output

    def training_step(self, batch, batch_idx):
        train_loss_dict, output = self.common_step(batch, batch_idx, train=True)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': output, '_batch': qry_batch}

    def validation_step(self, batch, batch_idx):
        val_loss_dict, predictions = self.common_step(batch, batch_idx, train=False)
        qry_batch = batch[0][1][0]

        # visualization
        if batch_idx == 0:
            layer_weights = F.softmax(self.embedding_generator.weighted_sum.weight_raw, dim=0)
            self.saver.log_layer_weights(self.logger, layer_weights.data, self.global_step + 1, "val")
        
        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': qry_batch}
    
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
