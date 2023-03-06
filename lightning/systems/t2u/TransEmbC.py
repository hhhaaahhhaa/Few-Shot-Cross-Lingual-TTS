import torch
import torch.nn as nn
import torch.nn.functional as F
import jiwer
from collections import OrderedDict

from dlhlp_lib.s3prl import S3PRLExtractor

import Define
from lightning.build import build_id2symbols
from lightning.systems.adaptor import AdaptorSystem
from lightning.callbacks.t2u.saver import Saver
from lightning.utils.tool import ssl_match_length
from Objects.visualization import CodebookAnalyzer
from lightning.model.reduction import PhonemeQueryExtractor
from ..phoneme_recognition.loss import PRFramewiseLoss
from .tacotron2.tacot2u import TacoT2U
from .tacotron2.hparams import hparams
from .downstreams import Downstream2, LinearDownstream
from ..language.embeddings import SoftMultiAttCodebook


class TransEmbCSystem(AdaptorSystem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.codebook_analyzer = CodebookAnalyzer(self.result_dir)

    def build_model(self):
        id2symbols = build_id2symbols(self.data_configs)
        n_units = len(id2symbols[self.data_configs[0]["target"]["unit_name"]])   # all target unit names from data configs should be the same!
        setattr(hparams, "n_units", n_units)
        self.model = TacoT2U(self.model_config)
        self.loss_func = PRFramewiseLoss()

        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.embedding_generator = Downstream2(
            self.model_config,
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            specific_layer=Define.LAYER_IDX
        )
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)

    def build_optimized_model(self):
        return nn.ModuleList([self.embedding_generator, self.model])

    def build_saver(self):
        self.saver = Saver(self.data_configs, self.log_dir, self.result_dir, re_id=False)
        return self.saver

    def build_embedding_table(self, batch, return_attn=False):
        _, _, sup_info = batch[0]

        # TODO: Mel version
        self.upstream.eval()
        with torch.no_grad():
            ssl_repr, _ = self.upstream.extract(sup_info["raw_feat"])  # B, L, n_layers, dim
            ssl_repr = ssl_match_length(ssl_repr, sup_info["max_len"].item())
            ssl_repr = ssl_repr.detach()

        if return_attn:
            x, attn = self.embedding_generator(ssl_repr, sup_info["lens"].cpu(), need_weights=True)
        else:
            x = self.embedding_generator(ssl_repr, sup_info["lens"].cpu())
        table = self.phoneme_query_extractor(x, sup_info["avg_frames"], 
                            sup_info["n_symbols"], sup_info["phonemes"])  # 1, n_symbols, n_layers, dim
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
        assert len(batch[0][1]) == 1, "n_batch == 1"
        assert len(batch[0][1][0]) == 11, f"data with 11 elements, but get {len(batch[0][1][0])}"
    
    def common_step(self, batch, batch_idx, train=True):
        if not train:
            emb_table, attn = self.build_embedding_table(batch, return_attn=True)
        else:
            emb_table = self.build_embedding_table(batch)
        qry_batch = batch[0][1][0]
        emb_texts = F.embedding(qry_batch[3], emb_table, padding_idx=0)
        # (emb_texts, text_lengths, units, max_len, output_lengths, spks, lang_ids)
        inputs = (emb_texts, qry_batch[4], qry_batch[6], qry_batch[5], qry_batch[7], qry_batch[3], qry_batch[9])
        self.model.decoder.teacher_forcing_ratio = schedule_f(self.global_step + 1)
        output, alignments = self.model(inputs)
        # print(output.shape)
        # print(alignments.shape)
        # print(qry_batch[6].shape)
        loss = self.loss_func(qry_batch[6], output)
        loss_dict = {
            "Total Loss": loss,
        }

        if not train:
            return loss_dict, output, alignments, attn
        else:
            return loss_dict, output, alignments

    def training_step(self, batch, batch_idx):
        train_loss_dict, predictions, alignment = self.common_step(batch, batch_idx, train=True)

        qry_batch = batch[0][1][0]
        mask = (qry_batch[6] != 0)
        acc = ((qry_batch[6] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
        self.log_dict({"Train/Acc": acc.item()}, sync_dist=True, batch_size=len(qry_batch[0]))

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=len(qry_batch[0]))
        self.log("Schedule sampling ratio", schedule_f(self.global_step + 1), sync_dist=True)

        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions,
                '_batch': qry_batch, 'symbol_id': qry_batch[10][0], 'alignment': alignment}

    def validation_step(self, batch, batch_idx):
        val_loss_dict, predictions, alignment, attn = self.common_step(batch, batch_idx, train=False)

        qry_batch = batch[0][1][0]
        mask = (qry_batch[6] != 0)
        acc = ((qry_batch[6] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
        self.log_dict({"Val/Acc": acc.item()}, sync_dist=True, batch_size=len(qry_batch[0]))

        # visualization
        if batch_idx == 0:
            layer_weights = F.softmax(self.embedding_generator.weighted_sum.weight_raw, dim=0)
            self.saver.log_layer_weights(self.logger, layer_weights.data, self.global_step + 1, "val")
        if batch_idx in [0, 8, 16]:
            lang_id = qry_batch[9][0].item()  # all batch belongs to the same language
            self.saver.log_codebook_attention(self.logger, attn, lang_id, batch_idx, self.global_step + 1, "val")

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=len(qry_batch[0]))
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, 
                '_batch': qry_batch, 'symbol_id': qry_batch[10][0], 'alignment': alignment}

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

    def teacher_inference(self, text, text_gt, symbol_id):
        """
        Return TacoT2U results:
            (
                output,
                alignment,
            )
        """
        # Match input format for model.forward
        texts = torch.from_numpy(text).long().unsqueeze(0).to(self.device)
        emb_texts = self.embedding_model(texts, symbol_id)
        units = torch.from_numpy(text_gt).long().unsqueeze(0).to(self.device)
        text_lengths = torch.LongTensor([texts.shape[1]]).to(self.device)
        max_len = torch.max(text_lengths)
        output_lengths = torch.LongTensor([units.shape[1]]).to(self.device)

        inputs = (emb_texts, text_lengths, units, max_len, output_lengths, None, None)        
        with torch.no_grad():
            self.model.decoder.teacher_forcing_ratio = 1.0
            output = self.model(inputs)

        return output

    def inference(self, text, symbol_id):
        """
        Return TacoT2U results:
            (
                output,
                alignment,
            )
        """
        texts = torch.from_numpy(text).long().unsqueeze(0).to(self.device)
        emb_texts = self.embedding_model(texts, symbol_id)
        
        with torch.no_grad():
            output = self.model.inference(emb_texts, None, None)

        return output


class TransEmbOrigSystem(TransEmbCSystem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        encoder_dim = self.model_config["tacotron2"]["symbols_embedding_dim"]
        id2symbols = build_id2symbols(self.data_configs)
        n_units = len(id2symbols[self.data_configs[0]["target"]["unit_name"]])   # all target unit names from data configs should be the same!
        setattr(hparams, "n_units", n_units)
        self.model = TacoT2U(self.model_config)
        self.loss_func = PRFramewiseLoss()

        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.embedding_generator = LinearDownstream(
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            d_out=encoder_dim,
            specific_layer=Define.LAYER_IDX
        )
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)
        self.codebook_attention = SoftMultiAttCodebook(
            codebook_size=self.model_config["codebook_size"],
            embed_dim=self.model_config["transformer"]["d_model"],
            num_heads=self.model_config["transformer"]["nhead"],
        )
        
    def build_embedding_table(self, batch, return_attn=False):  
        _, _, sup_info = batch[0]

        # TODO: Mel version
        self.upstream.eval()
        with torch.no_grad():
            ssl_repr, _ = self.upstream.extract(sup_info["raw_feat"])  # B, L, n_layers, dim
            ssl_repr = ssl_match_length(ssl_repr, sup_info["max_len"].item())
            ssl_repr = ssl_repr.detach()

        # This is the order of original version
        x = self.embedding_generator.weighted_sum(ssl_repr, dim=2)
        table_pre = self.phoneme_query_extractor(x, sup_info["avg_frames"], 
                            sup_info["n_symbols"], sup_info["phonemes"])  # 1, n_symbols, dim
        table_pre = self.embedding_generator.proj(table_pre)

        table, attn = self.codebook_attention(table_pre, need_weights=return_attn)
        table = table.squeeze(0)  # n_symbols, dim
        table[0].fill_(0)
        
        # print("Table shape and gradient required: ", table.shape)
        # print(table.requires_grad)
        
        if return_attn:
            return table, attn
        else:
            return table
        
    def build_optimized_model(self):
        return nn.ModuleList([self.codebook_attention, self.embedding_generator, self.model])


def schedule_f(step: int) -> float:
    return 1.0
    # return max(0.5, 1 - step / 20000)
    # else:
    #     return max(0, 0.5 - (step - 20000) / 20000)
