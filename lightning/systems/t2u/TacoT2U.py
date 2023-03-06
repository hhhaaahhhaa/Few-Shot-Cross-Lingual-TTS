import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning.build import build_id2symbols
from lightning.systems.system import System
from lightning.callbacks.t2u.saver import Saver
from ..phoneme_recognition.loss import PRFramewiseLoss
from ..language.embeddings import MultilingualEmbedding
from .tacotron2.tacot2u import TacoT2U
from .tacotron2.hparams import hparams
from .schedules import no_schedule as schedule_f


class TacoT2USystem(System):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        id2symbols = build_id2symbols(self.data_configs)
        n_units = len(id2symbols[self.data_configs[0]["target"]["unit_name"]])   # all target unit names from data configs should be the same!
        setattr(hparams, "n_units", n_units)
        encoder_dim = self.model_config["tacotron2"]["symbols_embedding_dim"]
        self.embedding_model = MultilingualEmbedding(id2symbols, dim=encoder_dim)
        self.model = TacoT2U(self.model_config)
        self.loss_func = PRFramewiseLoss()

    def build_optimized_model(self):
        return nn.ModuleList([self.embedding_model, self.model])

    def build_saver(self):
        self.saver = Saver(self.data_configs, self.log_dir, self.result_dir)
        return self.saver

    def common_step(self, batch, batch_idx, train=True):
        emb_texts = self.embedding_model(batch[3])
        # (emb_texts, text_lengths, units, max_len, output_lengths, spks, lang_ids)
        inputs = (emb_texts, batch[4], batch[6], batch[5], batch[7], batch[3], batch[9])
        self.model.decoder.teacher_forcing_ratio = schedule_f(self.global_step + 1)
        output, alignments = self.model(inputs)
        # print(output.shape)
        # print(alignments.shape)
        # print(batch[6].shape)
        loss = self.loss_func(batch[6], output)
        loss_dict = {
            "Total Loss": loss,
        }

        return loss_dict, output, alignments

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        """
        batch: (
            ids,
            raw_texts,
            speaker_args,
            torch.from_numpy(texts).long(),
            torch.from_numpy(text_lens),
            max(text_lens),
            torch.from_numpy(units).long(),
            torch.from_numpy(unit_lens),
            max(unit_lens),
            torch.from_numpy(lang_ids).long(),
            target_symbol_ids
        )
        """
        assert len(batch) == 11, f"data with 11 elements, but get {len(batch)}"
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 11, f"data with 11 elements, but get {len(batch)}"
    
    def training_step(self, batch, batch_idx):
        train_loss_dict, predictions, alignment = self.common_step(batch, batch_idx, train=True)

        mask = (batch[6] != 0)
        acc = ((batch[6] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
        self.log_dict({"Train/Acc": acc.item()}, sync_dist=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        self.log("Schedule sampling ratio", schedule_f(self.global_step + 1), sync_dist=True)

        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions,
                '_batch': batch, 'symbol_id': batch[10][0], 'alignment': alignment}

    def validation_step(self, batch, batch_idx):
        val_loss_dict, predictions, alignment = self.common_step(batch, batch_idx)

        mask = (batch[6] != 0)
        acc = ((batch[6] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
        self.log_dict({"Val/Acc": acc.item()}, sync_dist=True)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, 
                '_batch': batch, 'symbol_id': batch[10][0], 'alignment': alignment}

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
