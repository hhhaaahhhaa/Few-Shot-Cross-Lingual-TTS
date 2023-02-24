import torch
import torch.nn as nn
import torch.nn.functional as F
import jiwer
from tqdm import tqdm
from collections import OrderedDict

from dlhlp_lib.s3prl import S3PRLExtractor

import Define
from lightning.build import build_id2symbols
from lightning.systems import System
from lightning.callbacks.t2u.saver import Saver
from lightning.utils.tool import ssl_match_length
from lightning.model.reduction import PhonemeQueryExtractor
from ..language.embeddings import MultilingualEmbedding
from ..phoneme_recognition.loss import PRFramewiseLoss
from .tacotron2.tacot2u import TacoT2U
from .tacotron2.hparams import hparams
from .downstreams import Downstream1


class TransEmbTuneSystem(System):

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

        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.embedding_generator = Downstream1(
            self.model_config,
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            specific_layer=Define.LAYER_IDX
        )
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)

    def build_optimized_model(self):
        return nn.ModuleList([self.embedding_model, self.model])

    def build_saver(self):
        saver = Saver(self.data_configs, self.log_dir, self.result_dir)
        return saver

    def tune_init(self, data_configs):
        from dlhlp_lib.utils import batchify, segment2duration
        from Parsers.utils import read_queries_from_txt
        from text.define import LANG_ID2SYMBOLS
        from text import text_to_sequence
        
        assert len(data_configs) == 1
        data_config = data_configs[0]

        print("Generate reference...")
        data_parser = Define.DATAPARSERS[data_config["name"]]
        lang_id = data_config["lang_id"]
        symbol_id = data_config["symbol_id"]
        queries = read_queries_from_txt(data_config["subsets"]["train"])

        self.cuda()  # Move to GPU
        hiddens = []
        avg_frames_list = []
        phonemes_list = []

        # Extract representation information batchwise
        for query_batch in batchify(queries, batch_size=16):
            info = {
                "raw_feat": [],
                "max_len": [],
                "lens": []
            }
            for query in query_batch:
                # Transfer learning module
                segment = data_parser.mfa_segment.read_from_query(query)
                if Define.UPSTREAM == "mel":
                    pass  # TODO: Mel version
                else:
                    raw_feat = data_parser.wav_trim_16000.read_from_query(query)
                    avg_frames = segment2duration(segment, fp=0.02)
                    info["raw_feat"].append(torch.from_numpy(raw_feat).float().cuda())
                    avg_frames_list.append(avg_frames)
                    info["lens"].append(sum(avg_frames))

                phns = data_parser.phoneme.read_from_query(query)
                phns = f"{{{phns}}}"  # match input format of text_to_sequence()
                phns = text_to_sequence(phns, data_config["text_cleaners"], lang_id)
                phonemes_list.append(phns)
            info["lens"] = torch.LongTensor(info["lens"]).cuda()
            info["max_len"] = max(info["lens"])

            self.upstream.eval()
            with torch.no_grad():
                ssl_repr, _ = self.upstream.extract(info["raw_feat"])  # B, L, n_layers, dim
                ssl_repr = ssl_match_length(ssl_repr, info["max_len"].item())
                x = self.embedding_generator(ssl_repr, info["lens"])
                hiddens.extend([x1 for x1 in x])

        # Merge all information and perform embedding layer initialization
        with torch.no_grad():
            table = self.phoneme_query_extractor(hiddens, avg_frames_list, 
                                len(LANG_ID2SYMBOLS[lang_id]), phonemes_list)  # 1, n_symbols, dim
            table = table.squeeze(0)  # n_symbols, dim
            # print(table.shape)
            self.embedding_model.tables[f"table-{symbol_id}"].copy_(table)
        for p in self.embedding_model.parameters():
            p.requires_grad = True
        print("Generate reference done.")
        self.cpu()
        # for p in self.embedding_model.parameters():
        #     print(p.requires_grad)

        # tune partial model
        for p in self.embedding_model.parameters():
            p.requires_grad = False
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in self.model.decoder.parameters():
            p.requires_grad = False
        for p in self.model.decoder.linear_projection.parameters():
            p.requires_grad = True
        for p in self.model.decoder.final_proj.parameters():
            p.requires_grad = True

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

    def on_save_checkpoint(self, checkpoint):
        """ (Hacking!) Remove pretrained weights in checkpoint to save disk space. """
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k in state_dict:
            if k.split('.')[0] == "upstream" or k.split('.')[0] == "embedding_generator":
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


def schedule_f(step: int) -> float:
    return 1.0
    # return max(0.5, 1 - step / 20000)
    # else:
    #     return max(0, 0.5 - (step - 20000) / 20000)
