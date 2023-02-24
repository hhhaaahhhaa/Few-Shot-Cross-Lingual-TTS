import torch
import torch.nn as nn
import torch.nn.functional as F
import jiwer
from tqdm import tqdm
import json
import yaml
from collections import OrderedDict

from dlhlp_lib.utils.tool import get_mask_from_lengths
from dlhlp_lib.s3prl import S3PRLExtractor
from dlhlp_lib.transformers import CodebookAttention

import Define
from text.define import LANG_NAME2ID
from lightning.build import build_id2symbols
from lightning.systems import System
from lightning.callbacks.t2u.saver import Saver
from lightning.utils.tool import ssl_match_length
from lightning.model.reduction import PhonemeQueryExtractor
from ..language.embeddings import MultilingualEmbedding
from ..language.FastSpeech2 import BaselineSystem
from ..phoneme_recognition.loss import PRFramewiseLoss
from .tacotron2.tacot2u import TacoT2U
from .tacotron2.hparams import hparams
from .downstreams import Downstream1, Downstream2
from .modules import DA
from Objects.config import LanguageDataConfigReader


class TransEmbDAE2ETuneSystem(System):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        t2u_model_config = self.model_config
        id2symbols = build_id2symbols(self.data_configs)
        n_units = len(id2symbols[self.data_configs[0]["target"]["unit_name"]])   # all target unit names from data configs should be the same!
        setattr(hparams, "n_units", n_units)
        encoder_dim = t2u_model_config["tacotron2"]["symbols_embedding_dim"]
        self.embedding_model = MultilingualEmbedding(id2symbols, dim=encoder_dim)
        self.model = TacoT2U(t2u_model_config)
        self.loss_func = PRFramewiseLoss()

        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.embedding_generator = Downstream1(
            t2u_model_config,
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            specific_layer=Define.LAYER_IDX
        )
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)

        self.build_DA()
        self.build_u2s()

    def build_DA(self):
        self.DA = DA(hparams.n_units)
        self.da_loss_func = nn.BCEWithLogitsLoss()

    def build_u2s(self):
        # Load tuned u2s
        with open(self.model_config["u2s"]["model_cards"], 'r', encoding='utf-8') as f:  # "evaluation/_exp1/model.json"
            MODEL_CARDS = json.load(f)
        config_reader = LanguageDataConfigReader()
        model_info = MODEL_CARDS[self.model_config["u2s"]["model_name"]]  # "u2s-zhkofrdees-hubert_large_ll60k-24-512c"

        # Load model and data
        data_configs = [config_reader.read(path) for path in model_info["config_paths"]]  # reconstruct data config in local since data path is different
        self.u2s = BaselineSystem.load_from_checkpoint(model_info["ckpt"], data_configs=data_configs)
        self.u2s.eval()
        self.u2s_loss_func = self.u2s.loss_func

    def build_optimized_model(self):
        return nn.ModuleList([self.DA, self.embedding_model, self.model])

    def build_saver(self):
        saver = Saver(self.data_configs, self.log_dir, self.result_dir)
        return saver

    def generate_reference_info(self, data_config):
        from dlhlp_lib.utils import batchify, segment2duration
        from Parsers.utils import read_queries_from_txt
        from text import text_to_sequence

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
        print("Generate reference done.")
        
        return {
            "hiddens": hiddens,
            "avg_frames_list": avg_frames_list,
            "phonemes_list": phonemes_list,
            "lang_id": lang_id,
            "symbol_id": symbol_id,
        }

    def tune_init(self, data_configs):
        from text.define import LANG_ID2SYMBOLS

        assert len(data_configs) > 1
        ref_info = self.generate_reference_info(data_configs[0])
        self.target_lang_id = ref_info["lang_id"]
        print(f"Target Language: {self.target_lang_id}.")

        # Merge all information and perform embedding layer initialization
        with torch.no_grad():
            table = self.phoneme_query_extractor(ref_info["hiddens"], ref_info["avg_frames_list"], 
                                len(LANG_ID2SYMBOLS[ref_info["lang_id"]]), ref_info["phonemes_list"])  # 1, n_symbols, dim
            table = table.squeeze(0)  # n_symbols, dim
            # print(table.shape)
            self.embedding_model.tables[f"table-{ref_info['symbol_id']}"].copy_(table)
        for p in self.embedding_model.parameters():
            p.requires_grad = True
        self._set_tune_stage(0)
        self.cpu()

    def _set_tune_stage(self, stage):
        self.tune_stage = stage
        if stage == 0:
            for p in self.embedding_model.parameters():
                p.requires_grad = False
            for p in self.model.parameters():
                p.requires_grad = False
        elif stage == 1:
            for p in self.embedding_model.parameters():
                p.requires_grad = True
            for p in self.model.parameters():
                p.requires_grad = True
        else:
            raise NotImplementedError

    def common_t2u_step(self, batch, batch_idx, train=True):
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
            "T2U Loss": loss,
        }

        return loss_dict, output, alignments

    def common_u2s_step(self, logits, batch, batch_idx, train=True):
        table = self.u2s.embedding_model.tables["table-zhkofrdees-hubert_large_ll60k-24-512c"]
        emb_units = logits @ table 
        output = self.u2s.model(batch[2], emb_units, *(batch[4:]))
        loss = self.u2s_loss_func(batch[:-1], output)
        loss_dict = {
            "U2S Loss"         : loss[0],
            "Mel Loss"         : loss[1],
            "Mel-Postnet Loss" : loss[2],
            "Pitch Loss"       : loss[3],
            "Energy Loss"      : loss[4],
            "Duration Loss"    : loss[5],
        }
        return loss_dict, output

    def common_da_step(self, f_logits, logit_lengths, batch, batch_idx, train=True):
        f_label = torch.zeros(len(f_logits)).to(self.device).float()
        unsup_label = (batch[5] == LANG_NAME2ID[self.target_lang_id]).float()
        unsup_logits = self.DA.embed_discrete(batch[2])

        f_score = self.DA(f_logits, logit_lengths, gradient_reverse=True)
        unsup_score = self.DA(unsup_logits, batch[3])
        
        g_loss = self.da_loss_func(f_score, f_label)
        d_loss = self.da_loss_func(unsup_score, unsup_label)
        loss_dict = {
            "DA Loss": g_loss + d_loss,
            "G loss": g_loss,
            "D loss": d_loss
        }
        return loss_dict, unsup_score
    
    def common_step(self, batch, batch_idx, train=True):
        if self.global_step == self.train_config["optimizer"]["anneal_steps"][0]:
            self._set_tune_stage(1)
        t2u_batch, u2s_batch, da_batch = batch["t2u2s"]["t2u"], batch["t2u2s"]["u2s"], batch["da"]
        t2u_loss_dict, t2u_output, alignments = self.common_t2u_step(t2u_batch, batch_idx, train)

        # note that t2u append <eos> at the end so need to adjust the shape
        logits = F.softmax(t2u_output[:, :-1, :], dim=2)
        unit_lengths = t2u_batch[7] - 1

        u2s_loss_dict, u2s_output = self.common_u2s_step(logits, u2s_batch, batch_idx, train)
        da_loss_dict, da_output = self.common_da_step(logits, unit_lengths, da_batch, batch_idx, train)

        loss_dict = t2u_loss_dict
        loss_dict.update(u2s_loss_dict)
        loss_dict.update(da_loss_dict)
        lamd_t2u = schedule_lamd_t2u(self.global_step + 1)
        loss_dict["Total Loss"] = lamd_t2u * loss_dict["T2U Loss"] + loss_dict["U2S Loss"] + loss_dict["DA Loss"]

        # check entropy
        h_loss = self.DA.check_entropy(logits, unit_lengths)
        loss_dict["entropy"] = h_loss
        
        return loss_dict, t2u_output, alignments, da_output

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
        assert len(batch["t2u2s"]["t2u"]) == 11, f"data with 11 elements, but get {len(batch)}"
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch["t2u2s"]["t2u"]) == 11, f"data with 11 elements, but get {len(batch)}"
    
    def training_step(self, batch, batch_idx):
        train_loss_dict, predictions, alignment, da_output = self.common_step(batch, batch_idx, train=True)

        da_batch = batch["da"]
        batch = batch["t2u2s"]["t2u"]
        mask = (batch[6] != 0)
        acc = ((batch[6] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
        self.log("Train/Acc", acc.item(), sync_dist=True)

        # DA acc
        unsup_label = (da_batch[5] == LANG_NAME2ID[self.target_lang_id]).float()
        da_acc = (unsup_label == (da_output > 0)).sum() / len(unsup_label)
        self.log("Train/DA Acc", da_acc.item(), sync_dist=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, batch_size=4, sync_dist=True)
        self.log("Schedule sampling ratio", schedule_f(self.global_step + 1), sync_dist=True)

        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions,
                '_batch': batch, 'symbol_id': batch[10][0], 'alignment': alignment}

    def validation_step(self, batch, batch_idx):
        val_loss_dict, predictions, alignment, da_output = self.common_step(batch, batch_idx, train=False)

        da_batch = batch["da"]
        batch = batch["t2u2s"]["t2u"]
        mask = (batch[6] != 0)
        acc = ((batch[6] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
        self.log("Val/Acc", acc.item(), sync_dist=True)

        # DA acc
        unsup_label = (da_batch[5] == LANG_NAME2ID[self.target_lang_id]).float()
        da_acc = (unsup_label == (da_output > 0)).sum() / len(unsup_label)
        self.log("Val/DA Acc", da_acc.item(), sync_dist=True)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, batch_size=4, sync_dist=True)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, 
                '_batch': batch, 'symbol_id': batch[10][0], 'alignment': alignment}

    def on_save_checkpoint(self, checkpoint):
        """ (Hacking!) Remove pretrained weights in checkpoint to save disk space. """
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k in state_dict:
            if k.split('.')[0] in ["upstream", "u2s", "DA", "embedding_generator"]:
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


class TransEmbCDAE2ETuneSystem(TransEmbDAE2ETuneSystem):
    """
    Change Downstream1 to Downstream2
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        t2u_model_config = self.model_config["t2u"]
        id2symbols = build_id2symbols(self.data_configs)
        n_units = len(id2symbols[self.data_configs[0]["target"]["unit_name"]])   # all target unit names from data configs should be the same!
        setattr(hparams, "n_units", n_units)
        encoder_dim = t2u_model_config["tacotron2"]["symbols_embedding_dim"]
        self.embedding_model = MultilingualEmbedding(id2symbols, dim=encoder_dim)
        self.model = TacoT2U(t2u_model_config)
        self.loss_func = PRFramewiseLoss()

        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.embedding_generator = Downstream2(
            t2u_model_config,
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            specific_layer=Define.LAYER_IDX
        )
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)

        super().build_u2s()
        super().build_DA()


class TransEmbC2DAE2ETuneSystem(TransEmbDAE2ETuneSystem):
    """
    Passed through Codebook after PhonemeQuery extraction
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        t2u_model_config = self.model_config["t2u"]
        id2symbols = build_id2symbols(self.data_configs)
        n_units = len(id2symbols[self.data_configs[0]["target"]["unit_name"]])   # all target unit names from data configs should be the same!
        setattr(hparams, "n_units", n_units)
        encoder_dim = t2u_model_config["tacotron2"]["symbols_embedding_dim"]
        self.embedding_model = MultilingualEmbedding(id2symbols, dim=encoder_dim)
        self.model = TacoT2U(t2u_model_config)
        self.loss_func = PRFramewiseLoss()

        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.embedding_generator = Downstream1(
            t2u_model_config,
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            specific_layer=Define.LAYER_IDX
        )
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)
        self.codebook_attention = CodebookAttention(
            codebook_size=t2u_model_config["codebook_size"],
            embed_dim=t2u_model_config["transformer"]["d_model"],
            num_heads=t2u_model_config["transformer"]["nhead"],
        )

        super().build_u2s()
        super().build_DA()

    def tune_init(self, data_configs):
        from text.define import LANG_ID2SYMBOLS

        assert len(data_configs) == 1
        ref_info = super().generate_reference_info(data_configs[0])

        # Merge all information and perform embedding layer initialization
        with torch.no_grad():
            table_pre = self.phoneme_query_extractor(ref_info["hiddens"], ref_info["avg_frames_list"], 
                                len(LANG_ID2SYMBOLS[ref_info["lang_id"]]), ref_info["phonemes_list"])  # 1, n_symbols, dim
            table, _ = self.codebook_attention(table_pre)
            table = table.squeeze(0)  # n_symbols, dim
            # print(table.shape)
            self.embedding_model.tables[f"table-{ref_info['symbol_id']}"].copy_(table)
        for p in self.embedding_model.parameters():
            p.requires_grad = True
        super()._set_tune_stage(0)
        self.cpu()


def schedule_f(step: int) -> float:
    return 1.0


def schedule_lamd_t2u(step: int) -> float:
    # return 1.0
    return max(0.01, 1 - step / 500)