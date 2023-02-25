import numpy as np
import torch
import torch.nn as nn

from dlhlp_lib.s3prl import S3PRLExtractor

import Define
from lightning.build import build_all_speakers, build_id2symbols
from lightning.systems.system import System
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.model.reduction import PhonemeQueryExtractor
from lightning.utils.tool import ssl_match_length
from lightning.callbacks.language.baseline_saver import Saver
from .embeddings import *
from ..t2u.downstreams import Downstream1


class TransEmbTuneSystem(System):
    """ 
    Tune version of TransEmb system.
    """

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
        self.embedding_model = MultilingualEmbedding(
            id2symbols=build_id2symbols(self.data_configs), dim=encoder_dim)
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
        return nn.ModuleList([self.embedding_model, self.model])

    def build_saver(self):
        self.saver = Saver(self.data_configs, self.model_config, self.log_dir, self.result_dir)
        return self.saver
    
    def generate_reference_info(self, data_config):
        from dlhlp_lib.utils import batchify, segment2duration
        from Parsers.utils import read_queries_from_txt
        from text import text_to_sequence

        data_parser = Define.DATAPARSERS[data_config["name"]]
        lang_id = data_config["lang_id"]
        symbol_id = data_config["symbol_id"]
        queries = read_queries_from_txt(data_config["subsets"]["train"])

        infos = []
        # Extract representation information batchwise
        for query_batch in batchify(queries, batch_size=16):
            info = {
                "raw_feat": [],
                "lens": [],
                "max_len": None,
                "phonemes": [],
                "avg_frames": [],
                "lang_id": lang_id,
                "symbol_id": symbol_id,
            }
            for query in query_batch:
                # Transfer learning module
                segment = data_parser.mfa_segment.read_from_query(query)
                if Define.UPSTREAM == "mel":
                    pass  # TODO: Mel version
                else:
                    raw_feat = data_parser.wav_trim_16000.read_from_query(query)
                    avg_frames = segment2duration(segment, fp=0.02)
                    info["raw_feat"].append(torch.from_numpy(raw_feat).float())
                    info["avg_frames"].append(avg_frames)
                    info["lens"].append(sum(avg_frames))

                phns = data_parser.phoneme.read_from_query(query)
                phns = f"{{{phns}}}"  # match input format of text_to_sequence()
                phns = text_to_sequence(phns, data_config["text_cleaners"], lang_id)
                info["phonemes"].append(phns)
            info["lens"] = torch.LongTensor(info["lens"])
            info["max_len"] = max(info["lens"])
            infos.append(info)
        
        return infos

    def tune_init(self, data_configs):
        from text.define import LANG_ID2SYMBOLS

        assert len(data_configs) == 1
        print("Generate reference...")
        ref_infos = self.generate_reference_info(data_configs[0])
        self.target_lang_id = ref_infos[0]["lang_id"]
        print(f"Target Language: {self.target_lang_id}.")

        # Merge all information and perform embedding layer initialization
        print("Embedding initialization...")
        self.cuda()
        self.upstream.eval()
        with torch.no_grad():
            hiddens, avg_frames_list, phonemes_list = [], [], []
            for info in ref_infos:
                ssl_repr, _ = self.upstream.extract(info["raw_feat"])  # B, L, n_layers, dim
                ssl_repr = ssl_match_length(ssl_repr, info["max_len"].item())
                x = self.embedding_generator(ssl_repr, info["lens"].cuda())
                hiddens.extend([x1 for x1 in x])
                avg_frames_list.extend(info["avg_frames"])
                phonemes_list.extend(info["phonemes"])
            
            table = self.phoneme_query_extractor(hiddens, avg_frames_list, 
                                len(LANG_ID2SYMBOLS[ref_infos[0]["lang_id"]]), phonemes_list)  # 1, n_symbols, dim
            table = table.squeeze(0)  # n_symbols, dim
            # print(table.shape)
            self.embedding_model.tables[f"table-{ref_infos[0]['symbol_id']}"].copy_(table)
        for p in self.embedding_model.parameters():
            p.requires_grad = True
        self.cpu()
        
        # tune part
        # for p in self.emb_layer.parameters():
        #     p.requires_grad = False
        # for p in self.model.encoder.parameters():
        #     p.requires_grad = False
        # for p in self.model.variance_adaptor.parameters():
        #     p.requires_grad = False
        # for p in self.model.decoder.parameters():
        #     p.requires_grad = False

    def build_optimized_model(self):
        return nn.ModuleList([self.model, self.embedding_model])
    
    def build_saver(self):
        saver = Saver(self.data_configs, self.model_config, self.log_dir, self.result_dir)
        return saver

    def common_step(self, batch, batch_idx, train=True):
        emb_texts = self.embedding_model(batch[3])
        output = self.model(batch[2], emb_texts, *(batch[4:]))
        loss = self.loss_func(batch[:-1], output)
        loss_dict = {
            "Total Loss"       : loss[0],
            "Mel Loss"         : loss[1],
            "Mel-Postnet Loss" : loss[2],
            "Pitch Loss"       : loss[3],
            "Energy Loss"      : loss[4],
            "Duration Loss"    : loss[5],
        }
        return loss_dict, output
    
    def synth_step(self, batch, batch_idx):
        emb_texts = self.embedding_model(batch[3])
        output = self.model(batch[2], emb_texts, *(batch[4:6]), lang_args=batch[-1], average_spk_emb=True)
        return output

    # def text_synth_step(self, batch, batch_idx):  # only used when inference (use TextDataset2)
    #     # TODO: try to determine a fix spk_args from training data, try to update to v2
    #     emb_texts = F.embedding(batch[2], self.embedding_model.get_new_embedding("table-sep", lang_id=self.lang_id, init=False), padding_idx=0)
    #     output = self.model(self.fix_spk_args, emb_texts, *(batch[3:5]), average_spk_emb=True)
    #     return output
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 13, f"data with 13 elements, but get {len(batch)}"
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 13, f"data with 13 elements, but get {len(batch)}"
    
    def training_step(self, batch, batch_idx):
        train_loss_dict, output = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': output, '_batch': batch}

    def validation_step(self, batch, batch_idx):
        val_loss_dict, predictions = self.common_step(batch, batch_idx, train=False)
        synth_predictions = self.synth_step(batch, batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': batch, 'synth': synth_predictions}
    