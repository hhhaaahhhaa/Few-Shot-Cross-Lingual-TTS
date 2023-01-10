import torch
import torch.nn as nn
import torch.nn.functional as F
import jiwer

from dlhlp_lib.s3prl import S3PRLExtractor

import Define
from lightning.build import build_id2symbols
from lightning.systems.adaptor import AdaptorSystem
from lightning.callbacks.t2u.saver import Saver
from Objects.visualization import CodebookAnalyzer
from lightning.model.reduction import PhonemeQueryExtractor
from ..phoneme_recognition.loss import PRFramewiseLoss
from ..language.embeddings import MultilingualEmbedding
from .tacotron2.tacot2u import TacoT2U
from .tacotron2.hparams import hparams


class TransEmbSystem(AdaptorSystem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_codebook_type()

        self.codebook_analyzer = CodebookAnalyzer(self.result_dir)

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
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
        return saver
    
    def init_codebook_type(self):        
        codebook_config = self.algorithm_config["adapt"]["phoneme_emb"]
        if codebook_config["type"] == "embedding":
            self.codebook_type = "table-sep"
        elif codebook_config["type"] == "codebook":
            self.codebook_type = codebook_config["attention"]["type"]
        else:
            raise NotImplementedError
        self.adaptation_steps = self.algorithm_config["adapt"]["train"]["steps"]

    def build_embedding_table(self, batch):
        _, _, repr_info = batch[0]

        self.upstream.eval()
        with torch.no_grad():
            ssl_repr, _ = self.upstream.extract(repr_info["sup_wav"])  # B, L, n_layers, dim
            ssl_repr = ssl_repr.detach()

        x = self.embedding_generator(ssl_repr, repr_info["sup_lens"].cpu())
        table = self.phoneme_query_extractor(x, repr_info["sup_avg_frames"], 
                            repr_info["n_symbols"], repr_info["sup_phonemes"])  # 1, n_symbols, n_layers, dim
        table = table.squeeze(0)  # n_symbols, dim
        
        # print("Table shape and gradient required: ", table.shape)
        # print(table.requires_grad)
        
        return table

    def common_step(self, batch, batch_idx, train=True):
        if Define.DEBUG:
            print("Extract embedding... ")
        emb_table = self.build_embedding_table(batch)
        _, qry_batch, _, _ = batch[0]
        qry_batch = qry_batch[0]
        emb_texts = F.embedding(qry_batch[3], emb_table, padding_idx=0)
        output = self.model(qry_batch[2], emb_texts, *(qry_batch[4:]))
        loss = self.loss_func(qry_batch, output)
        return loss, output

    def training_step(self, batch, batch_idx):
        train_loss, predictions = self.common_step(batch, batch_idx, train=True)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss[0], 'losses': train_loss, 'output': predictions, '_batch': qry_batch}

    def validation_step(self, batch, batch_idx):
        self.log_matching(batch, batch_idx)
        val_loss, predictions = self.common_step(batch, batch_idx)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': qry_batch}
    
    def visualize_matching(self, batch, batch_idx):
        if self.codebook_type != "table-sep":
            _, _, repr_info, lang_id = batch[0]
            with torch.no_grad():
                ref_phn_feats = self.reference_extractor.extract(repr_info, norm=False)
            matching = self.embedding_model.get_matching(self.codebook_type, ref_phn_feats=ref_phn_feats, lang_id=lang_id)
            self.codebook_analyzer.visualize_matching(batch_idx, matching)
        return None

    def log_matching(self, batch, batch_idx, stage="val"):
        step = self.global_step + 1
        _, _, repr_info, lang_id = batch[0]
        with torch.no_grad():
            ref_phn_feats = self.reference_extractor.extract(repr_info, norm=False)
        
        matchings = self.embedding_model.get_matching(self.codebook_type, ref_phn_feats=ref_phn_feats, lang_id=lang_id)
        for matching in matchings:
            fig = self.codebook_analyzer.plot_matching(matching, quantized=False)
            figure_name = f"{stage}/step_{step}_{batch_idx:03d}_{matching['title']}"
            self.logger.experiment.log_figure(
                figure_name=figure_name,
                figure=fig,
                step=step,
            )
            plt.close(fig)
