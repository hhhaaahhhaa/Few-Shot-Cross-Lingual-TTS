import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from lightning.systems.adaptor import AdaptorSystem
from lightning.utils.log import loss2dict
from lightning.utils.tool import LightningMelGAN
from lightning.model.phoneme_embedding import PhonemeEmbedding
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.model import get_reference_extractor_cls
from lightning.callbacks.language.fscl_saver import Saver
from Objects.visualization import CodebookAnalyzer
from lightning.model.reference_extractor import HubertExtractor, XLSR53Extractor, Wav2Vec2Extractor, MelExtractor
import Define
from transformer import Constants


class TransEmbSystem(AdaptorSystem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_codebook_type()

        # tests
        self.codebook_analyzer = CodebookAnalyzer(self.result_dir)
        # self.test_list = {
        #     "codebook visualization": self.visualize_matching,
        # }

    def build_model(self):
        self.embedding_model = PhonemeEmbedding(self.model_config, self.algorithm_config)
        self.model = FastSpeech2(self.preprocess_config, self.model_config, self.algorithm_config)
        self.loss_func = FastSpeech2Loss(self.preprocess_config, self.model_config)

        self.vocoder = LightningMelGAN()
        self.vocoder.freeze()

        self.reference_extractor = get_reference_extractor_cls(Define.UPSTREAM)()
        self.reference_extractor.freeze()

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
        _, _, repr_info, lang_id = batch[0]
        with torch.no_grad():
            ref_phn_feats = self.reference_extractor.extract(repr_info, norm=False)

        embedding = self.embedding_model.get_new_embedding(self.codebook_type, ref_phn_feats=ref_phn_feats, lang_id=lang_id)
        embedding = embedding.squeeze(0)
        embedding[Constants.PAD].fill_(0)

        if Define.DEBUG:
            print(embedding.requires_grad)
            print("Embedding shape and gradient required: ", embedding.shape)
            print(embedding.requires_grad)
        return embedding

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
