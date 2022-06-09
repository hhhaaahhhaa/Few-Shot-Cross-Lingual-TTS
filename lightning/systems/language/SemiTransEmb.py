import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from lightning.systems.adaptor import AdaptorSystem
from lightning.utils.log import loss2dict
from lightning.utils.tool import LightningMelGAN
from lightning.model.phoneme_embedding import PhonemeEmbedding
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.model import get_reference_extractor_cls
from lightning.callbacks.language.fscl_saver import Saver
from lightning.callbacks import GlobalProgressBar
from Objects.visualization import CodebookAnalyzer
from lightning.model.reference_extractor import HubertExtractor, XLSR53Extractor, Wav2Vec2Extractor, MelExtractor
from transformer import Constants
import Define


class SemiTransEmbSystem(AdaptorSystem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_codebook_type()

        # tests
        self.codebook_analyzer = CodebookAnalyzer(self.result_dir)
        self.test_list = {
            "codebook visualization": self.visualize_matching,
        }

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

    def build_embedding_table(self, repr_info):
        with torch.no_grad():
            ref_phn_feats = self.reference_extractor.extract(repr_info, norm=False)
        # print(ref_phn_feats.requires_grad)
        # torch.cuda.memory_summary()
        embedding = self.embedding_model.get_new_embedding(self.codebook_type, ref_phn_feats=ref_phn_feats)
        embedding = embedding.squeeze(0)
        embedding[Constants.PAD].fill_(0)

        if Define.DEBUG:
            print("Embedding shape ", embedding.shape)
        return embedding

    def get_unsup_representation(self, repr_info):
        with torch.no_grad():
            unsup_repr = self.reference_extractor.extract(repr_info, norm=False, no_text=True)
        # print(unsup_repr.requires_grad)
        # torch.cuda.memory_summary()
        unsup_repr = self.embedding_model.get_new_embedding(self.codebook_type, ref_phn_feats=unsup_repr)

        if Define.DEBUG:
            print("Unsup Representation shape ", unsup_repr.shape)
        return unsup_repr

    def s_common_step(self, s_batch, batch_idx, train=True):
        # supervised loss
        _, qry_batch, s_repr_info, lang_id = s_batch[0]
        emb_table = self.build_embedding_table(s_repr_info)
        qry_batch = qry_batch[0]
        emb_texts = F.embedding(qry_batch[3], emb_table, padding_idx=0)
        s_output = self.model(qry_batch[2], emb_texts, *(qry_batch[4:]))
        s_loss = self.loss_func(qry_batch, s_output)

        return s_loss, s_output

    def u_common_step(self, u_batch, batch_idx, train=True):
        # unsupervised loss
        u_batch_data, u_repr_info = u_batch
        unsup_repr = self.get_unsup_representation(u_repr_info)
        u_output = self.model(u_batch_data[2], unsup_repr, *(u_batch_data[4:]))
        u_loss = self.loss_func(u_batch_data, u_output)

        return u_loss

    def check_s_batch(self, s_batch):
        assert len(s_batch) == 1, "meta_batch_per_gpu"
        assert len(s_batch[0]) == 2 or len(s_batch[0]) == 4, "sup + qry (+ ref_phn_feats + lang_id)"
        assert len(s_batch[0][0]) == 1, "n_batch == 1"
        assert len(s_batch[0][0][0]) == 12, "data with 12 elements"

    def check_u_batch(self, u_batch):
        pass

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self.check_s_batch(batch["sup"])
        self.check_u_batch(batch["unsup"])
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            self.check_s_batch(batch)
        elif dataloader_idx == 1:
            self.check_u_batch(batch)
        else:
            raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        s_train_loss, predictions = self.s_common_step(batch["sup"], batch_idx, train=True)
        u_train_loss = self.u_common_step(batch["unsup"], batch_idx, train=True)
        train_loss = []
        for i in range(len(s_train_loss)):
            train_loss.append(0.5 * s_train_loss[i] + 0.5 * u_train_loss[i])

        qry_batch = batch["sup"][0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss[0], 'losses': train_loss, 'output': predictions, '_batch': qry_batch}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            self.log_matching(batch, batch_idx)
            val_loss, predictions = self.s_common_step(batch, batch_idx)
            qry_batch = batch[0][1][0]

            # Log metrics to CometLogger
            loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
            self.log_dict(loss_dict, sync_dist=True)
            return {'losses': val_loss, 'output': predictions, '_batch': qry_batch}
        elif dataloader_idx == 1:
            val_loss = self.u_common_step(batch, batch_idx)
            loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
            self.log_dict(loss_dict, sync_dist=True)
        else:
            raise NotImplementedError
    
    def visualize_matching(self, batch, batch_idx):
        if self.codebook_type != "table-sep":
            _, _, ref_phn_feats, lang_id = batch[0]
            with torch.no_grad():
                ref_phn_feats = self.reference_extractor.extract(ref_phn_feats, norm=False)
                ref_phn_feats = ref_phn_feats.squeeze(0)
                ref_phn_feats[Constants.PAD].fill_(0)

            matching = self.embedding_model.get_matching(self.codebook_type, ref_phn_feats=ref_phn_feats, lang_id=lang_id)
            self.codebook_analyzer.visualize_matching(batch_idx, matching)
        return None

    def log_matching(self, batch, batch_idx, stage="val"):
        step = self.global_step + 1
        _, _, ref_phn_feats, lang_id = batch[0]
        with torch.no_grad():
            ref_phn_feats = self.reference_extractor.extract(ref_phn_feats, norm=False)
            ref_phn_feats[0][Constants.PAD].fill_(0)
        
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

    def configure_callbacks(self):
        # PL does not support cross reduction directly for multiple dataloaders yet.
        # Checkpoint saver
        save_step = self.train_config["step"]["save_step"]
        checkpoint = ModelCheckpoint(
            monitor="Val/Total Loss/dataloader_idx_0", mode="min",
            every_n_train_steps=save_step, save_top_k=-1
        )

        # Progress bars (step/epoch)
        outer_bar = GlobalProgressBar(process_position=1)

        # Monitor learning rate / gpu stats
        lr_monitor = LearningRateMonitor()
        # gpu_monitor = GPUStatsMonitor(  stablize!
        #     memory_utilization=True, gpu_utilization=True, intra_step_time=True, inter_step_time=True
        # )
        
        # Save figures/audios/csvs
        saver = self.build_saver()
        if isinstance(saver, list):
            callbacks = [checkpoint, outer_bar, lr_monitor, *saver]
        else:
            callbacks = [checkpoint, outer_bar, lr_monitor, saver]
        return callbacks
