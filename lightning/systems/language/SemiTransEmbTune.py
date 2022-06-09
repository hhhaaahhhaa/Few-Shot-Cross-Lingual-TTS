import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from lightning.systems.system import System
from lightning.utils.log import loss2dict
from lightning.utils.tool import LightningMelGAN, generate_reference
from lightning.model import get_reference_extractor_cls
from lightning.model.phoneme_embedding import PhonemeEmbedding
from lightning.model.phoneme_embedding2 import MultilingualTablePhonemeEmbedding
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.callbacks.language.baseline_saver import Saver
from lightning.callbacks import GlobalProgressBar
from Objects.visualization import CodebookAnalyzer
from Objects.config import LanguageDataConfigReader
from transformer import Constants
import Define
from text.define import LANG_ID2SYMBOLS


class SemiTransEmbTuneSystem(System):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_codebook_type()

        # tests
        self.codebook_analyzer = CodebookAnalyzer(self.result_dir)
        self.test_list = {
            "azure": self.generate_azure_wavs,
        }

    def build_model(self):
        self.embedding_model = PhonemeEmbedding(self.model_config, self.algorithm_config)
        self.model = FastSpeech2(self.preprocess_config, self.model_config, self.algorithm_config)
        self.loss_func = FastSpeech2Loss(self.preprocess_config, self.model_config)

        self.vocoder = LightningMelGAN()
        self.vocoder.freeze()

        self.lang_id = self.preprocess_config["lang_id"]
        d_word_vec = self.model_config["transformer"]["encoder_hidden"]
        self.emb_layer = MultilingualTablePhonemeEmbedding(LANG_ID2SYMBOLS, d_word_vec)

        self.reference_extractor = get_reference_extractor_cls(Define.UPSTREAM)()
        self.reference_extractor.freeze()

    def build_optimized_model(self):
        return nn.ModuleList([self.emb_layer, self.model])

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

    def tune_init(self, data_config):
        print("Generate reference...")
        repr_info = generate_reference(
            path=data_config["subsets"]["train"], 
            data_parser=Define.DATAPARSERS[data_config["name"]],
            lang_id=data_config["lang_id"]
        )

        self.reference_extractor.cuda()

        with torch.no_grad():
            ref_phn_feats = self.reference_extractor.extract(repr_info, norm=False, batch_size=16)
            embedding = self.embedding_model.get_new_embedding(self.codebook_type, ref_phn_feats=ref_phn_feats, lang_id=self.lang_id)
            embedding = embedding.squeeze(0)
            embedding.requires_grad = True
            # print(embedding[:, :10])
            self.emb_layer.tables[f"table-{data_config['lang_id']}"].copy_(embedding)

            # visualization
            self.visualize_matching(ref_phn_feats)
        
        self.reference_extractor.cpu()
        print("Generate reference done.")

        # tune partial model
        for p in self.emb_layer.parameters():
            p.requires_grad = False
        # for p in self.model.encoder.parameters():
        #     p.requires_grad = False
        # for p in self.model.variance_adaptor.parameters():
        #     p.requires_grad = False
        # for p in self.model.decoder.parameters():
        #     p.requires_grad = False

    def get_unsup_representation(self, repr_info):
        with torch.no_grad():
            unsup_repr = self.reference_extractor.extract(repr_info, norm=False, no_text=True)
        unsup_repr = self.embedding_model.get_new_embedding(self.codebook_type, ref_phn_feats=unsup_repr)

        if Define.DEBUG:
            print("Unsup Representation shape ", unsup_repr.shape)
        return unsup_repr

    def u_common_step(self, u_batch, batch_idx, train=True):
        # unsupervised loss
        u_batch_data, u_repr_info = u_batch
        if Define.DEBUG:
            print("Check IDs:")
            print(u_batch_data[0])
        unsup_repr = self.get_unsup_representation(u_repr_info)
        u_output = self.model(u_batch_data[2], unsup_repr, *(u_batch_data[4:]))
        u_loss = self.loss_func(u_batch_data, u_output)

        return u_loss
    
    def s_common_step(self, s_batch, batch_idx, train=True):
        # print(self.emb_layer._parameters['weight'].requires_grad)
        if getattr(self, "fix_spk_args", None) is None:  # Save one speaker args in tuning stage for inference usage.
            self.fix_spk_args = s_batch[2]
        emb_table = self.emb_layer.get_new_embedding(lang_id=self.lang_id)
        emb_texts = F.embedding(s_batch[3], emb_table, padding_idx=0)

        s_output = self.model(s_batch[2], emb_texts, *(s_batch[4:]))
        s_loss = self.loss_func(s_batch, s_output)
        return s_loss, s_output

    def synth_step(self, batch, batch_idx):  # only used when tune
        emb_table = self.emb_layer.get_new_embedding(lang_id=self.lang_id)
        emb_texts = F.embedding(batch[3], emb_table, padding_idx=0)
        output = self.model(self.fix_spk_args, emb_texts, *(batch[4:6]), average_spk_emb=True)
        return output

    # def text_synth_step(self, batch, batch_idx):  # only used when predict (use TextDataset2)
    #     emb_texts = self.emb_layer(batch[2])
    #     output = self.model(self.fix_spk_args, emb_texts, *(batch[3:5]), average_spk_emb=True)
    #     return output
    #     # ids,
    #     # raw_texts,
    #     # torch.from_numpy(texts).long(),
    #     # torch.from_numpy(text_lens),
    #     # max(text_lens),
    
    def check_s_batch(self, s_batch):
        assert len(s_batch) == 12, "data with 12 elements"

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
        if Define.DEBUG:
            print("Sup/Unsup training loss:")
            print(s_train_loss[0], u_train_loss[0])
        for i in range(len(s_train_loss)):
            train_loss.append(1.0 * s_train_loss[i] + 0.0 * u_train_loss[i])

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss[0], 'losses': train_loss, 'output': predictions, '_batch': batch["sup"]}

    def validation_step(self, batch, batch_idx):
        val_loss, predictions = self.s_common_step(batch, batch_idx)
        synth_predictions = self.synth_step(batch, batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': batch, 'synth': synth_predictions}     

    def visualize_matching(self, ref_phn_feats):
        matching = self.embedding_model.get_matching(self.codebook_type, ref_phn_feats=ref_phn_feats, lang_id=self.lang_id)
        self.codebook_analyzer.visualize_matching(0, matching)

    def test_step(self, batch, batch_idx):
        outputs = {}
        for test_name, test_fn in getattr(self, "test_list", {}).items(): 
            outputs[test_name] = test_fn(batch, batch_idx)

        return outputs

    def generate_azure_wavs(self, batch, batch_idx):
        synth_predictions = self.synth_step(batch, batch_idx)
        return {'_batch': batch, 'synth': synth_predictions}
