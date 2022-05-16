import torch.nn as nn
import torch.nn.functional as F
from ..system import System
from lightning.utils.log import loss2dict
from lightning.utils.tool import LightningMelGAN
from lightning.model.phoneme_embedding import PhonemeEmbedding
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.callbacks.baseline_saver import Saver


class BaselineTuneSystem(System):
    """
    Concrete class for baseline multilingual FastSpeech2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # tests
        self.test_list = {
            "azure": self.generate_azure_wavs, 
        }

    def build_model(self):
        self.embedding_model = PhonemeEmbedding(self.model_config, self.algorithm_config)
        self.model = FastSpeech2(self.preprocess_config, self.model_config, self.algorithm_config)
        self.loss_func = FastSpeech2Loss(self.preprocess_config, self.model_config)
        
        # Although the vocoder is only used in callbacks, we need it to be
        # moved to cuda for faster inference, so it is initialized here with the
        # model, and let pl.Trainer handle the DDP devices.
        self.vocoder = LightningMelGAN()
        self.vocoder.freeze()

        # Tune
        self.model.freeze()

        # # Tune
        self.lang_id = self.preprocess_config["lang_id"]
        print("Current language: ", self.lang_id)

    def build_optimized_model(self):
        return nn.ModuleList([self.model, self.embedding_model])

    def build_saver(self):
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
        return saver
    
    # Tune Interface
    def tune_init(self):
        # Freeze part of the model
        self.model.freeze()
        self.lang_id = self.preprocess_config["lang_id"]
        self.tune = True
        print("Current language: ", self.lang_id)

    def common_step(self, batch, batch_idx, train=True):
        if getattr(self, "fix_spk_args", None) is None:
            self.fix_spk_args = batch[2]
            emb_texts = F.embedding(batch[3], self.embedding_model.get_new_embedding("table-sep", lang_id=self.lang_id, init=False), padding_idx=0)
        output = self.model(batch[2], emb_texts, *(batch[4:]))
        loss = self.loss_func(batch, output)
        return loss, output
    
    def synth_step(self, batch, batch_idx):
        emb_texts = F.embedding(batch[3], self.embedding_model.get_new_embedding("table-sep", lang_id=self.lang_id, init=False), padding_idx=0)
        output = self.model(self.fix_spk_args, emb_texts, *(batch[4:6]), average_spk_emb=True)
        return output

    def text_synth_step(self, batch, batch_idx):  # only used when inference (use TextDataset2)
        emb_texts = F.embedding(batch[2], self.embedding_model.get_new_embedding("table-sep", lang_id=self.lang_id, init=False), padding_idx=0)
        output = self.model(self.fix_spk_args, emb_texts, *(batch[3:5]), average_spk_emb=True)
        return output
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 12, f"data with 12 elements, but get {len(batch)}"

    def training_step(self, batch, batch_idx):
        loss, output = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}":v for k,v in loss2dict(loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': loss[0], 'losses': loss, 'output': output, '_batch': batch}

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 12, f"data with 12 elements, but get {len(batch)}"
    
    def validation_step(self, batch, batch_idx):
        val_loss, predictions = self.common_step(batch, batch_idx, train=False)
        synth_predictions = self.synth_step(batch, batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}":v for k,v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': batch, 'synth': synth_predictions}
    
    def test_step(self, batch, batch_idx):
        outputs = {}
        for test_name, test_fn in getattr(self, "test_list", {}).items(): 
            outputs[test_name] = test_fn(batch, batch_idx)

        return outputs

    def generate_azure_wavs(self, batch, batch_idx):
        synth_predictions = self.text_synth_step(batch, batch_idx)
        return {'_batch': batch, 'synth': synth_predictions}
