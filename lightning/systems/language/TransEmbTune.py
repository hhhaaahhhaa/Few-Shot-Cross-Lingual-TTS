import numpy as np
import torch
import torch.nn as nn
import pickle

from lightning.systems.system import System
from lightning.utils.log import loss2dict
from lightning.utils.tool import LightningMelGAN, generate_reference
from lightning.model.phoneme_embedding import PhonemeEmbedding
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.callbacks.baseline_saver import Saver
from Objects.visualization import CodebookAnalyzer
from lightning.model.reference_extractor import HubertExtractor, XLSR53Extractor, Wav2Vec2Extractor, MelExtractor
from text.define import LANG_ID2SYMBOLS
import Define


class TransEmbTuneSystem(System):
    """ 
    Tune version of TransEmb system.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_codebook_type()
        self.test_list = {
            "azure": self.generate_azure_wavs,
        }
        self.codebook_analyzer = CodebookAnalyzer(self.result_dir)

    def build_model(self):
        self.embedding_model = PhonemeEmbedding(self.model_config, self.algorithm_config)
        self.model = FastSpeech2(self.preprocess_config, self.model_config, self.algorithm_config)
        self.loss_func = FastSpeech2Loss(self.preprocess_config, self.model_config)

        self.vocoder = LightningMelGAN()
        self.vocoder.freeze()

        self.lang_id = self.preprocess_config["lang_id"]
        d_word_vec = self.model_config["transformer"]["encoder_hidden"]
        
        # Tune init
        self.emb_layer = nn.Embedding(len(LANG_ID2SYMBOLS[self.lang_id]), d_word_vec, padding_idx=0)

        # Tune loss
        self.loss_func = FastSpeech2Loss(self.preprocess_config, self.model_config)

        if Define.UPSTREAM == "hubert":
            self.reference_extractor = HubertExtractor()
        elif Define.UPSTREAM == "wav2vec2":
            self.reference_extractor = Wav2Vec2Extractor()
        elif Define.UPSTREAM == "xlsr53":
            self.reference_extractor = XLSR53Extractor()
        elif Define.UPSTREAM == "mel":
            self.reference_extractor = MelExtractor()
        else:
            raise NotImplementedError
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

    def mapping_vis(self):
        gen = False  # regenerate representations
        DOLIST = {
            "LibriTTS": 0,
            "AISHELL-3": 1,
            "GlobalPhone/fr": 2,
            "CSS10/german": 3,
            "JSUT": 6,
            "kss": 8,
        }
        if Define.UPSTREAM != "mel" and gen:
            self.reference_extractor.ssl_extractor.cuda()

        attns_map = {}
        for tt, lang_id in DOLIST.items():
            if gen:
                txt_path = f"preprocessed_data/{tt}/repr.txt"
                info = generate_reference(txt_path, lang_id=lang_id)

                with torch.no_grad():
                    ref_phn_feats = self.reference_extractor.extract(info, norm=False)

                # visualization
                matching = self.embedding_model.get_matching(self.codebook_type, ref_phn_feats=ref_phn_feats, lang_id=lang_id)
                attns = []
                for info in matching[:-1]:
                    attns.append(info["attn"])
                    y_labels = info["y_labels"]
                attns = np.concatenate(attns, axis=1)
                with open(f"preprocessed_data/{tt}/repr-{Define.UPSTREAM}.npy", 'wb') as f:
                    np.save(f, attns)
                with open(f"preprocessed_data/{tt}/repr-{Define.UPSTREAM}-ylabel.pickle", 'wb') as f:
                    pickle.dump(y_labels, f)
            else:
                attns = np.load(f"preprocessed_data/{tt}/repr-{Define.UPSTREAM}.npy")
                with open(f"preprocessed_data/{tt}/repr-{Define.UPSTREAM}-ylabel.pickle", 'rb') as f:
                    y_labels = pickle.load(f)

            attns[:, :128] = attns[:, :128] / np.sqrt((attns[:, :128] ** 2).sum(axis=1, keepdims=True))
            attns[:, 128:256] = attns[:, 128:256] / np.sqrt((attns[:, 128:256] ** 2).sum(axis=1, keepdims=True))
            attns[:, 256:384] = attns[:, 256:384] / np.sqrt((attns[:, 256:384] ** 2).sum(axis=1, keepdims=True))
            attns[:, 384:] = attns[:, 384:] / np.sqrt((attns[:, 384:] ** 2).sum(axis=1, keepdims=True))
            attns_map[lang_id] = {"attn": attns, "y-labels": y_labels}
        
        print("Start mapping!")
        self.codebook_analyzer.visualize_phoneme_mapping("./vis/phoneme-mapping", attns_map)
        
        if Define.UPSTREAM != "mel" and gen:
            self.reference_extractor.ssl_extractor.cpu()

    def tune_init(self):
        # self.mapping_vis()
        # assert 1 == 2
        if Define.UPSTREAM != "mel":
            self.reference_extractor.ssl_extractor.cuda()
        txt_path = f"{self.preprocess_config['path']['preprocessed_path']}/{self.preprocess_config['subsets']['train']}-{Define.EXP_IDX}.txt"
        print(txt_path)
        info = generate_reference(txt_path, lang_id=self.lang_id)
        # ref_phn_feats = torch.from_numpy(ref_phn_feats).float()

        with torch.no_grad():
            ref_phn_feats = self.reference_extractor.extract(info, norm=True)

        with torch.no_grad():
            embedding = self.embedding_model.get_new_embedding(self.codebook_type, ref_phn_feats=ref_phn_feats, lang_id=self.lang_id)
            embedding.requires_grad = True
            self.emb_layer._parameters['weight'] = embedding

            # visualization
            self.visualize_matching(ref_phn_feats)
        if Define.UPSTREAM != "mel":
            self.reference_extractor.ssl_extractor.cpu()

        # tune part
        # for p in self.emb_layer.parameters():
        #     p.requires_grad = False
        # for p in self.model.encoder.parameters():
        #     p.requires_grad = False
        # for p in self.model.variance_adaptor.parameters():
        #     p.requires_grad = False
        # for p in self.model.decoder.parameters():
        #     p.requires_grad = False

    def common_step(self, batch, batch_idx, train=True):
        # print(self.emb_layer._parameters['weight'].requires_grad)
        if getattr(self, "fix_spk_args", None) is None:
            self.fix_spk_args = batch[2]
        emb_texts = self.emb_layer(batch[3])
        output = self.model(batch[2], emb_texts, *(batch[4:]))
        loss = self.loss_func(batch, output)
        return loss, output

    def synth_step(self, batch, batch_idx):  # only used when tune
        emb_texts = self.emb_layer(batch[3])
        output = self.model(self.fix_spk_args, emb_texts, *(batch[4:6]), average_spk_emb=True)
        return output

    def text_synth_step(self, batch, batch_idx):  # only used when predict (use TextDataset2)
        emb_texts = self.emb_layer(batch[2])
        output = self.model(self.fix_spk_args, emb_texts, *(batch[3:5]), average_spk_emb=True)
        return output
        # ids,
        # raw_texts,
        # torch.from_numpy(texts).long(),
        # torch.from_numpy(text_lens),
        # max(text_lens),
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 12, "data with 12 elements"

    def training_step(self, batch, batch_idx):
        loss, output = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}":v for k,v in loss2dict(loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': loss[0], 'losses': loss, 'output': output, '_batch': batch}

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 12, "data with 12 elements"

    def validation_step(self, batch, batch_idx):
        val_loss, predictions = self.common_step(batch, batch_idx, train=False)
        synth_predictions = self.synth_step(batch, batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}":v for k,v in loss2dict(val_loss).items()}
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
        synth_predictions = self.text_synth_step(batch, batch_idx)
        return {'_batch': batch, 'synth': synth_predictions}