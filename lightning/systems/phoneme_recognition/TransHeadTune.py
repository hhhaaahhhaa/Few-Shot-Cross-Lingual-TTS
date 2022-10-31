import torch
import torch.nn as nn
import torch.nn.functional as F

from dlhlp_lib.s3prl import S3PRLExtractor

from lightning.systems.system import System
from lightning.utils.log import pr_loss2dict as loss2dict
from lightning.callbacks.phoneme_recognition.baseline_saver import Saver
import Define
from text.define import LANG_ID2SYMBOLS
# from .modules import BiLSTMDownstream, SoftAttCodebook, PRFramewiseLoss
from lightning.model.reduction import PhonemeQueryExtractor
from lightning.utils.tool import generate_reference, ssl_match_length


class TransHeadTuneSystem(System):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        self.upstream = S3PRLExtractor("hubert_large_ll60k")
        self.upstream.freeze()
        self.downstream = BiLSTMDownstream(n_in_layers=Define.UPSTREAM_LAYER, upstream_dim=Define.UPSTREAM_DIM, specific_layer=Define.LAYER_IDX)

        # TransHead modules
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average")
        self.codebook = SoftAttCodebook(
            self.model_config, self.algorithm_config, 
            n_in_layers=Define.UPSTREAM_LAYER, upstream_dim=Define.UPSTREAM_DIM, specific_layer=Define.LAYER_IDX
        )
        self.trans_head_bias = nn.Parameter(torch.zeros(1))

        # Tune init
        self.lang_id = self.preprocess_config["lang_id"]
        d_word_vec = self.model_config["transformer"]["encoder_hidden"]
        self.head = nn.Linear(d_word_vec, len(LANG_ID2SYMBOLS[self.lang_id]))

        self.loss_func = PRFramewiseLoss()

        if Define.DEBUG:
            print(self)

    def build_optimized_model(self):
        return nn.ModuleList([self.head])

    def build_saver(self):
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
        return saver
    
    def tune_init(self, data_config):
        from dlhlp_lib.utils import batchify

        print("Generate reference...")
        repr_info = generate_reference(
            path=data_config["subsets"]["train"], 
            data_parser=Define.DATAPARSERS[data_config["name"]],
            lang_id=data_config["lang_id"]
        )

        self.cuda()
        with torch.no_grad():
            ssl_reprs = []
            for wavs in batchify(repr_info["raw-feat"], batch_size=16):
                ssl_repr, _ = self.upstream.extract(wavs)
                ssl_reprs.extend([x for x in ssl_repr])
            phoneme_queries = self.phoneme_query_extractor(ssl_reprs, repr_info["avg-frames"], repr_info)  # 1, n_symbols, n_layers, dim
            print(phoneme_queries.shape)
            head_weights = self.codebook(phoneme_queries).squeeze(0)  # n_symbols, dim
            self.head.weight.copy_(head_weights)
        for p in self.head.parameters():
            p.requires_grad = True
        self.cpu()
        print("Generate reference done.")

        # tune partial model
        # for p in self.head.parameters():
        #     p.requires_grad = False
        # for p in self.downstream.parameters():
        #     p.requires_grad = False

    def common_step(self, batch, batch_idx, train=True):
        labels, repr_info = batch

        self.upstream.eval()
        ssl_repr, _ = self.upstream.extract(repr_info["wav"])  # B, L, n_layers, dim
        ssl_repr = ssl_match_length(ssl_repr, labels[5])
        ssl_repr = ssl_repr.detach()

        if Define.DEBUG:
            print(ssl_repr.shape)
            print(labels[3].shape)

        x = self.downstream(ssl_repr, labels[4].cpu())
       
        output = self.head(x)
        loss = self.loss_func(labels, output)

        return loss, output

    def training_step(self, batch, batch_idx):
        labels, repr_info = batch
        train_loss, predictions = self.common_step(batch, batch_idx, train=True)

        mask = (labels[3] != 0)
        acc = ((labels[3] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
        self.log_dict({"Train/Acc": acc.item()}, sync_dist=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss, 'losses': train_loss, 'output': predictions, '_batch': labels, 'lang_id': repr_info["lang_id"]}

    def validation_step(self, batch, batch_idx):
        labels, repr_info = batch
        val_loss, predictions = self.common_step(batch, batch_idx)

        mask = (labels[3] != 0)
        acc = ((labels[3] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
        self.log_dict({"Val/Acc": acc.item()}, sync_dist=True)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': labels, 'lang_id': repr_info["lang_id"]}
