import torch
import torch.nn as nn
import torch.nn.functional as F

from dlhlp_lib.s3prl import S3PRLExtractor

from lightning.systems.adaptor import AdaptorSystem
from lightning.utils.log import pr_loss2dict as loss2dict
from lightning.callbacks.phoneme_recognition.ssl_baseline_saver import Saver
import Define
from text.define import LANG_ID2SYMBOLS
from .modules import BiLSTMDownstream, SoftAttCodebook
from lightning.model.reduction import PhonemeQueryExtractor


class TransHeadSystem(AdaptorSystem):

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

        self.loss_func = SSLBaselineLoss()

        if Define.DEBUG:
            print(self)

    def _on_meta_batch_start(self, batch):
        """ Check meta-batch data """
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 2 or len(batch[0]) == 4, "sup + qry (+ ref_phn_feats + lang_id)"
        assert len(batch[0][1]) == 1, "n_batch == 1"
        assert len(batch[0][1][0]) == 7, "data with 7 elements"
    
    def build_optimized_model(self):
        return nn.ModuleList([self.downstream, self.codebook])

    def build_saver(self):
        saver = Saver(self.preprocess_config, self.log_dir, self.result_dir)
        return saver
    
    def build_head_weights(self, batch):
        _, _, repr_info, _ = batch[0]
        with torch.no_grad():
            ssl_repr, _ = self.upstream.extract(repr_info["raw-feat"])  # B, L, n_layers, dim
            phoneme_queries = self.phoneme_query_extractor(ssl_repr, repr_info["avg-frames"], repr_info)  # 1, n_symbols, n_layers, dim
        
        head_weights = self.codebook(phoneme_queries).squeeze(0)  # n_symbols, dim
        if Define.DEBUG:
            print("Head shape and gradient required: ", head_weights.shape)
            print(head_weights.requires_grad)
        
        return head_weights

    def common_step(self, batch, batch_idx, train=True):
        if Define.DEBUG:
            print("Generate head weights... ")
        head_weights = self.build_head_weights(batch)

        _, qry_batch, repr_info, _ = batch[0]
        labels = list(qry_batch[0])
        ssl_repr, _ = self.upstream.extract(repr_info["wav"])  # B, L, n_layers, dim
        if ssl_repr.shape[1] < labels[5]:
            ssl_repr = ssl_repr.detach()
            labels[3] = labels[3][:, :ssl_repr.shape[1]]
            repr_info["len"] = ssl_repr.shape[1]
        else:
            ssl_repr = ssl_repr[:, :labels[5]].detach()  # Reduce to the same size as labels, dirty
            repr_info["len"] = labels[5]
        
        # if Define.DEBUG:
        #     print(ssl_repr.shape)
        #     print(labels[3].shape)
        
        x = self.downstream(ssl_repr)

        # TransHead          
        output = F.linear(x, head_weights, bias=self.trans_head_bias)
        loss = self.loss_func(labels, output)

        return loss, output

    def training_step(self, batch, batch_idx):
        _, qry_batch, repr_info, _ = batch[0]
        labels = list(qry_batch[0])
        train_loss, predictions = self.common_step(batch, batch_idx, train=True)
        labels[3] = labels[3][:, :repr_info["len"]]

        mask = (labels[3] != 0)
        acc = ((labels[3] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
        self.log_dict({"Train/Acc": acc.item()}, sync_dist=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v for k, v in loss2dict(train_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss, 'losses': train_loss, 'output': predictions, '_batch': labels, 'lang_id': repr_info["lang_id"]}

    def validation_step(self, batch, batch_idx):
        _, qry_batch, repr_info, _ = batch[0]
        labels = list(qry_batch[0])
        val_loss, predictions = self.common_step(batch, batch_idx)
        labels[3] = labels[3][:, :repr_info["len"]]

        mask = (labels[3] != 0)
        acc = ((labels[3] == predictions.argmax(dim=2)) * mask).sum() / mask.sum()
        self.log_dict({"Train/Acc": acc.item()}, sync_dist=True)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v for k, v in loss2dict(val_loss).items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'losses': val_loss, 'output': predictions, '_batch': labels, 'lang_id': repr_info["lang_id"]}


class SSLBaselineLoss(nn.Module):
    """ Cross Entropy Loss """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, labels, preds):
        preds = preds.transpose(1, 2)  # B, N, L
        target = labels[3]  # B, L
        return self.loss(preds, target)
