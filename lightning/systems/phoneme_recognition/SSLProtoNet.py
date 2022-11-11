from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from dlhlp_lib.s3prl import S3PRLExtractor

import Define
from lightning.build import build_ctc_decoders, build_id2symbols
from lightning.systems.adaptor import AdaptorSystem
from lightning.callbacks.phoneme_recognition.baseline_saver import Saver
from lightning.model.reduction import PhonemeQueryExtractor
from lightning.utils.tool import ssl_match_length
from .loss import PRFramewiseLoss
from .downstreams import *
from .heads import *
from .SSLBaseline import training_step_template, validation_step_template


class SSLProtoNetSystem(AdaptorSystem):

    def __init__(self, *args, **kwargs):
        self.support_head = False
        self.use_ctc = False  # TODO: not finished yet (random crash bug)
        super().__init__(*args, **kwargs)

    def build_model(self):
        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.downstream = Downstream1(
            self.model_config,
            n_in_layers=Define.UPSTREAM_LAYER,
            upstream_dim=Define.UPSTREAM_DIM,
            specific_layer=Define.LAYER_IDX
        )
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=False)
        if self.support_head:
            self.head = MultilingualPRHead(
                build_id2symbols(self.data_configs), d_in=self.model_config["transformer"]["d_model"])
        
        if self.use_ctc:
            self.loss_func = nn.CTCLoss(blank=0, zero_infinity=True)
            build_ctc_decoders(self.data_configs)
        else:
            self.loss_func = PRFramewiseLoss()

    def _on_meta_batch_start(self, batch):
        """ Check meta-batch data """
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 2 or len(batch[0]) == 3, "sup + qry (+ ref_phn_feats)"
        assert len(batch[0][1]) == 1, "n_batch == 1"
        assert len(batch[0][1][0]) == 10, "data with 10 elements"
    
    def build_optimized_model(self):
        if self.support_head:
            return nn.ModuleList([self.downstream, self.head])
        return nn.ModuleList([self.downstream])

    def build_saver(self):
        saver = Saver(self.data_configs, self.log_dir, self.result_dir)
        return saver
    
    def build_prototype(self, batch):
        sup_batch, qry_batch, repr_info = batch[0]

        self.upstream.eval()
        with torch.no_grad():
            ssl_repr, _ = self.upstream.extract(repr_info["sup_wav"])  # B, L, n_layers, dim
            ssl_repr = ssl_match_length(ssl_repr, repr_info["sup_repr_max_len"])
            ssl_repr = ssl_repr.detach()

        x = self.downstream(ssl_repr, repr_info["sup_lens"].cpu())
        prototypes = self.phoneme_query_extractor(x, repr_info["sup_avg_frames"], 
                            repr_info["n_symbols"], repr_info["sup_phonemes"])  # 1, n_symbols, n_layers, dim
        prototypes = prototypes.squeeze(0)  # n_symbols, dim
        
        # print("Prototype shape and gradient required: ", prototypes.shape)
        # print(prototypes.requires_grad)
        
        return prototypes

    def common_step(self, batch, batch_idx, train=True):
        prototypes = self.build_prototype(batch)

        sup_batch, qry_batch, repr_info = batch[0]
        labels = qry_batch[0]

        self.upstream.eval()
        with torch.no_grad():
            ssl_repr, _ = self.upstream.extract(repr_info["qry_wav"])  # B, L, n_layers, dim
            ssl_repr = ssl_match_length(ssl_repr, labels[5])
            ssl_repr = ssl_repr.detach()

        x = self.downstream(ssl_repr, labels[4].cpu())

        # Prototype loss
        output = -torch.linalg.norm(prototypes.unsqueeze(0).unsqueeze(0) - x.unsqueeze(2), dim=3)  # B, L, n_c
        if self.use_ctc:
            loss = self.loss_func(labels[3], output)
        #     log_probs = torch.log_softmax(output, dim=2)
        #     with torch.backends.cudnn.flags(deterministic=True):  # for reproducibility
        #         loss = self.loss_func(log_probs.transpose(0, 1), labels[6], labels[4].cpu(), labels[7].cpu())
        else:
            loss = self.loss_func(labels[3], output)

        if self.support_head:
            output_support = self.head(x, lang_id=repr_info["lang_id"])
            loss_support = self.loss_func(labels, output_support)

        if self.support_head:
            loss_dict = {
                "Total Loss": loss + loss_support,
                "Proto Loss": loss,
                "Linear Loss": loss_support
            }
        else:
            loss_dict = {
                "Total Loss": loss
            }
            
        return loss_dict, output

    def training_step(self, batch, batch_idx):
        sup_batch, qry_batch, repr_info = batch[0]
        labels = qry_batch[0]
        return training_step_template(self, batch, batch_idx, labels, repr_info)

    def validation_step(self, batch, batch_idx):
        sup_batch, qry_batch, repr_info = batch[0]
        labels = qry_batch[0]
        return validation_step_template(self, batch, batch_idx, labels, repr_info)

    def on_save_checkpoint(self, checkpoint):
        """ (Hacking!) Remove pretrained weights in checkpoint to save disk space. """
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k in state_dict:
            if k.split('.')[0] == "upstream":
                continue
            new_state_dict[k] = state_dict[k]
        checkpoint["state_dict"] = new_state_dict

        return checkpoint
