# Deprecated
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from dlhlp_lib.s3prl import S3PRLExtractor

import Define
from lightning.build import build_all_speakers
from lightning.systems.adaptor import AdaptorSystem
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.model.reduction import PhonemeQueryExtractor
from lightning.model.old_extractor import S3PRLExtractor as OldExtractor
from lightning.callbacks.language.baseline_saver import Saver
from .embeddings import *
from transformer import Constants
from .embeddings import SoftMultiAttCodebook2


class TransEmbOrigSystem(AdaptorSystem): 

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
        self.model = FastSpeech2(self.model_config, spk_config=self.spk_config)
        self.loss_func = FastSpeech2Loss(self.model_config)
        
        self.upstream = S3PRLExtractor(Define.UPSTREAM)
        self.upstream.freeze()
        self.phoneme_query_extractor = PhonemeQueryExtractor(mode="average", two_stage=True)
        self.codebook_attention = SoftMultiAttCodebook2(
            codebook_size=self.model_config["codebook_size"],
            embed_dim=encoder_dim,
            num_heads=self.model_config["downstream"]["transformer"]["nhead"],
        )
        print(self.codebook_attention)
    
    def build_optimized_model(self):
        return nn.ModuleList([self.codebook_attention, self.model])
        
    def build_saver(self):
        self.saver = Saver(self.data_configs, self.model_config, self.log_dir, self.result_dir)
        return self.saver

    def build_embedding_table(self, batch, return_attn=False):  
        sup_info = batch[0][2]

        # TODO: Mel version
        self.upstream.eval()
        with torch.no_grad():
            ssl_repr, _ = self.upstream.extract(sup_info["raw_feat"])  # B, L, n_layers, dim
            ssl_repr = ssl_repr.detach()

        # print("Check s3prl")
        # for r in ssl_repr:
        #     print(r[:, 24, :].sum())
        # input()

        table_pre = self.phoneme_query_extractor(ssl_repr, sup_info["avg_frames"], 
                            sup_info["n_symbols"], sup_info["phonemes"])  # 1, n_symbols, 25, dim
        table, attn = self.codebook_attention(table_pre, need_weights=return_attn)
        table = table.squeeze(0)  # n_symbols, dim
        table[Constants.PAD].fill_(0)
        
        if (table_pre != table_pre).any():
            print("NaN table")
            assert 1 == 2
        
        # print("Table shape and gradient required: ", table.shape)
        # print(table.requires_grad)
        
        if return_attn:
            return table, attn
        else:
            return table

    def _on_meta_batch_start(self, batch):
        """ Check meta-batch data """
        assert len(batch) == 1, "meta_batch_per_gpu"
        assert len(batch[0]) == 3, "sup + qry + sup_info"
        assert len(batch[0][0]) == 1, "n_batch == 1"
        assert len(batch[0][0][0]) == 13, "data with 13 elements"
    
    def common_step(self, batch, batch_idx, train=True):
        if not train:
            emb_table, attn = self.build_embedding_table(batch, return_attn=True)
        else:
            emb_table = self.build_embedding_table(batch)
        qry_batch = batch[0][1][0]
        # print("Check input")
        # print(qry_batch[0])
        # print(qry_batch[4], qry_batch[5])
        # print(qry_batch[6].sum(), qry_batch[7])
        # print(qry_batch[8], qry_batch[9].sum(), qry_batch[10].sum())
        # input()
        
        emb_texts = F.embedding(qry_batch[3], emb_table, padding_idx=0)

        # print("Check emb_text")
        # print(emb_texts.sum(-1))
        # input()

        output = self.model(qry_batch[2], emb_texts, *(qry_batch[4:]), average_spk_emb=True)  # This must be true...
        loss = self.loss_func(qry_batch[:-1], output)
        loss_dict = {
            "Total Loss"       : loss[0],
            "Mel Loss"         : loss[1],
            "Mel-Postnet Loss" : loss[2],
            "Pitch Loss"       : loss[3],
            "Energy Loss"      : loss[4],
            "Duration Loss"    : loss[5],
        }
        # print(loss_dict)
        if not train:
            return loss_dict, output, attn
        else:
            return loss_dict, output

    def training_step(self, batch, batch_idx):
        train_loss_dict, output = self.common_step(batch, batch_idx, train=True)
        qry_batch = batch[0][1][0]

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': output, '_batch': qry_batch}

    def validation_step(self, batch, batch_idx):
        val_loss_dict, predictions, attn = self.common_step(batch, batch_idx, train=False)
        # print("val end")
        # input()
        qry_batch = batch[0][1][0]

        # visualization
        if batch_idx == 0:
            layer_weights = F.softmax(self.codebook_attention.weight_raw.squeeze(0).squeeze(-1), dim=0)
            self.saver.log_layer_weights(self.logger, layer_weights.data, self.global_step + 1, "val")
        if batch_idx % 4 == 0:
            lang_id = qry_batch[-1][0].item()  # all batch belongs to the same language
            self.saver.log_codebook_attention(self.logger, attn, lang_id, batch_idx, self.global_step + 1, "val")

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': qry_batch}

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

    # Enable loading checkpoints trained from the old repo
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.test_global_step = checkpoint["global_step"]
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        state_dict_pop_keys = []
        state_dict_remap_keys = []
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    if self.local_rank == 0:
                        print(f"Skip loading parameter: {k}, "
                                    f"required shape: {model_state_dict[k].shape}, "
                                    f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                if "embedding_model.hub.embeddings.soft-m." in k:
                    k_new = k.replace("embedding_model.hub.embeddings.soft-m.", "")
                    k_new = f"codebook_attention.{k_new}"
                    state_dict_remap_keys.append((k_new, k))
                
                if self.local_rank == 0:
                    print(f"Dropping parameter {k}")
                state_dict_pop_keys.append(k)
                is_changed = True

        if len(state_dict_remap_keys) > 0:
            for (k_new, k) in state_dict_remap_keys:
                print(f"Remap parameters from old to new ({k} => {k_new}).")
                state_dict[k_new] = state_dict[k]

        # modify state_dict format to model_state_dict format
        for k in state_dict_pop_keys:
            state_dict.pop(k)
        for k in model_state_dict:
            if k not in state_dict:
                if k.split('.')[0] in ["upstream", "embedding_generator"]:
                    pass
                else:
                    print("Reinitialized: ", k)
                state_dict[k] = model_state_dict[k]

        if is_changed:
            checkpoint.pop("optimizer_states", None)



from learn2learn.algorithms import MAML
from dlhlp_lib.utils.tool import get_mask_from_lengths


class TransEmbOrig2System(TransEmbOrigSystem):
    """
    Using MAML-like style which is closer to original style, should be identical to TransEmbOrigSystem.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        encoder_dim = self.model_config["transformer"]["encoder_hidden"]
        self.model = FastSpeech2(self.model_config, spk_config=self.spk_config)
        self.loss_func = FastSpeech2Loss(self.model_config)
        
        self.old_phoneme_query_extractor = OldExtractor(Define.UPSTREAM)
        self.old_phoneme_query_extractor.freeze()
        
        self.codebook_attention = SoftMultiAttCodebook2(
            codebook_size=self.model_config["codebook_size"],
            embed_dim=encoder_dim,
            num_heads=self.model_config["downstream"]["transformer"]["nhead"],
        )
        print(self.codebook_attention)
    
    def build_optimized_model(self):
        return nn.ModuleList([self.codebook_attention, self.model])
    
    def build_embedding_table(self, batch, return_attn=False):  
        _, _, sup_info = batch[0]
        
        self.old_phoneme_query_extractor.eval()
        table_pre = self.old_phoneme_query_extractor.extract(sup_info).to(self.device)  # 1, n_symbols, 25, dim
        table, attn = self.codebook_attention(table_pre, need_weights=return_attn)
        table = table.squeeze(0)  # n_symbols, dim
        table[Constants.PAD].fill_(0)

        if (table_pre != table_pre).any():
            print("NaN table")
            assert 1 == 2
        
        # print("Table shape and gradient required: ", table.shape)
        # print(table.requires_grad)
        
        if return_attn:
            return table, attn
        else:
            return table

    def common_step_old(self, batch, batch_idx, train=True):
        if not train:
            emb_table, attn = self.build_embedding_table(batch, return_attn=True)
        else:
            emb_table = self.build_embedding_table(batch)
        
        emb_layer = nn.Embedding(*emb_table.shape, padding_idx=0).to(self.device)
        emb_layer._parameters['weight'] = emb_table
        adapt_dict = nn.ModuleDict({
            k: getattr(self.model, k) for k in self.algorithm_config["adapt"]["modules"]
        })
        adapt_dict["embedding"] = emb_layer
        learner = MAML(adapt_dict, lr=self.adaptation_lr)

        qry_batch = batch[0][1][0]
        output = self.forward_learner(learner, *qry_batch[2:-1], average_spk_emb=True)

        loss = self.loss_func(qry_batch[:-1], output)
        loss_dict = {
            "Total Loss"       : loss[0],
            "Mel Loss"         : loss[1],
            "Mel-Postnet Loss" : loss[2],
            "Pitch Loss"       : loss[3],
            "Energy Loss"      : loss[4],
            "Duration Loss"    : loss[5],
        }
        if not train:
            return loss_dict, output, attn
        else:
            return loss_dict, output

    def forward_learner(
        self, learner, speaker_args, texts, src_lens, max_src_len,
        mels=None, mel_lens=None, max_mel_len=None,
        p_targets=None, e_targets=None, d_targets=None,
        p_control=1.0, e_control=1.0, d_control=1.0,
        average_spk_emb=False,
    ):
        _get_module = lambda name: getattr(learner.module, name, getattr(self.model, name, None))
        embedding        = _get_module('embedding')
        encoder          = _get_module('encoder')
        variance_adaptor = _get_module('variance_adaptor')
        decoder          = _get_module('decoder')
        mel_linear       = _get_module('mel_linear')
        postnet          = _get_module('postnet')
        speaker_emb      = _get_module('speaker_emb')

        if p_targets is not None:
            p_targets = p_targets.contiguous()
            e_targets = e_targets.contiguous()

        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        # print("text ids", texts)
        emb_texts = embedding(texts)
        if (emb_texts != emb_texts).any():
            print("NaN table")
        output = encoder(emb_texts, src_masks)
        if (output != output).any():
            print("encoder nan")

        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None
        )

        if speaker_emb is not None:
            spk_emb = speaker_emb(speaker_args)
            if average_spk_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True).expand(output.shape[0], -1)
            output += spk_emb.unsqueeze(1).expand(-1, max_src_len, -1)

        (
            output, p_predictions, e_predictions, log_d_predictions, d_rounded,
            mel_lens, mel_masks,
        ) = variance_adaptor(
            output, src_masks, mel_masks, max_mel_len,
            p_targets, e_targets, d_targets, p_control, e_control, d_control,
        )
        if (output != output).any():
            print("variance_adaptor nan")

        if speaker_emb is not None:
            spk_emb = speaker_emb(speaker_args)
            if average_spk_emb:
                spk_emb = spk_emb.mean(dim=0, keepdim=True).expand(output.shape[0], -1)
            if max_mel_len is None:  # inference stage
                max_mel_len = max(mel_lens)
            output += spk_emb.unsqueeze(1).expand(-1, max_mel_len, -1)

        output, mel_masks = decoder(output, mel_masks)
        if (output != output).any():
            print("decoder nan")
        output = mel_linear(output)
        if (output != output).any():
            print("mel linear nan")

        tmp = postnet(output)
        if (tmp != tmp).any():
            print("postnet nan")
        postnet_output = tmp + output

        return (
            output, postnet_output,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            src_masks, mel_masks, src_lens, mel_lens,
        )
    
    def common_step(self, batch, batch_idx, train=True):
        return self.common_step_old(batch, batch_idx, train)
