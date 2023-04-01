import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

from dlhlp_lib.utils.tool import get_mask_from_lengths
from dlhlp_lib.utils.numeric import torch_exist_nan

import Define
from lightning.build import build_all_speakers
from lightning.systems import System
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.callbacks.language.baseline_saver import Saver
from ...plugin.fscl import IFSCLPlugIn, OrigFSCLPlugIn, TransformerFSCLPlugIn
from ...plugin.tm import ITextMatchingPlugIn, TMPlugIn
from lightning.utils.tool import flat_merge_dict
from .utils import generate_reference_info
from text.define import LANG_ID2NAME, LANG_NAME2ID


def _dual_tune_fastspeech2_class_factory(FSCLPlugInClass: Type[IFSCLPlugIn], TMPlugInClass: Type[ITextMatchingPlugIn]):
    class DualTuneSystem(System):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def build_configs(self):
            self.spk_config = {
                "emb_type": self.model_config["speaker_emb"],
                "speakers": build_all_speakers(self.data_configs)
            }
            self.bs = self.train_config["optimizer"]["batch_size"]
        
        def setup(self, stage=None):
            self.tm.codebook_bind(self.fscl.codebook_attention)
            
        def build_model(self):
            self.threshold = self.algorithm_config.get("threshold", 0.999)
            self.model = FastSpeech2(self.model_config, spk_config=self.spk_config)
            self.loss_func = FastSpeech2Loss(self.model_config)
            self.fscl = FSCLPlugInClass(self.model_config)
            # this requires fscl class to equip a codebook module
            self.tm = TMPlugInClass(self.data_configs, self.model_config["text_matching"])

        def build_hooks(self):
            self.hooks = {
                "unused_ids": [],
                "seg_repr": None,
                "phoneme_query": None,
            }
            def _hook(module, input, output):
                unused_ids = []
                class_labels, table, _ = input
                for c in class_labels:
                    if len(table[c]) == 0:
                        unused_ids.append(c)
                self.hooks["unused_ids"] = unused_ids
                self.hooks["phoneme_query"] = output.clone().detach()
            self.fscl.phoneme_query_extractor.reduction.register_forward_hook(_hook)
            
            def _hook2(module, input, output):
                self.hooks["seg_repr"] = input[0][:, :, Define.LAYER_IDX].clone().detach()
            self.fscl.codebook_attention.register_forward_hook(_hook2)

        def tune_init(self, data_configs):
            print("Generate reference...")
            ref_infos = generate_reference_info(data_configs[0])
            self.target_lang_id = ref_infos[0]["lang_id"]
            print(f"Target Language: {self.target_lang_id}.")

            print("Embedding initialization...")
            self.build_hooks()
            self.cuda()
            with torch.no_grad():
                table, attn = self.fscl.build_embedding_table(ref_infos, return_attn=True)
                self.attn = attn
                self.tm.embedding_model.tables[f"table-{ref_infos[0]['symbol_id']}"].copy_(table)
            for p in self.tm.embedding_model.parameters():
                p.requires_grad = True
            
            self.reference = {
                "unused_ids": self.hooks["unused_ids"],
                "centers": self.hooks["phoneme_query"][:, 24],
            }

            self.cpu()

        def build_optimized_model(self):
            return nn.ModuleList([self.model, self.tm.build_optimized_model()])

        def build_saver(self):
            self.saver = Saver(self.data_configs, self.model_config, self.log_dir, self.result_dir)
            return self.saver

        def common_u2s_step(self, seg_repr, batch, batch_idx, train=True):
            labels, _ = batch
            output = self.model(labels[2], seg_repr, *(labels[4:]))
            loss = self.loss_func(labels[:-1], output)
            loss_dict = {
                "Total Loss"       : loss[0],
                "Mel Loss"         : loss[1],
                "Mel-Postnet Loss" : loss[2],
                "Pitch Loss"       : loss[3],
                "Energy Loss"      : loss[4],
                "Duration Loss"    : loss[5],
            }
            return loss_dict, output
        
        def common_tm_step(self, seg_repr, batch, batch_idx, train=True, pl=None):
            labels, _ = batch
            # seg_repr_clustered = self.tm.cluster(seg_repr, lang_args=labels[-1])
            # print(seg_repr.shape, seg_repr_clustered.shape)
            # c_loss = self.tm.cluster_loss_func(seg_repr, seg_repr_clustered, labels[4])
            if pl is not None:
                pseudo_idxs, mask = pl
                output = self.tm(pseudo_idxs, labels[4], mask)
                assert not torch_exist_nan(output)
            else:
                output = self.tm(labels[3], labels[4])
            # print(seg_repr.shape, output.shape)
            m_loss = self.tm.match_loss_func(seg_repr, output, labels[4])

            loss_dict = {
                "Total Loss": m_loss,  # Currently close c_loss (unstable)
                # "Cluster Loss": c_loss,
                "Match Loss": m_loss,
            }
            return loss_dict, output
        
        def online_pseudo_label(self):
            seg_repr = self.hooks["seg_repr"]
            centers = self.reference["centers"].to(self.device)  # K, dim
            assert not torch_exist_nan(seg_repr)
            assert not torch_exist_nan(centers)
            x = (seg_repr.unsqueeze(2) - centers) ** 2  # B, T, K, dim
            loss = torch.sum(x, dim=-1)  # B, T, K
            assert not torch_exist_nan(loss)
            
            if "unused_ids" in self.reference:
                loss[:, :, self.reference["unused_ids"]] = float("inf")
            loss = F.softmax(-loss - torch.max(-loss), dim=-1)  # normalize to avoid overflow
            assert not torch_exist_nan(loss)
            confidences, idxs = loss.max(dim=-1)

            return idxs, confidences

        def common_step(self, batch, batch_idx, train=True):
            _, ref_info = batch
            with torch.no_grad():
                seg_repr, attn_fscl = self.fscl.build_segmental_representation([ref_info])
                seg_repr = seg_repr.detach()
                assert not torch_exist_nan(seg_repr)
            
            if not train:
                tm_loss_dict, tm_output = self.common_tm_step(seg_repr, batch, batch_idx, train)  # be careful that seg_repr collapse
                u2s_loss_dict, u2s_output = self.common_u2s_step(tm_output, batch, batch_idx, train)
            else:
                with torch.no_grad():
                    pseudo_idxs, confidences = self.online_pseudo_label()
                assert not torch_exist_nan(pseudo_idxs)
                assert not torch_exist_nan(confidences)
                mask = (confidences < self.threshold)
                assert not torch_exist_nan(mask)
                tm_loss_dict, tm_output = self.common_tm_step(seg_repr, batch, batch_idx, train, pl=(pseudo_idxs, mask))  # be careful that seg_repr collapse
                # tm_loss_dict, tm_output = self.common_tm_step(seg_repr, batch, batch_idx, train)
                mixed_repr = torch.where(mask.unsqueeze(-1), seg_repr, tm_output)
                u2s_loss_dict, u2s_output = self.common_u2s_step(mixed_repr, batch, batch_idx, train)

                # Logging
                length_mask = get_mask_from_lengths(batch[0][4]).to(self.device)
                denom = torch.sum(~length_mask)
                numer = torch.sum((~length_mask) & (~mask))
                self.log("PL threshold", self.threshold, sync_dist=True)
                self.log("PL ratio", numer / max(denom, 1), sync_dist=True)
                correct = (batch[0][3] == pseudo_idxs)
                numer2 = torch.sum((~length_mask) & (~mask) & correct)
                self.log("PL rccuracy", numer2 / max(numer, 1), sync_dist=True)
                # print(denom, numer, numer2)
                # input()

            loss_dict = flat_merge_dict({
                "U2S": u2s_loss_dict,
                "TM": tm_loss_dict
            })

            loss_dict["Total Loss"] = 0.0 * tm_loss_dict["Total Loss"] + u2s_loss_dict["Total Loss"]
            return loss_dict, u2s_output

        def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
            assert len(batch[0]) == 13, f"data with 13 elements, but get {len(batch)}"
        
        def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
            assert len(batch[0]) == 13, f"data with 13 elements, but get {len(batch)}"
        
        def training_step(self, batch, batch_idx):
            train_loss_dict, output = self.common_step(batch, batch_idx, train=True)

            # Log metrics to CometLogger
            loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
            self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
            return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': output, '_batch': batch[0]}

        def validation_step(self, batch, batch_idx):
            val_loss_dict, predictions = self.common_step(batch, batch_idx, train=False)

            if batch_idx == 0:
                self.saver.log_2D_tensor(
                    self.logger, torch.sigmoid(self.tm.alpha[:10]).data, self.global_step + 1, "alpha",
                    x_labels=[str(i) for i in range(self.tm.alpha.shape[1])],
                    y_labels=[LANG_ID2NAME[i] for i in range(10)], 
                    stage="val"
                )
           
            # Log metrics to CometLogger
            loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
            self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
            return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': batch[0]}

        def on_save_checkpoint(self, checkpoint):
            checkpoint = self.fscl.on_save_checkpoint(checkpoint, prefix="fscl")
            return checkpoint
        
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
                    if k.startswith("codebook_attention"):  # Checkpoints before PlugIn classes are created
                        k_new = "fscl." + k
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
                    if k.startswith("fscl.upstream"):
                        pass
                    else:
                        print("Reinitialized: ", k)
                    state_dict[k] = model_state_dict[k]

            if is_changed:
                checkpoint.pop("optimizer_states", None)

        def inference(self, spk_ref_mel_slice: np.ndarray, text: np.ndarray, symbol_id: str, lang_id: str=None):
            """
            Return FastSpeech2 results:
                (
                    output,
                    postnet_output,
                    p_predictions,
                    e_predictions,
                    log_d_predictions,
                    d_rounded,
                    src_masks,
                    mel_masks,
                    src_lens,
                    mel_lens,
                )
            """
            spk_args = (torch.from_numpy(spk_ref_mel_slice).to(self.device), [slice(0, spk_ref_mel_slice.shape[0])])
            if lang_id is not None:
                lang_args = torch.LongTensor([LANG_NAME2ID[lang_id]]).to(self.device)
            else:
                lang_args = None
            texts = torch.from_numpy(text).long().unsqueeze(0).to(self.device)
            src_lens = torch.LongTensor([len(text)]).to(self.device)
            max_src_len = max(src_lens)
            
            with torch.no_grad():
                emb_texts = self.tm(texts, lengths=src_lens, symbol_id=symbol_id)
                output = self.model(spk_args, emb_texts, src_lens, max_src_len, lang_args=lang_args, average_spk_emb=True)

            return output

    return DualTuneSystem


def dual_tune_fastspeech2_class_factory(name):
    if name == "orig":
        return _dual_tune_fastspeech2_class_factory(OrigFSCLPlugIn, TMPlugIn)
    # elif name == "transformer":
    #     return _dual_fastspeech2_class_factory(TransformerFSCLPlugIn, TMPlugIn)
    else:
        raise NotImplementedError
