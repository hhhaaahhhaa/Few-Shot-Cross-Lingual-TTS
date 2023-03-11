import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.utils.tool import get_mask_from_lengths

from lightning.model import FastSpeech2ADALoss, ADAEncoder
from lightning.utils.tool import flat_merge_dict


class FastSpeech2ADAPlugIn(pl.LightningModule):
    def __init__(self, d_in, model_config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_in = d_in
        self.model_config = model_config

    def build_model(self):
        self.model = ADAEncoder(self.d_in, self.model_config)
        self.recon_loss_func = FastSpeech2ADALoss()
        self.match_loss_func = nn.MSELoss()

    def build_optimized_model(self):
        return nn.ModuleList([self.model])
    
    def forward(self, x, lengths, embed=True):
        return self.model(x, lengths, embed=embed)


def ada_class_factory(BaseClass, ada_stage: str):  # Class factory
    if ada_stage not in ["matching", "unsup_tuning"]:
        raise NotImplementedError(f"Unknown adaspeech2 stage ({ada_stage}).")
    class TransEmbADASystem(BaseClass): 
        def __init__(self, *args, **kwargs):
            super(TransEmbADASystem, self).__init__(*args, **kwargs)
    
        def build_model(self):
            super().build_model()
            self.ada_stage = ada_stage
            self._build_hook()
            self.ada = FastSpeech2ADAPlugIn(AUDIO_CONFIG["mel"]["n_mel_channels"], self.model_config)
            self.ada.build_model()
        
        def _build_hook(self):
            self.hooked_output = None
            self.model.decoder.register_forward_hook(self._input_hook())

        def _input_hook(self):
            def fn(module, input, output):
                self.hooked_output = input[0].clone()  # hook decoder's input
            return fn

        def build_unsup_optimized_model(self):
            freeze_non_norm_layer(self.model.decoder)
            freeze_non_norm_layer(self.model.postnet)
            return nn.ModuleList([self.model.decoder, self.model.postnet])
        
        def build_matching_optimized_model(self):
            return self.ada.build_optimized_model()

        def build_optimized_model(self):
            if self.ada_stage == "unsup_tuning":
                return self.build_unsup_optimized_model()
            elif self.ada_stage == "matching":
                return self.build_matching_optimized_model()
            else:
                raise NotImplementedError

        def decoder_forard(self, x, mask):
            output, mask = self.model.decoder(x, mask)
            output = self.model.mel_linear(output)
            postnet_output = self.model.postnet(output) + output
            predictions = (output, postnet_output, mask)

            return predictions

        def common_ada_step(self, batch, batch_idx, train=True):
            qry_batch = batch[0][1][0]
            mask = get_mask_from_lengths(qry_batch[7]).to(self.device)

            feat = self.ada(qry_batch[6], qry_batch[7])
            # print(ada_enc_output.shape, self.src_enc_output.shape, mask.shape)
            match_loss = self.ada.match_loss_func(
                self.hooked_output.masked_select((~mask).unsqueeze(-1)), feat.masked_select((~mask).unsqueeze(-1)))
            
            predictions = self.decoder_forard(feat, mask)
            # print(qry_batch[6].shape, predictions[0].shape, predictions[1].shape, predictions[2].shape)
            loss = self.ada.recon_loss_func(qry_batch[6], predictions)

            loss_dict = {
                "Total Loss"       : loss[0] + match_loss,
                "Recon Loss"       : loss[0],
                "Mel Loss"         : loss[1],
                "Mel-Postnet Loss" : loss[2],
                "Match Loss"       : match_loss 
            }
            return loss_dict, predictions

        def common_step(self, batch, batch_idx, train=True):
            with torch.no_grad():  # Save computation resource in original pass
                if train:
                    u2s_loss_dict, u2s_output = super().common_step(batch, batch_idx, train)
                else:
                    u2s_loss_dict, u2s_output, attn = super().common_step(batch, batch_idx, train)
            ada_loss_dict, ada_output = self.common_ada_step(batch, batch_idx, train)

            # Hacking: Replace u2s output with ADA reconstructed output simply for visualization convenience
            hacked_output = (ada_output[0], ada_output[1], *u2s_output[2:])

            loss_dict = flat_merge_dict({
                "U2S": u2s_loss_dict,
                "ADA": ada_loss_dict
            })

            if self.ada_stage == "unsup_tuning":
                loss_dict["Total Loss"] = ada_loss_dict["Recon Loss"]
            elif self.ada_stage == "matching":
                loss_dict["Total Loss"] = ada_loss_dict["Total Loss"]
            else:
                raise NotImplementedError

            if train:
                return loss_dict, hacked_output
            return loss_dict, hacked_output, attn
    
    return TransEmbADASystem


def ssl_ada_class_factory(BaseClass, ada_stage: str):  # Class factory
    ada_cls = ada_class_factory(BaseClass, ada_stage)
    class TransEmbSSLADASystem(ada_cls): 
        def __init__(self, *args, **kwargs):
            super(TransEmbSSLADASystem, self).__init__(*args, **kwargs)
        
        def get_qry_ssl_repr(self, batch):
            qry_info = batch[0][3]

            # TODO: Mel version
            self.upstream.eval()
            with torch.no_grad():
                ssl_repr, _ = self.upstream.extract(qry_info["raw_feat"])  # B, L, n_layers, dim
                ssl_repr, _ = self.codebook_attention(ssl_repr)
                ssl_repr = ssl_repr.detach()

            return ssl_repr
        
        def _on_meta_batch_start(self, batch):
            """ Check meta-batch data """
            assert len(batch) == 1, "meta_batch_per_gpu"
            assert len(batch[0]) == 4, "sup + qry + sup_info + qry_info"
            assert len(batch[0][0]) == 1, "n_batch == 1"
            assert len(batch[0][0][0]) == 13, "data with 13 elements"
        
        def common_ada_step(self, batch, batch_idx, train=True):
            qry_batch = batch[0][1][0]
            ssl_repr = self.get_qry_ssl_repr(batch)
            ssl_repr = F.interpolate(ssl_repr.transpose(1, 2), size=qry_batch[8]).transpose(1, 2).detach()  # Interpolate

            # print(ssl_repr.shape)
            # print(qry_batch[7])
            # input()
            mask = get_mask_from_lengths(qry_batch[7]).to(self.device)

            feat = self.ada(ssl_repr, qry_batch[7], embed=False)
            match_loss = self.ada.match_loss_func(
                self.hooked_output.masked_select((~mask).unsqueeze(-1)), feat.masked_select((~mask).unsqueeze(-1)))
            
            predictions = self.decoder_forard(feat, mask)
            loss = self.ada.recon_loss_func(qry_batch[6], predictions)

            loss_dict = {
                "Total Loss"       : loss[0] + match_loss,
                "Recon Loss"       : loss[0],
                "Mel Loss"         : loss[1],
                "Mel-Postnet Loss" : loss[2],
                "Match Loss"       : match_loss 
            }
            return loss_dict, predictions       
        
        def on_load_checkpoint(self, checkpoint: dict) -> None:
            self.test_global_step = checkpoint["global_step"]
            state_dict = checkpoint["state_dict"]
            model_state_dict = self.state_dict()
            is_changed = False
            state_dict_pop_keys = []
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
                    if self.local_rank == 0:
                        print(f"Dropping parameter {k}")
                    state_dict_pop_keys.append(k)
                    is_changed = True

            # modify state_dict format to model_state_dict format
            for k in state_dict_pop_keys:
                state_dict.pop(k)

            # Copy parameters of encoder to ada encoder
            for k in model_state_dict:
                if k.startswith("ada.model.encoder"):
                    k_src = k.replace("ada.model.encoder", "model.encoder")
                    print(f"Copy parameters {k_src} => {k}")
                    if k in state_dict:
                        assert state_dict[k].shape == state_dict[k_src].shape, f"required shape: {state_dict[k].shape}, loaded shape: {state_dict[k_src].shape}"
                    state_dict[k] = state_dict[k_src].clone().detach()
            
            for k in model_state_dict:
                if k not in state_dict:
                    if k.split('.')[0] in ["upstream"]:
                        pass
                    else:
                        print("Reinitialized: ", k)
                    state_dict[k] = model_state_dict[k]

            if is_changed:
                checkpoint.pop("optimizer_states", None)

    return TransEmbSSLADASystem


def freeze_non_norm_layer(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm1d):
            for param in module.parameters():
                param.requires_grad = True
