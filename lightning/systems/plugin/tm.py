from typing import Type
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from dlhlp_lib.utils.tool import get_mask_from_lengths
from dlhlp_lib.utils.numeric import torch_exist_nan

from lightning.build import build_id2symbols
from lightning.model import ADAEncoder as TMEncoder
from ..language.embeddings import MultilingualEmbedding, SoftMultiAttCodebook


class ITextMatchingPlugIn(pl.LightningModule):
    """ Interface for TMPlugIn """
    def build_model(self, *args, **kwargs):
        raise NotImplementedError
    
    def build_optimized_model(self, *args, **kwargs):
        raise NotImplementedError
    
    def cluster(self, x, *args, **kwargs):
        raise NotImplementedError
    
    def cluster_loss_func(self, x, y, lengths, *args, **kwargs):
        pass

    def match_loss_func(self, x, y, lengths, *args, **kwargs):
        pass


class FilterLinear(nn.Module):
    """ very cool linear layer """
    def __init__(self, d_in, d_out, prob=0.0, min_ratio=0.1) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.prob = prob
        self.min_ratio = min_ratio

        self.linear = nn.Linear(d_in, d_out)
        self.act = nn.ReLU()

    def build_filter(self, x, alpha):
        shape = [1] * x.dim()
        shape[0] = x.shape[0]
        shape[-1] = -1
        n_filter = int(self.prob * self.d_out)
        alpha = alpha.view(tuple(shape))
        filter = torch.ones_like(alpha) * float("inf")
        filter[..., -n_filter:] = alpha[..., -n_filter:]
        return (1 - self.min_ratio) * torch.sigmoid(filter) + self.min_ratio

    def forward(self, x, alpha=None):
        x = self.linear(x)
        if alpha is None:
            return self.act(x)
        filter = self.build_filter(x, alpha)
        return self.act(x * filter)


class TMPlugIn(ITextMatchingPlugIn):
    def __init__(self, data_configs, model_config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data_configs = data_configs
        self.model_config = model_config
        self.build_model()

    def codebook_bind(self, module: SoftMultiAttCodebook):
        print("Codebook bind...")
        self.codebook_attention.emb_banks = module.emb_banks  # bind
        self.codebook_attention.emb_banks.requires_grad = False

    def build_model(self):
        encoder_dim = self.model_config["transformer"]["encoder_hidden"]
        self.embedding_model = MultilingualEmbedding(
            id2symbols=build_id2symbols(self.data_configs), dim=encoder_dim)
        self.encoder = TMEncoder(1, self.model_config)  # d_in is not important here since we will use embed=False

        self.n_layers = self.model_config["cluster"]["layer"]
        self.alpha = nn.Parameter(torch.randn(100, self.n_layers - 1))
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            if i != self.n_layers - 1:
                self.layers.append(FilterLinear(encoder_dim, encoder_dim, self.model_config["cluster"]["min_ratio"]))
            else:
                self.layers.append(nn.Linear(encoder_dim, encoder_dim))
        
        self.codebook_attention = SoftMultiAttCodebook(
            codebook_size=self.model_config["codebook"]["codebook_size"],
            embed_dim=self.model_config["transformer"]["encoder_hidden"],
            num_heads=self.model_config["codebook"]["nhead"],
        )
        self.use_matching = self.model_config.get("use_matching", True)

    def build_optimized_model(self):
        return self

    def cluster(self, x, lang_args):
        for i, layer in enumerate(self.layers):
            if i == self.n_layers - 1:
                x = layer(x)
            else:
                x = layer(x, self.alpha[lang_args, i])
        return x
    
    def forward(self, x, lengths, lang_args=None, mask=None, symbol_id=None):
        # TODO: make it lanugage dependent
        emb_texts = self.embedding_model(x, symbol_id)
        assert not torch_exist_nan(x)
        for p in self.embedding_model.parameters():
            assert not torch_exist_nan(p.data)
        x = self.encoder(emb_texts, lengths, embed=False, mask=mask)
        assert not torch_exist_nan(x)
        if self.use_matching:
            output, _ = self.codebook_attention(x)
            return output
        return x

    def loss_func(self, x, y, lengths):
        mask = get_mask_from_lengths(lengths).to(self.device)
        return F.mse_loss(x.masked_select((~mask).unsqueeze(-1)), y.masked_select((~mask).unsqueeze(-1)))

    def cluster_loss_func(self, x, y, lengths):
        return self.loss_func(x, y, lengths)
    
    def match_loss_func(self, x, y, lengths):
        return self.loss_func(x, y, lengths)
    
    # def build_embedding_table(self, ref_infos, return_attn=False):
    #     self.upstream.eval()
    #     hiddens, avg_frames_list, phonemes_list = [], [], []
    #     for info in ref_infos:
    #         with torch.no_grad():
    #             ssl_repr, _ = self.upstream.extract(info["raw_feat"])  # B, L, n_layers, dim
    #             ssl_repr = ssl_repr.detach()
    #         hiddens.extend([x1 for x1 in ssl_repr])
    #         avg_frames_list.extend(info["avg_frames"])
    #         phonemes_list.extend(info["phonemes"])
        
    #     table_pre = self.phoneme_query_extractor(hiddens, avg_frames_list, 
    #                         len(LANG_ID2SYMBOLS[ref_infos[0]["lang_id"]]), phonemes_list)  # 1, n_symbols, dim
        
    #     table, attn = self.codebook_attention(table_pre, need_weights=return_attn)
    #     table = table.squeeze(0)  # n_symbols, dim
    #     table[0].fill_(0)
    #     return table, attn
   