from typing import Optional
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from dlhlp_lib.utils import Dict2Class
from dlhlp_lib.common.wav2vec2U import SamePad
from dlhlp_lib.common.layers import WeightedSumLayer


class Classifier(nn.Module):
    def __init__(
        self, 
        n_in_layers: int,
        upstream_dim: int,
        specific_layer: Optional[int]=None,
        mode="readout",
    ):
        super(Classifier, self).__init__()
        self.mode = mode

        if self.mode == "readout":
            self.n_in_layers = n_in_layers
            self.weighted_sum = WeightedSumLayer(n_in_layers, specific_layer)

            self.layerwise_convolutions = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(upstream_dim, 768, 9, 1, 8),
                    SamePad(kernel_size=9, causal=True),
                    nn.ReLU(),
                ) for _ in range(self.n_in_layers)
            ])
            self.network = nn.Sequential(
                nn.Conv1d(768, 512, kernel_size=3, stride=1, padding=2),
                SamePad(kernel_size=3, causal=True),
                nn.ReLU(),
                nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=2),
                SamePad(kernel_size=3, causal=True),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=2),
                SamePad(kernel_size=3, causal=True),
                nn.ReLU(),
                nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=2),
                SamePad(kernel_size=3, causal=True),
                nn.ReLU(),
                nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=2),
                SamePad(kernel_size=3, causal=True),
                nn.ReLU(),
            )
            self.out = nn.Linear(32, 1)
        elif self.mode == "finetune":
            self.out = nn.Linear(upstream_dim, 1)
        else:
            raise NotImplementedError
    
    def forward(self, x):
        """
        Args:
            x: S3PRL output tensor with shape (B, L, n_layer, dim)
        """
        if self.mode == "readout":
            layers = []
            for i in range(self.n_in_layers):
                x_slc = x[:, :, i, :].permute(0, 2, 1).contiguous()
                layers.append(self.layerwise_convolutions[i](x_slc))
            x = torch.stack(layers, dim=0)  # n_in_layer, B, 768, L
            x = self.weighted_sum(x, dim=0)  # B, 768, L
            x = self.network(x)
            x = x.permute(0, 2, 1).contiguous()  # B, L, 32
        elif self.mode == "finetune":
            x = x[:, :, -1, :]  # B, L, upstream_dim
        out = self.out(x).squeeze(-1)  # B, L

        return out

    def get_tune_params(self) -> nn.ModuleList:
        modules = list(self.network.children())[-3:] + [self.out]
        return nn.ModuleList(modules)  



class NormedLinear(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.net = nn.Linear(*args, **kwargs)
    
    def forward(self, x):
        x = self.net(x)
        x = x / (torch.linalg.norm(self.net.weight, dim=1) + 1e-6)
        return x


class Segmentor(nn.Module):
    def __init__(self, model_config):
        super(Segmentor, self).__init__()
        hparams = Dict2Class(model_config)
        self.hparams = hparams
        self.device = 'cuda' if hparams.cuda else 'cpu'
        self.min_seg_size = hparams.min_seg_size
        self.max_seg_size = hparams.max_seg_size
        self.max_dpdp_size = hparams.max_dpdp_size
        assert hparams.max_dpdp_size >= hparams.max_seg_size
        self.use_bin = hparams.use_bin
        self.use_cls = hparams.use_cls

        self.rnn = nn.LSTM(hparams.rnn_input_size,
                           hparams.rnn_hidden_size,
                           num_layers=hparams.rnn_layers,
                           batch_first=True,
                           dropout=hparams.rnn_dropout,
                           bidirectional=hparams.birnn)

        # score calculation modules
        self.scorer = nn.Sequential(
                nn.PReLU(),
                nn.Linear((2 if hparams.birnn else 1) * 3 * hparams.rnn_hidden_size, 100),
                nn.PReLU(),
                NormedLinear(100, 1),
                )

        if self.use_cls:
            self.classifier = nn.Sequential(
                    nn.PReLU(),
                    nn.Linear((2 if hparams.birnn else 1) * hparams.rnn_hidden_size, hparams.n_classes * 2),
                    nn.PReLU(),
                    nn.Linear(hparams.n_classes * 2, hparams.n_classes),
                    )

        if self.use_bin:
            self.bin_classifier = nn.Sequential(
                    nn.PReLU(),
                    nn.Linear((2 if hparams.birnn else 1) * hparams.rnn_hidden_size, hparams.n_classes * 2),
                    nn.PReLU(),
                    nn.Linear(hparams.n_classes * 2, 2),
                    )

    def get_segmentation_score(self, rnn_out, rnn_cum, segmentations):
        """get_segmentation_score
        calculate the overall score for a whole segmentation for a batch of
        segmentations
        :param unary_scores:
        :param binary_scores:
        :param segmentations:
        returns: tensor of shape Bx1 where scores[i] = score for segmentation i
        """
        out_scores = torch.zeros((len(segmentations))).to(self.device)
        for seg_idx, seg in enumerate(segmentations):
            cs, ds, es = [], [], []
            for start, end in zip(seg[:-1], seg[1:]):
                cs.append(rnn_cum[seg_idx, end] - rnn_cum[seg_idx, start])
                ds.append(rnn_out[seg_idx, start])
                es.append(rnn_out[seg_idx, end])
            cs = torch.stack(cs, dim=0)  # n_seg, dim
            ds = torch.stack(ds, dim=0)  # n_seg, dim
            es = torch.stack(es, dim=0)  # n_seg, dim
            phi = torch.cat([cs, ds, es], dim=-1)  # n_seg, 3 * dim
            score = self.scorer(phi).squeeze(-1)
            # if seg_idx == 0:
            #     print(seg)
            #     print(score)
            score = score.sum()
            out_scores[seg_idx] = score

        return out_scores

    def calc_score(self, rnn_out, rnn_cum, st, ed):
        ed = min(ed, rnn_cum.shape[1])
        c = -rnn_cum[:, st:ed].unsqueeze(2) + rnn_cum[:, st:ed].unsqueeze(1)  # B, ed-st, ed-st, dim
        d = rnn_out[:, st:ed].unsqueeze(2).expand(-1, -1, ed-st, -1)
        e = rnn_out[:, st:ed].unsqueeze(1).expand(-1, ed-st, -1, -1)
        phi = torch.cat([c, d, e], dim=-1)  # B, ed-st, ed-st, 3 * dim
        scores = self.scorer(phi).squeeze(-1)  # B, ed-st, ed-st

        return scores
    
    # Matching to check vectorize implementation's correctness.
    # def calc_score_match(self, rnn_out, rnn_cum, st, ed):
    #     ed = min(ed, rnn_cum.shape[1])
    #     B = rnn_out.shape[0]
    #     scores = torch.zeros(B, ed-st, ed-st).to(self.device)
    #     for b in range(B):
    #         for i in range(ed-st):
    #             for j in range(ed-st):
    #                 c = rnn_cum[b, j] - rnn_cum[b, i]
    #                 d = rnn_out[b, i]
    #                 e = rnn_out[b, j]
    #                 phi = torch.cat([c, d, e], dim=-1)  # 3 * dim
    #                 score = self.scorer(phi)
    #                 scores[b][i][j] = score
    #     return scores

    def segment_search(self, rnn_out, rnn_cum, lengths):
        '''
        Apply dynamic programming algorithm for finding the best segmentation when
        k (the number of segments) is unknown.
        Parameters:
            batch :     A 3D torch tensor: (batch_size, sequence_size, input_size)
            lengths:    A 1D tensor containing the lengths of the batch sequences
            [gold_seg]: A python list containing batch_size lists with the gold
                        segmentations. If given, we will return the best segmentation
                        excluding the gold one, for the structural hinge loss with
                        margin algorithm (see Kiperwasser, Eliyahu, and Yoav Goldberg
                        "Simple and accurate dependency parsing using bidirectional LSTM feature representations).
        Notes:
            The algorithm complexity is O(n**2)
        '''
        batch_size, max_length = rnn_out.shape[:2]
        lengths = lengths.to('cpu')

        # Dynamic programming algorithm for inference (with batching)
        best_scores = torch.zeros(batch_size, max_length).to(self.device)
        prev = torch.zeros(batch_size, max_length).to(self.device)
        # segmentations = [[[0]] for _ in range(batch_size)]

        score_start_idx = 0
        score_end_idx = self.max_dpdp_size
        scores = self.calc_score(rnn_out, rnn_cum, score_start_idx, score_end_idx)
        # scores2 = self.calc_score_match(rnn_out, rnn_cum, score_start_idx, score_end_idx)
        for i in range(1, max_length):
            # Get scores of subsequences of seq[:i] that ends with i
            start_index = max(0, i - self.max_seg_size)

            # Calculte score online.
            # Calculate once for fix iterations to achieve good tradeoff between GPU usage and runtime!
            if i == score_end_idx:
                score_start_idx += self.max_dpdp_size // 2
                score_end_idx += self.max_dpdp_size // 2
                scores = self.calc_score(rnn_out, rnn_cum, score_start_idx, score_end_idx)

            assert start_index >= score_start_idx
            assert i - score_start_idx < self.max_dpdp_size
            current_scores = best_scores[:, start_index:i] + scores[:, start_index-score_start_idx:i-score_start_idx, i-score_start_idx]

            # Choose the best scores and their corresponding indexes
            max_scores, k = torch.max(current_scores, 1)
            k = start_index + k # Convert indexes to numpy (relative to the starting index)

            # Add current best score and best segmentation
            best_scores[:, i] = max_scores
            prev[:, i] = k

        # Get real segmentations according to the real lengths of the sequences in the batch
        pred_seg = []
        for i in range(batch_size):
            segs = []
            cur = lengths[i].item()
            while cur > 0:
                segs.append(cur)
                cur = int(prev[i][cur].item())
            segs.append(0)
            segs.reverse()
            pred_seg.append(segs)
            # if i == 0:
            #     print("predict")
            #     print(segs)
            #     for s in segs:
            #         if s == 0:
            #             continue
            #         s1 = int(prev[i][s].item())
            #         print(best_scores[i, s] - best_scores[i, s1])

        return pred_seg

    def forward(self, x, length, gt_seg=None):
        """forward
        :param x:
        :param length:
        """
        results = {}

        # feed through rnn
        x = nn.utils.rnn.pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(x)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        # feed through classifiers
        if self.use_cls:
            results['cls_out'] = self.classifier(rnn_out)
        if self.use_bin:
            results['bin_out'] = self.bin_classifier(rnn_out)

        # feed through search
        # score need to be calculated online since GPU usage is way too large
        # scores = self.calc_all_scores(phi)

        # calc_phi will append zero frame at the begining so that the shape of score becomes len + 1.
        batch_size, _, feat_dim = rnn_out.shape
        rnn_out = torch.cat((torch.zeros((batch_size, 1, feat_dim)).to(rnn_out.device), rnn_out), dim=1)
        rnn_cum = torch.cumsum(rnn_out, dim=1)

        results['pred'] = self.segment_search(rnn_out, rnn_cum, length)
        results['pred_scores'] = self.get_segmentation_score(rnn_out, rnn_cum, results['pred'])

        if gt_seg is not None:
            results['gt_scores'] = self.get_segmentation_score(rnn_out, rnn_cum, gt_seg)

        return results
