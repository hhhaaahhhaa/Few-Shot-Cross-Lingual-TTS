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


class Segmentor(nn.Module):
    def __init__(self, model_config):
        super(Segmentor, self).__init__()
        hparams = Dict2Class(model_config)
        self.hparams = hparams
        self.device = 'cuda' if hparams.cuda else 'cpu'
        self.min_seg_size = hparams.min_seg_size
        self.max_seg_size = hparams.max_seg_size
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
                nn.Linear(100, 1),
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

    def calc_phi(self, rnn_out):
        batch_size, seq_len, feat_dim = rnn_out.shape
        rnn_out = torch.cat((torch.zeros((batch_size, 1, feat_dim)).to(rnn_out.device), rnn_out), dim=1)
        rnn_cum = torch.cumsum(rnn_out, dim=1)

        a = rnn_cum.repeat(1, seq_len, 1)
        b = rnn_cum.repeat(1, 1, seq_len).view(batch_size, -1, feat_dim)
        c = a.sub_(b).view(batch_size, seq_len, seq_len, feat_dim)

        d = rnn_out.repeat(1, 1, seq_len).view(batch_size, seq_len, seq_len, feat_dim)
        e = rnn_out.repeat(1, seq_len, 1).view(batch_size, seq_len, seq_len, feat_dim)
        phi = torch.cat([c, d, e], dim=-1)  # B, L + 1, L + 1, 3 * dim

        return phi

    def calc_all_scores(self, phi):
        scores = self.scorer(phi).squeeze(-1)  # B, L + 1, L + 1
        return scores

    def get_segmentation_score(self, scores, segmentations):
        """get_segmentation_score
        calculate the overall score for a whole segmentation for a batch of
        segmentations
        :param unary_scores:
        :param binary_scores:
        :param segmentations:
        returns: tensor of shape Bx1 where scores[i] = score for segmentation i
        """
        out_scores = torch.zeros((scores.shape[0])).to(scores.device)
        for seg_idx, seg in enumerate(segmentations):
            score = 0
            seg = zip(seg[:-1], seg[1:])
            for start, end in seg:
                score += scores[seg_idx, start, end]
            out_scores[seg_idx] = score

        return out_scores

    def segment_search(self, scores, lengths):
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
        batch_size, max_length = scores.shape[:2]
        scores, lengths = scores.to('cpu'), lengths.to('cpu')

        # Dynamic programming algorithm for inference (with batching)
        best_scores = torch.zeros(batch_size, max_length + 1)
        prev = torch.zeros(batch_size, max_length + 1)
        # segmentations = [[[0]] for _ in range(batch_size)]

        for i in range(1, max_length + 1):
            # Get scores of subsequences of seq[:i] that ends with i
            start_index = max(0, i - self.max_seg_size)
            current_scores = best_scores[:, start_index:i] + scores[:, start_index:i, i]

            # Choose the best scores and their corresponding indexes
            max_scores, k = torch.max(current_scores, 1)
            k = start_index + k # Convert indexes to numpy (relative to the starting index)

            # Add current best score and best segmentation
            best_scores[:, i] = max_scores
            prev[:, i] = k
            # for batch_index in range(batch_size):
            #     currrent_segmentation = segmentations[batch_index][k[batch_index]] + [i]
            #     segmentations[batch_index].append(currrent_segmentation)

        # Get real segmentations according to the real lengths of the sequences
        # in the batch
        pred_seg = []
        for i in range(batch_size):
            segs = []
            cur = lengths[i].item()
            while cur > 0:
                segs.append(cur)
                cur = prev[i][cur].item()
            segs.append(0)
            segs.reverse()
            pred_seg.append(segs)
        # for i, seg in enumerate(segmentations):
        #     last_index = lengths[i].item() - 1
        #     pred_seg.append(seg[last_index])

        return pred_seg

    def forward(self, x, length, gt_seg=None):
        """forward
        :param x:
        :param length:
        """
        results = {}

        # feed through rnn
        x = nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(x)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        # calc_phi will append zero frame at the begining so that the shape of score becomes len + 1.
        phi = self.calc_phi(rnn_out)  # B, L + 1, L + 1

        # feed through classifiers
        if self.use_cls:
            results['cls_out'] = self.classifier(rnn_out)
        if self.use_bin:
            results['bin_out'] = self.bin_classifier(rnn_out)

        # feed through search
        scores = self.calc_all_scores(phi)
        results['pred'] = self.segment_search(scores, length)
        results['pred_scores'] = self.get_segmentation_score(scores, results['pred'])

        if gt_seg is not None:
            results['gt_scores'] = self.get_segmentation_score(scores, gt_seg)

        return results
