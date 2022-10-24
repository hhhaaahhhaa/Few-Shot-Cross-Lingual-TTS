from typing import List, Dict, Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
import random

from dlhlp_lib.utils import DataPool

import Define
from lightning.utils.tool import torch_exist_nan


class SegmentationLevelAverage(pl.LightningModule):
    """
    Perform segmentation-level average on a batch of speech representations base on segmentations.
    e.g. | 3 2 1 | 5 1 | 6 | 7 8 9 6 | -> [2, 3, 6, 7.5]  
    Output tensor shape is (B, L, *dims).
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, representations, avg_frames: List[List[int]]):
        B, _, *dims = representations.shape
        avg_repr = []
        for d_list, repr in zip(avg_frames, representations):
            pos = 0
            for i, d in enumerate(d_list):
                if d > 0 and not torch_exist_nan(repr[pos: pos + d]):
                    repr[:, i] = torch.mean(repr[pos: pos + d], axis=0)
                else:
                    repr[:, i] = torch.zeros(dims).to(self.device)
                pos += d
            repr = repr[:len(d_list)]  # len(d_list), *dims
            avg_repr.append(repr)
        avg_repr = torch.nn.utils.rnn.pad_sequence(avg_repr, batch_first=True)  # B, L, *dims
        if Define.DEBUG:
            print("Average representations shape:")
            print(avg_repr.shape)
        return avg_repr


class PhonemeQueryExtractor(pl.LightningModule):
    """
    Perform phoneme-level average first, then reduct across the same phoneme class within the batch again.
    Support different second stage reduction modes: "average", "random", "pool".
    Output tensor shape is (1, n_symbols, *dims) for consistency.

    If 'two_stage' is false, skip phoneme-level average step, and second stage will operates on frame-level.
    """
    def __init__(self, mode: str="average", two_stage: bool=True) -> None:
        super().__init__()
        self.two_stage = two_stage
        if mode == "average":
            self.reduction = AverageReductionModule()
        elif mode == "random":
            self.reduction = RandomSelectReductionModule()
        elif mode == "pool":
            self.reduction = PoolReductionModule()
        else:
            raise NotImplementedError

    def forward(self, representations, avg_frames: List[List[int]], n_symbols: int, phonemes):
        table = {i: [] for i in range(n_symbols)}

        _, *dims = representations[0].shape
        for phoneme, d_list, repr in zip(phonemes, avg_frames, representations):
            assert not torch_exist_nan(repr)
            pos = 0
            for p, d in zip(phoneme, d_list):
                if d > 0:
                    if self.two_stage:
                        table[int(p)].append(repr[pos: pos + d].mean(dim=0))
                    else:
                        for r in repr[pos: pos + d]:
                            table[int(p)].append(r)
                pos += d

        phn_query = self.reduction(list(range(n_symbols)), table, dims)
        phn_query = phn_query.unsqueeze(0)  # 1, n_symbols, layer, dim
        if Define.DEBUG:
            print("Phoneme query shape:")
            print(phn_query.shape)
        return phn_query



class AverageReductionModule(pl.LightningModule):
    """
    Given K class labels and table of representations, average representations class-wise.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, class_labels, table, repr_shape):
        """
        Args:
            class_labels: Classes' name.
            table: Dictionary with class name as key and list of representations belong to the
                label as value.
            repr_shape: Shape of single representation.
        Return:
            Tensor with shape (n_class, *repr_shape).
        """
        avg_repr = []
        for c in class_labels:
            if len(table[c]) == 0:
                avg_repr.append(torch.zeros(repr_shape).to(self.device))
            else:
                avg_repr.append(torch.mean(torch.stack(table[c], axis=0), axis=0))
        avg_repr = torch.stack(avg_repr, dim=0).float()

        return avg_repr


class RandomSelectReductionModule(pl.LightningModule):
    """
    Given K class labels and some representations, random select one representation for each class.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, class_labels, table, repr_shape):
        """
        Args:
            class_labels: Classes' name.
            table: Dictionary with class label as key and list of representations belong to the
                label as value.
            repr_shape: Shape of single representation.
        Return:
            Tensor with shape (n_class, *repr_shape).
        """
        selected_repr = []
        for c in class_labels:
            if len(table[c]) == 0:
                selected_repr.append(torch.zeros(repr_shape).to(self.device))
            else:
                idx = random.randint(0, len(table[c]) - 1)
                selected_repr.append(table[c][idx])
        selected_repr = torch.stack(selected_repr, dim=0).float()

        return selected_repr


class PoolReductionModule(pl.LightningModule):
    """
    Given K class labels and some representations, perform partial average representation for each class.
    A representation pool is maintained for each class.
    """

    pools: Dict[Any, DataPool]

    def __init__(self, max_size: int=100):
        super().__init__()
        self.max_size = max_size
        self.pools = {}
        
    def forward(self, class_labels, table, repr_shape):
        """
        Args:
            class_labels: Classes' name.
            table: Dictionary with class label as key and list of representations belong to the
                label as value.
            repr_shape: Shape of single representation.
        Return:
            Tensor with shape (n_class, *repr_shape).
        """
        pooled_repr = []
        for c in class_labels:
            if len(table[c]) == 0:
                pooled_repr.append(torch.zeros(repr_shape).to(self.device))
            else:
                self.pools[c] = DataPool(max_size=self.max_size, auto_resize=False)
                self.pools[c].extend(table[c])
                self.pools[c].resize()

                pooled = torch.mean(torch.stack(self.pools[c]._data, axis=0), axis=0)
                pooled_repr.append(pooled)
        pooled_repr = torch.stack(pooled_repr, dim=0).float()

        return pooled_repr
