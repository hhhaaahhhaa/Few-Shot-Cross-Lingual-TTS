from typing import List, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl


class Padding(pl.LightningModule):
    def __init__(self):
        pass

    def extract(self, data: Union[List[torch.Tensor], List[np.ndarray]], norm=False) -> Tuple[torch.FloatTensor, List[int]]:
        """
        Perform simple padding and normalization.
        Args:
            data: List of data represented as numpy arrays or torch tensors.
            norm: Normalize representation or not.
        Return:
            All hidden states and n_frames (for client to remove padding). Hidden states shape are (B, L, dim).
        """
        n_frames = [d.shape[0] for d in data]
        if isinstance(data[0], np.ndarray):
            temp = [torch.from_numpy(d).float().to(self.device) for d in data]
        else:
            temp = data        
        representation = torch.nn.utils.rnn.pad_sequence(temp, batch_first=True)  # B, L, dim
        if norm:
            representation = torch.nn.functional.normalize(representation, dim=-1)
              
        return representation, n_frames
