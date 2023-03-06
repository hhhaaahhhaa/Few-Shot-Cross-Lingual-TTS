"""
Loss function
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PRFramewiseLoss(nn.Module):
    """ Cross Entropy Loss """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, labels, preds):
        preds = preds.transpose(1, 2)  # B, N, L
        target = labels  # B, L
        return self.loss(preds, target)


class OrthoLoss(nn.Module):
    """ Orthogonal Loss """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x: Tensor with shape (*other, size, dim).
        """
        gram = F.cosine_similarity(x.unsqueeze(-3), x.unsqueeze(-2), dim=-1)  # (*other, size, size)
        return gram.mean()
