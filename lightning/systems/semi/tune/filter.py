import torch


class BaselineFilter(object):
    def __init__(self, threshold: float=0.0) -> None:
        self.threshold = threshold
        
    def calc(self, scores: torch.FloatTensor, lengths) -> torch.BoolTensor:
        """
        Args:
            scores: Tensor with shape (B, L, n_symbols)
        Return:
            Frame-level binary mask indicating pseudo label should be accepted or not.
            0: unmasked, 1: masked
        """
        confidences, idxs = scores.max(dim=-1)
        sentence_confidences = confidences.sum(dim=-1) / lengths
        accepted = (sentence_confidences >= self.threshold)
        return (~accepted).expand(-1, scores.shape[1])


class FramewiseFilter(object):
    def __init__(self, threshold: float=0.0) -> None:
        self.threshold = threshold
        
    def calc(self, scores: torch.FloatTensor, lengths) -> torch.BoolTensor:
        """
        Args:
            scores: Tensor with shape (B, L, n_symbols)
        Return:
            Frame-level binary mask indicating pseudo label should be accepted or not.
            0: unmasked, 1: masked
        """
        confidences, idxs = scores.max(dim=-1)
        accepted = (confidences >= self.threshold)
        return ~accepted
