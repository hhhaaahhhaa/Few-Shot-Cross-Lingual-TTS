import torch


class BaselineFilter(object):
    def __init__(self, threshold: float=0.0) -> None:
        self.threshold = threshold
        
    def mask_gen(self, scores: torch.FloatTensor, lengths, durations, **args) -> torch.BoolTensor:
        """
        Args:
            scores: Tensor with shape (B, L, n_symbols)
        Return:
            Frame-level binary mask indicating pseudo label should be accepted or not.
            0: accepted, 1: rejected
        """
        confidences, idxs = scores.max(dim=-1)
        # TODO: expand first here before taking average
        # for loop?

        sentence_confidences = confidences.sum(dim=-1) / durations.sum()
        accepted = (sentence_confidences >= self.threshold)
        return (~accepted).expand(-1, scores.shape[1])


class FramewiseFilter(object):
    def __init__(self, threshold: float=0.0) -> None:
        self.threshold = threshold
        
    def mask_gen(self, scores: torch.FloatTensor, lengths, **args) -> torch.BoolTensor:
        """
        Args:
            scores: Tensor with shape (B, L, n_symbols)
        Return:
            Frame-level binary mask indicating pseudo label should be accepted or not.
            0: accepted, 1: rejected
        """
        confidences, idxs = scores.max(dim=-1)
        accepted = (confidences >= self.threshold)
        return ~accepted
