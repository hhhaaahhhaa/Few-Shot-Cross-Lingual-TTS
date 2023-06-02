import torch


class BaselineFilter(object):
    def __init__(self, threshold: float=0.0) -> None:
        self.threshold = threshold
        
    def mask_gen(self, scores: torch.FloatTensor, durations=None, *args, **kwargs) -> torch.BoolTensor:
        """
        Args:
            scores: Tensor with shape (B, L, n_symbols)
        Return:
            Frame-level binary mask indicating pseudo label should be accepted or not.
            0: accepted, 1: rejected
        """
        confidences, idxs = scores.max(dim=-1)

        sentence_confidences = (confidences * durations).sum(dim=-1) / durations.sum(dim=-1)  # weighted avg
        accepted = (sentence_confidences >= self.threshold)
        return (~accepted).unsqueeze(-1).expand(-1, scores.shape[1])


class FramewiseFilter(object):
    def __init__(self, threshold: float=0.0) -> None:
        self.threshold = threshold
        
    def mask_gen(self, scores: torch.FloatTensor, *args, **kwargs) -> torch.BoolTensor:
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
