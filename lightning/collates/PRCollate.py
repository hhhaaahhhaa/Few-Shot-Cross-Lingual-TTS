import numpy as np
import torch
from functools import partial
from collections import defaultdict

from text.define import LANG_ID2SYMBOLS
from .utils import reprocess_pr


class SSLPRCollate(object):
    def __init__(self):
        pass

    def collate_fn(self, sort=False, mode="sup"):
        return partial(self._collate_fn, sort=sort, mode=mode)

    def _collate_fn(self, data, sort=False, mode="sup"):
        data_size = len(data)

        if sort:
            len_arr = np.array([d["duration"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)
        labels = reprocess_pr(data, idx_arr, mode=mode)

        repr_info = {}
        repr_info["wav"] = [torch.from_numpy(data[idx]["wav"]).float() for idx in idx_arr]
        repr_info["n_symbols"] = data[0]["n_symbols"]
        repr_info["lang_id"] = data[0]["lang_id"]

        return (labels, repr_info)
