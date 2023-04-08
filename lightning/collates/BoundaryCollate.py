import numpy as np
import torch
from functools import partial
from collections import defaultdict
import time

from text.define import LANG_ID2SYMBOLS
from .utils import reprocess_bd, reprocess_bd2


class BoundaryCollate(object):
    def __init__(self, data_configs):
        pass

    def collate_fn(self, sort=False):
        return partial(self._collate_fn, sort=sort)

    def _collate_fn(self, data, sort=False):
        data_size = len(data)

        if sort:
            len_arr = np.array([d["mel"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)
        labels = reprocess_bd(data, idx_arr)

        repr_info = {}
        lang_id = data[0]["lang_id"]
        repr_info["raw_feat"] = [torch.from_numpy(data[idx]["raw_feat"]).float() for idx in idx_arr]
        repr_info["avg_frames"] = [data[idx]["avg_frames"] for idx in idx_arr]
        repr_info["lens"] = torch.LongTensor([sum(data[idx]["avg_frames"]) for idx in idx_arr])
        repr_info["max_len"] = max(repr_info["lens"])
        repr_info["lang_id"] = lang_id

        return (labels, repr_info)


class BoundaryCollate2(object):
    def __init__(self, data_configs):
        pass

    def collate_fn(self, sort=False):
        return partial(self._collate_fn, sort=sort)

    def _collate_fn(self, data, sort=False):
        # st = time.time()
        data_size = len(data)

        if sort:
            len_arr = np.array([d["mel"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)
        labels = reprocess_bd2(data, idx_arr)
        # print("collate time: ", time.time() - st)

        return labels
