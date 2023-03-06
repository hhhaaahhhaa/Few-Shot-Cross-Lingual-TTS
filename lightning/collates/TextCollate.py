import numpy as np
from functools import partial

from lightning.build import build_id2symbols, build_all_speakers
from .utils import reprocess


class TextCollate(object):
    """
    Inference dataset only have text and target speaker information.
    Require a map from symbol_id to symbols for re-id for multilingual-batch inference.
    """
    def __init__(self, data_configs):
        # calculate re-id increment
        id2symbols = build_id2symbols(data_configs)
        increment = 0
        self.re_id_increment = {}
        for k, v in id2symbols.items():
            self.re_id_increment[k] = increment
            increment += len(v)
        self.n_symbols = increment

        # calculate speaker map
        speakers = build_all_speakers(data_configs)
        self.speaker_map = {spk: i for i, spk in enumerate(speakers)}

    def collate_fn(self, sort=False, re_id=True):
        return partial(self._collate_fn, sort=sort, re_id=re_id)

    def _collate_fn(self, data, sort=False, re_id=True):
        data_size = len(data)

        if sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        # concat embedding, re-id each phoneme
        if re_id:
            for idx in idx_arr:
                data[idx]["text"] += self.re_id_increment[data[idx]["lang_id"]]
        output = reprocess(data, idx_arr, mode="inference")

        return output
