import numpy as np
import torch
from functools import partial

from text.define import LANG_NAME2ID, LANG_ID2NAME
from lightning.utils.tool import pad_2D
from lightning.build import build_all_speakers, build_id2symbols
from .utils import reprocess


class SemiCollate(object):
    """
    Provide raw features and segments and for speech representation extraction + provide confidence scores.
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

    def collate_fn(self, sort=False, re_id=False, mode="sup"):
        return partial(self._collate_fn, sort=sort, re_id=re_id, mode=mode)

    def _collate_fn(self, data, sort=False, re_id=False, mode="sup"):
        data_size = len(data)

        if sort:
            len_arr = np.array([d["duration"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        # concat embedding, re-id each phoneme
        if re_id:
            for idx in idx_arr:
                data[idx]["text"] += self.re_id_increment[data[idx]["symbol_id"]]

        # remap speakers and language
        for idx in idx_arr:
            data[idx]["speaker"] = self.speaker_map[data[idx]["speaker"]]
            data[idx]["lang_id"] = LANG_NAME2ID[data[idx]["lang_id"]]
        
        output = reprocess(data, idx_arr, mode=mode)

        repr_info = {}
        lang_id = data[0]["lang_id"]
        if mode in ["sup", "unsup"]:
            repr_info["raw_feat"] = [torch.from_numpy(data[idx]["raw-feat"]).float() for idx in idx_arr]
            repr_info["avg_frames"] = [data[idx]["avg-frames"] for idx in idx_arr]
            repr_info["lens"] = torch.LongTensor([sum(data[idx]["avg-frames"]) for idx in idx_arr])
            repr_info["max_len"] = max(repr_info["lens"])
            repr_info["lang_id"] = LANG_ID2NAME[lang_id]
            repr_info["phoneme_score"] = torch.from_numpy(pad_2D([data[idx]["phoneme_score"] for idx in idx_arr])).float()
            if mode == "sup":
                repr_info["n_symbols"] = data[0]["n_symbols"]
                repr_info["phonemes"] = [data[idx]["text"] for idx in idx_arr]
        else:
            raise NotImplementedError

        return (output, repr_info)
