import numpy as np
import torch
from functools import partial

from lightning.build import build_id2symbols, build_all_speakers
from lightning.utils.tool import pad_1D
from text.define import LANG_NAME2ID


class T2UCollate(object):
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
                data[idx]["text"] += self.re_id_increment[data[idx]["symbol_id"]]

        # remap speakers and language
        for idx in idx_arr:
            data[idx]["speaker"] = self.speaker_map[data[idx]["speaker"]]
            data[idx]["lang_id"] = LANG_NAME2ID[data[idx]["lang_id"]]
        
        output = reprocess(data, idx_arr)

        return output


def reprocess(data, idxs):
    """
    Pad data and calculate length of data.
    """
    ids = [data[idx]["id"] for idx in idxs]
    speakers = [data[idx]["speaker"] for idx in idxs]
    lang_ids = [data[idx]["lang_id"] for idx in idxs]
    target_symbol_ids = [data[idx]["target_symbol_id"] for idx in idxs]
    speakers = np.array(speakers)
    lang_ids = np.array(lang_ids)

    texts = [data[idx]["text"] for idx in idxs]
    raw_texts = [data[idx]["raw_text"] for idx in idxs]
    text_lens = np.array([text.shape[0] for text in texts])
    texts = pad_1D(texts)

    units = [data[idx]["unit"] for idx in idxs]
    unit_lens = np.array([unit.shape[0] for unit in units])
    units = pad_1D(units)

    speaker_args = torch.from_numpy(speakers).long()

    return (
        ids,
        raw_texts,
        speaker_args,
        torch.from_numpy(texts).long(),
        torch.from_numpy(text_lens),
        max(text_lens),
        torch.from_numpy(units).long(),
        torch.from_numpy(unit_lens),
        max(unit_lens),
        torch.from_numpy(lang_ids).long(),
        target_symbol_ids,
    )
