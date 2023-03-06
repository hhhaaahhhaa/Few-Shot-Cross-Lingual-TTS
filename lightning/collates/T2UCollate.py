import numpy as np
import torch
from functools import partial
from collections import defaultdict

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


class T2UFSCLCollate(object):
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

    def collate_fn(self, shots, queries, re_id=False):
        return partial(self._collate_fn, shots=shots, queries=queries, re_id=re_id)

    def _collate_fn(self, data, shots, queries, re_id=False):
        # import time
        # st = time.time()
        batch_size = shots + queries
        data_size = len(data)
        # assert data_size % batch_size == 0, "Assume batch_size = 1 way * (shots + queries)"
        # assert data_size // batch_size > 0, "Assume batch_size = 1 way * (shots + queries)"
        assert data_size == batch_size, "len(data) = K + Q     [SGD, K%B=0]"
        
        idx_arr = np.arange(data_size)
        # concat embedding, re-id each phoneme
        if re_id:
            for idx in idx_arr:
                data[idx]["text"] += self.re_id_increment[data[idx]["symbol_id"]]

        # remap speakers and language
        for idx in idx_arr:
            data[idx]["speaker"] = self.speaker_map[data[idx]["speaker"]]
            data[idx]["lang_id"] = LANG_NAME2ID[data[idx]["lang_id"]]
        
        idx_arr = idx_arr.reshape((-1, batch_size))

        sup_out = list()
        qry_out = list()
        for idxs in idx_arr:  # Currently batch size is fixed to 1 when using this class.
            sup_ids, qry_ids = self.split_sup_qry(data, idxs, shots, queries)
            sup_out.append(reprocess(data, sup_ids))
            # here we need to sort qry_id by input text length so that lstm can work correctly
            qry_lens = np.array([data[qid]["text"].shape[0] for qid in qry_ids])
            qry_ids = qry_ids[np.argsort(-qry_lens)]

            qry_out.append(reprocess(data, qry_ids))

            sup_info = {}
            sup_info["raw_feat"] = [torch.from_numpy(data[idx]["raw_feat"]).float() for idx in sup_ids]
            sup_info["n_symbols"] = data[idxs[0]]["n_symbols"]
            sup_info["avg_frames"] = [data[idx]["avg_frames"] for idx in sup_ids]
            sup_info["phonemes"] = [data[idx]["text"] for idx in sup_ids]
            sup_info["lens"] = torch.LongTensor([sum(data[idx]["avg_frames"]) for idx in sup_ids])
            sup_info["max_len"] = max(sup_info["lens"])

        return (sup_out, qry_out, sup_info)

    def split_sup_qry(self, data, idxs, shots, queries):
        assert len(idxs) == shots + queries
        phn2idxs = defaultdict(list)
        for idx in idxs:
            phn_set = set(data[idx]["text"])
            for phn in phn_set:
                phn2idxs[phn].append(idx)

        sup_ids = []
        qry_ids = []
        for idx in idxs:
            flag = False
            if len(qry_ids) < queries:
                phn_set = set(data[idx]["text"])
                for phn in phn_set:
                    if len(phn2idxs[phn]) == 1:
                        sup_ids.append(idx)
                        flag = True
                        break
                if flag == False :
                    qry_ids.append(idx)
                    for phn in phn_set:
                        phn2idxs[phn].remove(idx)
            else:
                sup_ids.append(idx)

        ids_list = sup_ids + qry_ids
        sanity_check = (len(sup_ids) == shots and len(qry_ids) == queries)
        if sanity_check == False : # Force redestribution
            sup_ids = ids_list[:shots]
            qry_ids = ids_list[shots:]
        assert len(sup_ids) == shots and len(qry_ids) == queries
        return np.array(sup_ids), np.array(qry_ids)


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
