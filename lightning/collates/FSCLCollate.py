import numpy as np
import torch
from functools import partial
from collections import defaultdict

from text.define import LANG_NAME2ID, LANG_ID2NAME
from lightning.build import build_all_speakers, build_id2symbols
from .utils import reprocess


class FSCLCollate(object):
    """
    Split data into support set and query set satisfying phoneme coverage condition. All 
    data in a batch belongs to the same language. This class is utilized for few-shot 
    language learning, including meta learning and transfer learning.

    Task: N spks (N >= 1), 1 way(lang), K shots, Q queries, B batch_size
    data: len(data) = K + Q     [SGD, K%B=0]
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

    def collate_fn(self, shots, queries, re_id=False, query_info=False):
        return partial(self._collate_fn, shots=shots, queries=queries, re_id=re_id, query_info=query_info)

    def _collate_fn(self, data, shots, queries, re_id=False, query_info=False):
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
            # print("S/Q ids", sup_ids, qry_ids)

            # st1 = time.time()
            sup_out.append(reprocess(data, sup_ids))
            # pad_sup = time.time() - st1

            qry_out.append(reprocess(data, qry_ids))
            # pad_qry = time.time() - st1

            lang_id = data[idxs[0]]["lang_id"]
            n_symbols = data[idxs[0]]["n_symbols"]
            sup_info = {}
            sup_info["lang_id"] = LANG_ID2NAME[lang_id]
            sup_info["n_symbols"] = n_symbols
            sup_info["phonemes"] = [data[idx]["text"] for idx in sup_ids]
            sup_info["raw_feat"] = [torch.from_numpy(data[idx]["raw-feat"]).float() for idx in sup_ids]
            sup_info["avg_frames"] = [data[idx]["avg-frames"] for idx in sup_ids]
            sup_info["lens"] = torch.LongTensor([sum(data[idx]["avg-frames"]) for idx in sup_ids])
            sup_info["max_len"] = max(sup_info["lens"])

            if query_info:
                qry_info = {}
                qry_info["raw_feat"] = [torch.from_numpy(data[idx]["raw-feat"]).float() for idx in qry_ids]
                qry_info["avg_frames"] = [data[idx]["avg-frames"] for idx in qry_ids]
                qry_info["lens"] = torch.LongTensor([sum(data[idx]["avg-frames"]) for idx in qry_ids])
                qry_info["max_len"] = max(qry_info["lens"])
                return (sup_out, qry_out, sup_info, qry_info)
        # calc_ref = time.time() - st1

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


class GeneralFSCLCollate(object):
    """
    Provide raw features and segments for speech representation extraction.
    This is a general version of FSCLCollate (without split).
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
            if mode == "sup":
                repr_info["n_symbols"] = data[0]["n_symbols"]
                repr_info["phonemes"] = [data[idx]["text"] for idx in idx_arr]
        else:
            raise NotImplementedError

        return (output, repr_info)
