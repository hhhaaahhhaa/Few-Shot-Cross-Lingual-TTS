import torch

from dlhlp_lib.utils import batchify, segment2duration

import Define
from Parsers.utils import read_queries_from_txt
from text import text_to_sequence


def generate_reference_info(data_config, batch_size=16):
    """
    Generate reference information from data_config directly, mainly used when tuning.
    Returns a list, which is reference information for each batch.
    Information format:
        {
            "raw_feat": [],
            "lens": [],
            "max_len": None,
            "phonemes": [],
            "avg_frames": [],
            "lang_id": lang_id,
            "symbol_id": symbol_id,
        }
    """
    data_parser = Define.DATAPARSERS[data_config["name"]]
    lang_id = data_config["lang_id"]
    symbol_id = data_config["symbol_id"]
    queries = read_queries_from_txt(data_config["subsets"]["train"])

    infos = []
    # Extract representation information batchwise
    for query_batch in batchify(queries, batch_size=batch_size):
        info = {
            "raw_feat": [],
            "lens": [],
            "max_len": None,
            "phonemes": [],
            "avg_frames": [],
            "lang_id": lang_id,
            "symbol_id": symbol_id,
        }
        for query in query_batch:
            # Transfer learning module
            segment = data_parser.mfa_segment.read_from_query(query)
            if Define.UPSTREAM == "mel":
                pass  # TODO: Mel version
            else:
                raw_feat = data_parser.wav_trim_16000.read_from_query(query)
                avg_frames = segment2duration(segment, fp=0.02)
                info["raw_feat"].append(torch.from_numpy(raw_feat).float())
                info["avg_frames"].append(avg_frames)
                info["lens"].append(sum(avg_frames))

            phns = data_parser.phoneme.read_from_query(query)
            phns = f"{{{phns}}}"  # match input format of text_to_sequence()
            phns = text_to_sequence(phns, data_config["text_cleaners"], lang_id)
            info["phonemes"].append(phns)
        info["lens"] = torch.LongTensor(info["lens"])
        info["max_len"] = max(info["lens"])
        infos.append(info)
    
    return infos
