import torch
import numpy as np
import json

from text.define import LANG_ID2SYMBOLS


LOCAL = True
DEBUG = False
CUDA_LAUNCH_BLOCKING = False
MAX_WORKERS = 4
DATAPARSERS = {}
CTC_DECODERS = {}
ALLSTATS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if LOCAL:
    CUDA_LAUNCH_BLOCKING = True  # TODO: Always crash on my PC if false
    MAX_WORKERS = 2


def merge_stats(stats_dict, keys):
    num = len(keys)
    pmi, pmx, pmu, pstd, emi, emx, emu, estd = \
        np.finfo(np.float64).max, np.finfo(np.float64).min, 0.0, 0.0, np.finfo(np.float64).max, np.finfo(np.float64).min, 0.0, 0.0
    for k in keys:
        pmu += stats_dict[k][2]
        pstd += stats_dict[k][3] ** 2
        emu += stats_dict[k][6]
        estd += stats_dict[k][7] ** 2
        pmi = min(pmi, stats_dict[k][0] * stats_dict[k][3] + stats_dict[k][2])
        pmx = max(pmx, stats_dict[k][1] * stats_dict[k][3] + stats_dict[k][2])
        emi = min(emi, stats_dict[k][4] * stats_dict[k][7] + stats_dict[k][6])
        emx = max(emx, stats_dict[k][5] * stats_dict[k][7] + stats_dict[k][6])

    pmu, pstd, emu, estd = pmu / num, (pstd / num) ** 0.5, emu / num, (estd / num) ** 0.5
    pmi, pmx, emi, emx = (pmi - pmu) / pstd, (pmx - pmu) / pstd, (emi - emu) / estd, (emx - emu) / estd
    
    return [pmi, pmx, pmu, pstd, emi, emx, emu, estd]


# Experiment parameters
USE_COMET = True
UPSTREAM = "hubert_large_ll60k"
UPSTREAM_DIM = 1024
LAYER_IDX = None
UPSTREAM_LAYER = 25

def set_upstream(x):
    global UPSTREAM
    global UPSTREAM_DIM
    global UPSTREAM_LAYER

    if x == "mel":
        UPSTREAM = x
        UPSTREAM_DIM = 80
    elif x in ["hubert", "wav2vec2"]:
        UPSTREAM = x
        UPSTREAM_DIM = 768
        UPSTREAM_LAYER = 13
    elif x in ["hubert_large_ll60k", "wav2vec2_large_ll60k", "wav2vec2_xlsr"]:
        UPSTREAM = x
        UPSTREAM_DIM = 1024
        UPSTREAM_LAYER = 25
    else:
        raise NotImplementedError


from torchaudio.models.decoder import ctc_decoder
def get_ctc_decoder(lang_id):
    global CTC_DECODERS
    if lang_id not in CTC_DECODERS:
        print(f"Constrct ctc decoder for lang_id: {lang_id}")
        CTC_DECODERS[lang_id] = ctc_decoder(
            lexicon=None,
            tokens=LANG_ID2SYMBOLS[lang_id],
            lm=None,
            nbest=1,
            beam_size=50,
            beam_size_token=30
        )
    return CTC_DECODERS[lang_id]


if __name__ == "__main__":
    import json
    print(json.dumps(ALLSTATS, indent=4))
