import torch
import numpy as np
import json


LOCAL = True
USE_OLD_CONFIG = False
DEBUG = False
CUDA_LAUNCH_BLOCKING = False
MAX_WORKERS = 4
DATAPARSERS = {}
ALLSTATS = {}
CTC_DECODERS = {}

# with open("stats.json", 'r', encoding="utf-8") as f:
#     stats = json.load(f)
#     ALLSTATS["global"] = stats["pitch"] + stats["energy"]

"""
Use old ALLSTAT when reproducing old baseline
"""
def merge_stats(stats_dict, keys):
    num = len(keys)
    pmi, pmx, pmu, pstd, emi, emx, emu, estd = \
        np.finfo(np.float64).max, np.finfo(np.float64).min, 0.0, 0.0, np.finfo(np.float64).max, np.finfo(np.float64).min, 0.0, 0.0
    for k in keys:
        pmu += stats_dict[k][2]
        pstd += stats_dict[k][3] ** 2
        emu += stats_dict[k][6]
        estd += stats_dict[k][7] ** 2

        pmi = min(pmi, stats_dict[k][0])
        pmx = max(pmx, stats_dict[k][1])
        emi = min(emi, stats_dict[k][4])
        emx = max(emx, stats_dict[k][5])

        # Corpus wise
        # print(f"{k}:")
        # print((stats_dict[k][0] - stats_dict[k][2]) / stats_dict[k][3])
        # print((stats_dict[k][1] - stats_dict[k][2]) / stats_dict[k][3])
        # print((stats_dict[k][4] - stats_dict[k][6]) / stats_dict[k][7])
        # print((stats_dict[k][5] - stats_dict[k][6]) / stats_dict[k][7])
        # pmi = min(pmi, (stats_dict[k][0] - stats_dict[k][2]) / stats_dict[k][3])
        # pmx = max(pmx, (stats_dict[k][1] - stats_dict[k][2]) / stats_dict[k][3])
        # emi = min(emi, (stats_dict[k][4] - stats_dict[k][6]) / stats_dict[k][7])
        # emx = max(emx, (stats_dict[k][5] - stats_dict[k][6]) / stats_dict[k][7])

    pmu, pstd, emu, estd = pmu / num, (pstd / num) ** 0.5, emu / num, (estd / num) ** 0.5
    
    # Corpus wise
    # pmi = pmi * pstd + pmu
    # pmx = pmx * pstd + pmu
    # emi = emi * estd + emu
    # emx = emx * estd + emu
    
    return [pmi, pmx, pmu, pstd, emi, emx, emu, estd]

STATSDICT = {
    0: "/work/u5550322/fscl/LibriTTS",
    1: "/work/u5550322/fscl/AISHELL-3",
    2: "/work/u5550322/fscl/CSS10/french",
    3: "/work/u5550322/fscl/CSS10/german",
    4: "/work/u5550322/fscl/CSS10/spanish",
    5: "/work/u5550322/fscl/CSS10/spanish",
    6: "/work/u5550322/fscl/JSUT",
    7: "/work/u5550322/fscl/GlobalPhone/cz",
    8: "/work/u5550322/fscl/kss",
}

for k, v in STATSDICT.items():
    try:
        with open(f"{v}/stats.json") as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"]
            ALLSTATS[k] = stats
    except:
        ALLSTATS[k] = None

LANGUSAGE = [0, 1, 8]  # Dirty! Need to adjust manually for every experiment.
ALLSTATS["global"] = merge_stats(ALLSTATS, LANGUSAGE)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if LOCAL:
    CUDA_LAUNCH_BLOCKING = True  # TODO: Always crash on my PC if false
    MAX_WORKERS = 2


# Experiment parameters
USE_COMET = False
UPSTREAM = "hubert_large_ll60k"
UPSTREAM_DIM = 1024
LAYER_IDX = None
UPSTREAM_LAYER = 25
ADAPART = False
NOLID = False
TUNET2U = False
ATTTEMP = False

PL_CONF = 0.0


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
    elif x in ["hubert_large_ll60k", "wav2vec2_large_ll60k", "xlsr_53"]:
        UPSTREAM = x
        UPSTREAM_DIM = 1024
        UPSTREAM_LAYER = 25
    else:
        raise NotImplementedError
