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

with open("stats.json", 'r', encoding="utf-8") as f:
    stats = json.load(f)
    ALLSTATS["global"] = stats["pitch"] + stats["energy"]

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
