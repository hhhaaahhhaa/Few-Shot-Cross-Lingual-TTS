# from .fastspeech2 import FastSpeech2
from .fastspeech2m import FastSpeech2
from .loss import FastSpeech2Loss
from .optimizer import ScheduledOptim
# from .reference_extractor import *


# REFERENCE_EXTRACTORS = {
#     "mel": MelExtractor, 
#     "hubert_large_ll60k": HubertExtractor,
#     "wav2vec2_large_ll60k": Wav2Vec2Extractor,
#     "wav2vec2_xlsr": XLSR53Extractor,
# }


# def get_reference_extractor_cls(tag: str):
#     return REFERENCE_EXTRACTORS[tag]
