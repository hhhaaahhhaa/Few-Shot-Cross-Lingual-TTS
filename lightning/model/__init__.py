# from .fastspeech2 import FastSpeech2
from .fastspeech2m import FastSpeech2
from .loss import FastSpeech2Loss
from .optimizer import ScheduledOptim
from .reference_extractor import *


REFERENCE_EXTRACTORS = {
    "mel": MelExtractor, 
    "hubert": HubertExtractor,
    "wav2vec2": Wav2Vec2Extractor,
    "xlsr53": XLSR53Extractor,
}


def get_reference_extractor_cls(tag: str):
    return REFERENCE_EXTRACTORS[tag]
