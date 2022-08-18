from .phoneme_recognition.TransHeadTune import TransHeadTuneSystem
from . import language
from . import phoneme_recognition

# Old
# SYSTEM = {
#     "meta": TTS.TransEmb.TransEmbSystem,
#     "baseline": TTS.baseline.BaselineSystem,
#     "asr-codebook": ASR.codebook.CodebookSystem,
#     "asr-baseline": ASR.center.CenterSystem,
#     "asr-center": ASR.center.CenterSystem,
#     "asr-center-ref": ASR.center_ref.CenterRefSystem,
#     "dual-meta": Dual.dual.DualMetaSystem,
#     "dual-tune": Dual.dual_tune.DualMetaTuneSystem,
#     "meta-tune": TTS.TransEmb_tune.TransEmbTuneSystem,
# }

SYSTEM = {
    "fscl": language.TransEmbSystem,
    "fscl-tune": language.TransEmbTuneSystem,
    "semi-fscl": language.SemiTransEmbSystem,
    "semi-fscl-tune": language.SemiTransEmbTuneSystem,
    "multilingual-baseline": language.BaselineSystem,
    "multilingual-baseline-tune": language.BaselineTuneSystem,
    "pr-ssl-baseline": phoneme_recognition.SSLBaselineSystem,
    "pr-ssl-baseline-tune": phoneme_recognition.SSLBaselineTuneSystem,
    "pr-fscl": phoneme_recognition.TransHeadSystem,
    "pr-fscl-tune": phoneme_recognition.TransHeadTuneSystem,
}

def get_system(algorithm):
    return SYSTEM[algorithm]
