from . import language

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
    "multilingual-baseline-tune": language.BaselineSystem,
}

def get_system(algorithm):
    return SYSTEM[algorithm]
