from . import language
from . import phoneme_recognition


SYSTEM = {
    "fscl": language.TransEmbSystem,
    "fscl-tune": language.TransEmbTuneSystem,
    "semi-fscl": language.SemiTransEmbSystem,
    "semi-fscl-tune": language.SemiTransEmbTuneSystem,
    "multilingual-baseline": language.BaselineSystem,
    "multilingual-baseline-tune": language.BaselineTuneSystem,

    "pr-ssl-linear-tune": phoneme_recognition.SSLLinearTuneSystem,
    "pr-ssl-baseline": phoneme_recognition.SSLBaselineSystem,
    "pr-ssl-cluster": phoneme_recognition.SSLClusterSystem,
    "pr-ssl-codebook-cluster": phoneme_recognition.SSLCodebookClusterSystem,
    "pr-ssl-baseline-tune": phoneme_recognition.SSLBaselineTuneSystem,
    "pr-ssl-cluster-tune": phoneme_recognition.SSLClusterTuneSystem,
    "pr-fscl": phoneme_recognition.TransHeadSystem,
    "pr-fscl-tune": phoneme_recognition.TransHeadTuneSystem,
}

def get_system(algorithm):
    return SYSTEM[algorithm]
