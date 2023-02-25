from typing import Type

from . system import System
from . import language
from . import phoneme_recognition
from . import t2u


SYSTEM = {
    # "semi-fscl": language.SemiTransEmbSystem,
    # "semi-fscl-tune": language.SemiTransEmbTuneSystem,
    "baseline": language.BaselineSystem,
    # "baseline-tune": language.BaselineTuneSystem,
    "fscl": language.TransEmbSystem,
    "fscl-orig": language.TransEmbOrigSystem,
    "fscl-tune": language.TransEmbTuneSystem,

    "pr-ssl-linear-tune": phoneme_recognition.SSLLinearSystem,
    "pr-ssl-baseline": phoneme_recognition.SSLBaselineSystem,
    "pr-ssl-cluster": phoneme_recognition.SSLClusterSystem,
    # "pr-ssl-codebook-cluster": phoneme_recognition.SSLCodebookClusterSystem,
    "pr-ssl-baseline-tune": phoneme_recognition.SSLBaselineTuneSystem,
    "pr-ssl-cluster-tune": phoneme_recognition.SSLClusterTuneSystem,
    # "pr-fscl": phoneme_recognition.TransHeadSystem,
    # "pr-fscl-tune": phoneme_recognition.TransHeadTuneSystem,
    "pr-ssl-protonet": phoneme_recognition.SSLProtoNetSystem,

    "tacot2u": t2u.TacoT2USystem,
    "fscl-t2u": t2u.TransEmbSystem,
    "fscl-t2u-codebook": t2u.TransEmbCSystem,
    "fscl-t2u-codebook2": t2u.TransEmbC2System,

    "fscl-t2u-tune": t2u.TransEmbTuneSystem,
    "fscl-t2u-da-tune": t2u.TransEmbDATuneSystem,
    "fscl-t2u-e2e-tune": t2u.TransEmbE2ETuneSystem,
    "fscl-t2u-c-e2e-tune": t2u.TransEmbCE2ETuneSystem,
    "fscl-t2u-c2-e2e-tune": t2u.TransEmbC2E2ETuneSystem,
    "fscl-t2u-da-e2e-tune": t2u.TransEmbDAE2ETuneSystem,
    "fscl-t2u-c-da-e2e-tune": t2u.TransEmbCDAE2ETuneSystem,
    "fscl-t2u-c2-da-e2e-tune": t2u.TransEmbC2DAE2ETuneSystem,


    "conti-ae": language.ContiAESystem,
}

def get_system(algorithm) -> Type[System]:
    return SYSTEM[algorithm]
