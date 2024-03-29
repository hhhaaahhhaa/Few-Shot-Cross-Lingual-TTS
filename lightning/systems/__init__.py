from typing import Type

from . system import System
from . import language
from . import phoneme_recognition
from . import t2u


SYSTEM_SYNTHESIS = {
    "baseline": language.BaselineSystem,
    "baseline-tune": language.BaselineTuneSystem,
    "conti-ae": language.ContiAESystem,

    "fscl-orig": language.TransEmbOrigSystem,
    "fscl-orig-tune": language.TransEmbOrigTuneSystem,
    "fscl-ada1": language.ada_class_factory(language.TransEmbOrigSystem, ada_stage="matching"),
    "fscl-ada2": language.ada_class_factory(language.TransEmbOrigSystem, ada_stage="unsup_tuning"),
    "fscl-ssl_ada1": language.ssl_ada_class_factory(language.TransEmbOrigSystem, ada_stage="matching"),
    "fscl-ssl_ada2": language.ssl_ada_class_factory(language.TransEmbOrigSystem, ada_stage="unsup_tuning"),
}


SYSTEM_PR = {
    "pr-ssl-linear-tune": phoneme_recognition.SSLLinearSystem,
    "pr-ssl-baseline": phoneme_recognition.SSLBaselineSystem,
    "pr-ssl-cluster": phoneme_recognition.SSLClusterSystem,
    # "pr-ssl-codebook-cluster": phoneme_recognition.SSLCodebookClusterSystem,
    "pr-ssl-baseline-tune": phoneme_recognition.SSLBaselineTuneSystem,
    "pr-ssl-cluster-tune": phoneme_recognition.SSLClusterTuneSystem,
    # "pr-fscl": phoneme_recognition.TransHeadSystem,
    # "pr-fscl-tune": phoneme_recognition.TransHeadTuneSystem,
    "pr-ssl-protonet": phoneme_recognition.SSLProtoNetSystem,
}


SYSTEM_T2U = {
    "tacot2u": t2u.TacoT2USystem,
    "fscl-t2u": t2u.TransEmbSystem,
    "fscl-t2u-orig": t2u.TransEmbOrigSystem,
    "fscl-t2u-codebook": t2u.TransEmbCSystem,
    "fscl-t2u-codebook2": t2u.TransEmbC2System,

    "fscl-t2u-tune": t2u.TransEmbTuneSystem,
    "fscl-t2u-orig-tune": t2u.TransEmbOrigTuneSystem,
    "fscl-t2u-orig-e2e-tune": t2u.TransEmbOrigE2ETuneSystem,
    "fscl-t2u-da-tune": t2u.TransEmbDATuneSystem,
    "fscl-t2u-e2e-tune": t2u.TransEmbE2ETuneSystem,
    "fscl-t2u-c-e2e-tune": t2u.TransEmbCE2ETuneSystem,
    "fscl-t2u-c2-e2e-tune": t2u.TransEmbC2E2ETuneSystem,
    "fscl-t2u-da-e2e-tune": t2u.TransEmbDAE2ETuneSystem,
    "fscl-t2u-c-da-e2e-tune": t2u.TransEmbCDAE2ETuneSystem,
    "fscl-t2u-c2-da-e2e-tune": t2u.TransEmbC2DAE2ETuneSystem,
}


SYSTEM = {
    **SYSTEM_SYNTHESIS,
    # **SYSTEM_PR,
    **SYSTEM_T2U
}


def get_system(algorithm) -> Type[System]:
    return SYSTEM[algorithm]


# Deprecated
SYSTEM_OLD = {
    # "semi-fscl": language.SemiTransEmbSystem,
    # "semi-fscl-tune": language.SemiTransEmbTuneSystem,
    "baseline": language.BaselineSystem,
    "baseline-tune": language.BaselineTuneSystem,
    # "fscl": language.TransEmbSystem,
    "fscl-ada1": language.ada_class_factory(language.TransEmbOrigSystem, ada_stage="matching"),
    "fscl-ada2": language.ada_class_factory(language.TransEmbOrigSystem, ada_stage="unsup_tuning"),
    "fscl-ssl_ada1": language.ssl_ada_class_factory(language.TransEmbOrigSystem, ada_stage="matching"),
    "fscl-ssl_ada2": language.ssl_ada_class_factory(language.TransEmbOrigSystem, ada_stage="unsup_tuning"),
    "fscl-orig": language.TransEmbOrigSystem,
    # "fscl-tune": language.TransEmbTuneSystem,
    "fscl-orig-tune": language.TransEmbOrigTuneSystem,

    # "pr-ssl-linear-tune": phoneme_recognition.SSLLinearSystem,
    # "pr-ssl-baseline": phoneme_recognition.SSLBaselineSystem,
    # "pr-ssl-cluster": phoneme_recognition.SSLClusterSystem,
    # # "pr-ssl-codebook-cluster": phoneme_recognition.SSLCodebookClusterSystem,
    # "pr-ssl-baseline-tune": phoneme_recognition.SSLBaselineTuneSystem,
    # "pr-ssl-cluster-tune": phoneme_recognition.SSLClusterTuneSystem,
    # # "pr-fscl": phoneme_recognition.TransHeadSystem,
    # # "pr-fscl-tune": phoneme_recognition.TransHeadTuneSystem,
    # "pr-ssl-protonet": phoneme_recognition.SSLProtoNetSystem,

    "tacot2u": t2u.TacoT2USystem,
    "fscl-t2u": t2u.TransEmbSystem,
    "fscl-t2u-orig": t2u.TransEmbOrigSystem,
    "fscl-t2u-codebook": t2u.TransEmbCSystem,
    "fscl-t2u-codebook2": t2u.TransEmbC2System,

    "fscl-t2u-tune": t2u.TransEmbTuneSystem,
    "fscl-t2u-orig-tune": t2u.TransEmbOrigTuneSystem,
    "fscl-t2u-orig-e2e-tune": t2u.TransEmbOrigE2ETuneSystem,
    "fscl-t2u-da-tune": t2u.TransEmbDATuneSystem,
    "fscl-t2u-e2e-tune": t2u.TransEmbE2ETuneSystem,
    "fscl-t2u-c-e2e-tune": t2u.TransEmbCE2ETuneSystem,
    "fscl-t2u-c2-e2e-tune": t2u.TransEmbC2E2ETuneSystem,
    "fscl-t2u-da-e2e-tune": t2u.TransEmbDAE2ETuneSystem,
    "fscl-t2u-c-da-e2e-tune": t2u.TransEmbCDAE2ETuneSystem,
    "fscl-t2u-c2-da-e2e-tune": t2u.TransEmbC2DAE2ETuneSystem,

    "conti-ae": language.ContiAESystem,
}