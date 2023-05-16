from typing import Type

from .system import System
from .adaptor import AdaptorSystem
from . import language
from . import phoneme_recognition
from . import t2u
from . import boundary_detection
from . import semi


SYSTEM_SYNTHESIS = {
    "baseline": language.BaselineSystem,
    "baseline-tune": language.BaselineTuneSystem,
    "conti-ae": language.ContiAESystem,

    "fscl-orig": language.fscl_fastspeech2_class_factory("orig"),
    "fscl-linear": language.fscl_fastspeech2_class_factory("linear"),
    "fscl-transformer": language.fscl_fastspeech2_class_factory("transformer"),
    "fscl-orig-tune": language.fscl_tune_fastspeech2_class_factory("orig"),
    "fscl-linear-tune": language.fscl_tune_fastspeech2_class_factory("linear"),
    "fscl-transformer-tune": language.fscl_tune_fastspeech2_class_factory("transformer"),

    "fscl-orig-seg": language.seg_fastspeech2_class_factory("orig"),
    "fscl-linear-seg": language.seg_fastspeech2_class_factory("linear"),
    "fscl-transformer-seg": language.seg_fastspeech2_class_factory("transformer"),

    "fscl-ada1": language.ada_class_factory(language.fscl_fastspeech2_class_factory("orig"), ada_stage="matching"),
    "fscl-ada2": language.ada_class_factory(language.fscl_fastspeech2_class_factory("orig"), ada_stage="unsup_tuning"),
    "fscl-ssl_ada1": language.ssl_ada_class_factory(language.fscl_fastspeech2_class_factory("orig"), ada_stage="matching"),
    "fscl-ssl_ada2": language.ssl_ada_class_factory(language.fscl_fastspeech2_class_factory("orig"), ada_stage="unsup_tuning"),
}


SYSTEM_DUAL = {
    "dual-orig": language.dual_fastspeech2_class_factory("orig"),
    "dual-orig-tune": language.dual_tune_fastspeech2_class_factory("orig"),
    "dual-transformer": language.dual_fastspeech2_class_factory("transformer"),
}


SYSTEM_PR = {
    "pr-ssl-linear": phoneme_recognition.SSLLinearSystem,
    # "pr-ssl-baseline": phoneme_recognition.SSLBaselineSystem,
    # "pr-ssl-cluster": phoneme_recognition.SSLClusterSystem,
    # "pr-ssl-codebook-cluster": phoneme_recognition.SSLCodebookClusterSystem,
    # "pr-ssl-baseline-tune": phoneme_recognition.SSLBaselineTuneSystem,
    # "pr-ssl-cluster-tune": phoneme_recognition.SSLClusterTuneSystem,
    # "pr-fscl": phoneme_recognition.TransHeadSystem,
    # "pr-fscl-tune": phoneme_recognition.TransHeadTuneSystem,
    # "pr-ssl-protonet": phoneme_recognition.SSLProtoNetSystem,
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


SYSTEM_BD = {
    "bd-ssl-conv": boundary_detection.SSLConvSystem,
    "bd-ssl-conv-tune": boundary_detection.SSLConvTuneSystem,
    "bd-segfeat": boundary_detection.SegFeatSystem,
    "bd-segfeat-tune": boundary_detection.SegFeatTuneSystem,
}


SYSTEM_SEMI = {
    "semi-baseline": semi.BaselineSystem,
    "semi": semi.SemiSystem,
    "semi-fscl": semi.SemiTransEmbSystem,

    "semi-baseline-tune": semi.BaselineTuneSystem,
    "semi-tune": semi.SemiTuneSystem,
    "semi-fscl-tune": semi.SemiTransEmbTuneSystem,
}


SYSTEM = {
    **SYSTEM_SYNTHESIS,
    **SYSTEM_DUAL,
    **SYSTEM_PR,
    **SYSTEM_T2U,
    **SYSTEM_BD,
    **SYSTEM_SEMI,
}


def get_system(algorithm) -> Type[System]:
    return SYSTEM[algorithm]


# Deprecated
# SYSTEM_OLD = {
#     "semi-fscl": language.SemiTransEmbSystem,
#     "semi-fscl-tune": language.SemiTransEmbTuneSystem,
#     "baseline": language.BaselineSystem,
#     "baseline-tune": language.BaselineTuneSystem,
#     "fscl": language.TransEmbSystem,
#     "fscl-ada1": language.ada_class_factory(language.TransEmbOrigSystem, ada_stage="matching"),
#     "fscl-ada2": language.ada_class_factory(language.TransEmbOrigSystem, ada_stage="unsup_tuning"),
#     "fscl-ssl_ada1": language.ssl_ada_class_factory(language.TransEmbOrigSystem, ada_stage="matching"),
#     "fscl-ssl_ada2": language.ssl_ada_class_factory(language.TransEmbOrigSystem, ada_stage="unsup_tuning"),
#     "fscl-orig": language.TransEmbOrigSystem,
#     "fscl-tune": language.TransEmbTuneSystem,
#     "fscl-orig-tune": language.TransEmbOrigTuneSystem,

#     "pr-ssl-linear": phoneme_recognition.SSLLinearSystem,
#     "pr-ssl-baseline": phoneme_recognition.SSLBaselineSystem,
#     "pr-ssl-cluster": phoneme_recognition.SSLClusterSystem,
#     # "pr-ssl-codebook-cluster": phoneme_recognition.SSLCodebookClusterSystem,
#     "pr-ssl-baseline-tune": phoneme_recognition.SSLBaselineTuneSystem,
#     "pr-ssl-cluster-tune": phoneme_recognition.SSLClusterTuneSystem,
#     # "pr-fscl": phoneme_recognition.TransHeadSystem,
#     # "pr-fscl-tune": phoneme_recognition.TransHeadTuneSystem,
#     "pr-ssl-protonet": phoneme_recognition.SSLProtoNetSystem,

#     "tacot2u": t2u.TacoT2USystem,
#     "fscl-t2u": t2u.TransEmbSystem,
#     "fscl-t2u-orig": t2u.TransEmbOrigSystem,
#     "fscl-t2u-codebook": t2u.TransEmbCSystem,
#     "fscl-t2u-codebook2": t2u.TransEmbC2System,

#     "fscl-t2u-tune": t2u.TransEmbTuneSystem,
#     "fscl-t2u-orig-tune": t2u.TransEmbOrigTuneSystem,
#     "fscl-t2u-orig-e2e-tune": t2u.TransEmbOrigE2ETuneSystem,
#     "fscl-t2u-da-tune": t2u.TransEmbDATuneSystem,
#     "fscl-t2u-e2e-tune": t2u.TransEmbE2ETuneSystem,
#     "fscl-t2u-c-e2e-tune": t2u.TransEmbCE2ETuneSystem,
#     "fscl-t2u-c2-e2e-tune": t2u.TransEmbC2E2ETuneSystem,
#     "fscl-t2u-da-e2e-tune": t2u.TransEmbDAE2ETuneSystem,
#     "fscl-t2u-c-da-e2e-tune": t2u.TransEmbCDAE2ETuneSystem,
#     "fscl-t2u-c2-da-e2e-tune": t2u.TransEmbC2DAE2ETuneSystem,

#     "conti-ae": language.ContiAESystem,
# }