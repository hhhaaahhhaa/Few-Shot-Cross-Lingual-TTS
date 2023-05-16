from . import language
from . import phoneme_recognition
from . import t2u
from . import boundary_detection
from . import semi


DATA_MODULE = {
    "baseline": language.FastSpeech2DataModule,
    "fscl": language.FSCLDataModule,
    "fscl-orig": language.FSCLDataModule,
    "fscl-linear": language.FSCLDataModule,
    "fscl-transformer": language.FSCLDataModule,
    "fscl-orig-seg": language.UnitDataModule,
    "fscl-linear-seg": language.UnitDataModule,
    "fscl-transformer-seg": language.UnitDataModule,
    "fscl-ada1": language.FSCLDataModule,
    "fscl-ada2": language.FSCLDataModule,
    "fscl-ssl_ada1": language.FSCLDataModule,
    "fscl-ssl_ada2": language.FSCLDataModule,
    
    "fscl-tune": language.FastSpeech2DataModule,
    "fscl-orig-tune": language.FastSpeech2DataModule,
    "fscl-linear-tune": language.FastSpeech2DataModule,
    "fscl-transformer-tune": language.FastSpeech2DataModule,

    "dual-orig": language.UnitDataModule,
    "dual-transformer": language.UnitDataModule,
    "dual-orig-tune": language.UnitDataModule,

    "pr-ssl-linear": phoneme_recognition.SSLPRDataModule,
    "pr-ssl-baseline": phoneme_recognition.SSLPRDataModule,
    "pr-ssl-cluster": phoneme_recognition.SSLPRDataModule,
    # "pr-ssl-codebook-cluster": phoneme_recognition.SSLPRDataModule,
    "pr-ssl-baseline-tune": phoneme_recognition.SSLPRDataModule,
    "pr-ssl-cluster-tune": phoneme_recognition.SSLPRDataModule,
    # "pr-fscl": phoneme_recognition.FSCLDataModule,
    # "pr-fscl-tune": phoneme_recognition.SSLPRDataModule,
    "pr-ssl-protonet": phoneme_recognition.FSCLDataModule,

    "tacot2u": t2u.T2UDataModule,
    "fscl-t2u": t2u.FSCLDataModule,
    "fscl-t2u-orig": t2u.FSCLDataModule,
    "fscl-t2u-codebook": t2u.FSCLDataModule,
    "fscl-t2u-codebook2": t2u.FSCLDataModule,
    
    "fscl-t2u-tune": t2u.T2UDataModule,
    "fscl-t2u-orig-tune": t2u.T2UDataModule,
    "fscl-t2u-orig-e2e-tune": t2u.T2U2SDataModule,
    "fscl-t2u-da-tune": t2u.T2UDADataModule,
    "fscl-t2u-e2e-tune": t2u.T2U2SDataModule,
    "fscl-t2u-c-e2e-tune": t2u.T2U2SDataModule,
    "fscl-t2u-c2-e2e-tune": t2u.T2U2SDataModule,
    "fscl-t2u-da-e2e-tune": t2u.T2U2SDADataModule,
    "fscl-t2u-c-da-e2e-tune": t2u.T2U2SDADataModule,
    "fscl-t2u-c2-da-e2e-tune": t2u.T2U2SDADataModule,

    "conti-ae": language.ContiAEDataModule,
    "bd-ssl-conv": boundary_detection.SSLDataModule,
    "bd-ssl-conv-tune": boundary_detection.SSLDataModule,
    "bd-segfeat": boundary_detection.MelDataModule,
    "bd-segfeat-tune": boundary_detection.MelDataModule,

    "semi-baseline": language.FastSpeech2DataModule,
    "semi": language.UnitDataModule,
    "semi-fscl": language.FSCLDataModule,
    "semi-baseline-tune": semi.FastSpeech2DataModule,
    "semi-tune": semi.UnitDataModule,
    "semi-fscl-tune": semi.UnitDataModule,
}

def get_datamodule(algorithm):
    return DATA_MODULE[algorithm]
