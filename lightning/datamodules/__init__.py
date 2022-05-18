from . import language


# Old
# DATA_MODULE = {
#     "base": BaseDataModule,
#     "meta": MetaDataModule,
#     "imaml": MetaDataModule,
#     "asr-codebook": MetaDataModule,
#     "asr-baseline": BaselineV2DataModule,
#     "asr-center": BaselineV2DataModule,
#     "asr-center-ref": MetaDataModule,
#     "dual-meta": MetaDataModule,
#     "dual-tune": BaselineDataModule,
#     "meta-tune": BaselineDataModule,
#     "baseline": BaselineDataModule,
# }

DATA_MODULE = {
    "fscl": language.FSCLDataModule,
    "fscl-tune": language.FastSpeech2DataModule,
    "semi-fscl": language.SemiFSCLDataModule,
    "multilingual-baseline": language.FastSpeech2DataModule,
    "multilingual-baseline-tune": language.FastSpeech2DataModule,
}

def get_datamodule(algorithm):
    return DATA_MODULE[algorithm]
