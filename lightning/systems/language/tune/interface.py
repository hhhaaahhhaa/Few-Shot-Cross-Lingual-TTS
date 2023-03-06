from ..FastSpeech2 import BaselineSystem


class IFastSpeech2TuneSystem(BaselineSystem):
    """
    Base class for FastSpeech2 tuning
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tune_init(self, data_configs) -> None:
        raise NotImplementedError
