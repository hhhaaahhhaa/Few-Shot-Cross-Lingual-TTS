import numpy as np

from dlhlp_lib.utils.tool import expand

from Parsers.parser import DataParser
from ..language.FSCLDataset import UnitFSCLDataset


class UnitPseudoDataset(UnitFSCLDataset):
    """ Extension of UnitFSCLDataset to support sample level pseudo-label filtering. """
    def __init__(self, filename, data_parser: DataParser, config=None, spk_refer_wav=False, map2phoneme=False, threshold=0.0):
        super().__init__(filename, data_parser, config, spk_refer_wav, map2phoneme)
        self.threshold = threshold
        self.filter_sample()

    def filter_sample(self) -> None:
        if self.unit_name == "gt":
            return
        self.unit_parser.phoneme_score.read_all()
        self.unit_parser.duration.read_all()
        accept_idxs = []
        for idx in range(len(self.basename)):
            basename = self.basename[idx]
            speaker = self.speaker[idx]
            query = {
                "spk": speaker,
                "basename": basename,
            }
            scores = self.unit_parser.phoneme_score.read_from_query(query)
            duration = self.unit_parser.duration.read_from_query(query)
            confidence = scores.max(axis=1)
            confidence = expand(confidence.tolist(), duration)
            if np.mean(confidence) >= self.threshold:
                accept_idxs.append(idx)
        print("Data Ratio: ", len(accept_idxs) / len(self.basename))
        self.basename = [self.basename[idx] for idx in accept_idxs]
        self.speaker = [self.speaker[idx] for idx in accept_idxs]


class UnitPseudoScoreDataset(UnitFSCLDataset):
    """
    Extension of UnitFSCLDataset to support pseudo-label filtering by providing explicit confidence scores.
    """
    def __init__(self, filename, data_parser: DataParser, config=None, spk_refer_wav=False, map2phoneme=False):
        super().__init__(filename, data_parser, config, spk_refer_wav, map2phoneme)
        self.n_symbols = len(self.id2symbols[self.symbol_id])

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        query = {
            "spk": speaker,
            "basename": basename,
        }
        if self.unit_name == "gt":
            phoneme_score = np.ones((len(sample["text"]), self.n_symbols)) * 10
        else:
            phoneme_score = self.unit_parser.phoneme_score.read_from_query(query)
        sample.update({"phoneme_score": phoneme_score})

        return sample
