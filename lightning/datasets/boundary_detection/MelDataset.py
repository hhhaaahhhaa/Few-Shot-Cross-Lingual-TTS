import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from lightning.build import build_id2symbols
from Parsers.parser import DataParser


class MelDataset(Dataset):
    """
    Provide mel/text/duration features.
    """
    def __init__(self, filename, data_parser: DataParser, config=None):
        self.data_parser = data_parser
        self.config = config

        self.name = config["name"]
        self.unit_name = config.get("unit_name", "gt")
        self.lang_id = config["lang_id"]
        self.symbol_id = config["symbol_id"]
        self.cleaners = config["text_cleaners"]
        self.id2symbols = build_id2symbols([config])
        self.use_real_phoneme = config["use_real_phoneme"]

        self.unit_parser = self.data_parser.ssl_units[self.unit_name]
        if not self.use_real_phoneme:
            self.unit2id = {p: i for i, p in enumerate(self.id2symbols[self.symbol_id])}

        self.basename, self.speaker = self.process_meta(filename)

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        query = {
            "spk": speaker,
            "basename": basename,
        }

        duration = self.unit_parser.duration.read_from_query(query)
        mel = self.data_parser.mel.read_from_query(query)
        mel = np.transpose(mel[:, :sum(duration)])
        phonemes = self.unit_parser.phoneme.read_from_query(query)
        
        if self.use_real_phoneme:
            phonemes = f"{{{phonemes}}}"
            text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))
        else:
            text = np.array([self.unit2id[phn] for phn in phonemes.split(" ")])
       
        sample = {
            "id": basename,
            "text": text,
            "mel": mel,
            "duration": duration,
            "lang_id": self.lang_id,
            "symbol_id": self.symbol_id,
        }

        return sample

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
            return name, speaker
