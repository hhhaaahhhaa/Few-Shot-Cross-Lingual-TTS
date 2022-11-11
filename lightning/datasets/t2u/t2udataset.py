import numpy as np
from torch.utils.data import Dataset
import json

from dlhlp_lib.utils.tool import segment2duration
from dlhlp_lib.utils.numeric import numpy_exist_nan

from text import text_to_sequence
from text.define import LANG_ID2SYMBOLS
from lightning.build import build_id2symbols
from Parsers.parser import DataParser


class T2UDataset(Dataset):
    """
    Text-to-unit phoneme recognition dataset, actually support any unit-to-unit since text is also a unit.
    """
    def __init__(self, filename, data_parser: DataParser, config):
        self.data_parser = data_parser

        self.name = config["name"]
        self.lang_id = config["lang_id"]
        self.symbol_id = config["symbol_id"]
        self.cleaners = config["text_cleaners"]

        self.target_unit_name = config["target"]["unit_name"]
        self.target_symbol_id = config["target"]["symbol_id"]
        self.unit_parser = self.data_parser.ssl_units[self.target_unit_name]

        self.data_parser = data_parser
        self.config = config
        self.id2symbols = build_id2symbols([config])

        self.unit2id = {p: i for i, p in enumerate(self.id2symbols[self.target_unit_name])}
        self.text2id = {"@" + p: i for i, p in enumerate(self.id2symbols[self.lang_id])}

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

        phonemes = self.data_parser.phoneme.read_from_query(query)
        phonemes = f"{{{phonemes}}}"
        text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))
        text = np.append(text, 8)  # append <eos>

        unit = self.unit_parser.phoneme.read_from_query(query)
        unit = np.array([self.unit2id[phn] for phn in unit.split(" ")])
        unit = np.append(unit, 8)  # append <eos>

        raw_text = self.data_parser.text.read_from_query(query)

        sample = {
            "id": basename,
            "speaker": speaker,
            "text": text,
            "raw_text": raw_text,
            "unit": unit,
            "lang_id": self.lang_id,
            "symbol_id": self.symbol_id,
            "target_symbol_id": self.target_symbol_id,
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
