import numpy as np
from torch.utils.data import Dataset
import json

from text import text_to_sequence
from lightning.build import build_id2symbols
from Parsers.parser import DataParser


class DADataset(Dataset):
    """
    DA unit dataset, return unit sequence and lang_id.
    """
    def __init__(self, filename, data_parser: DataParser, config):
        self.data_parser = data_parser

        self.name = config["name"]
        self.lang_id = config["lang_id"]
        self.symbol_id = config["symbol_id"]
        self.unit_name = config["unit_name"]
        self.cleaners = config["text_cleaners"]

        self.unit_parser = self.data_parser.ssl_units[self.unit_name]

        self.data_parser = data_parser
        self.config = config
        self.id2symbols = build_id2symbols([config])

        self.unit2id = {p: i for i, p in enumerate(self.id2symbols[self.unit_name])}

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

        unit = self.unit_parser.phoneme.read_from_query(query)
        unit = np.array([self.unit2id[phn] for phn in unit.split(" ")])

        sample = {
            "id": basename,
            "speaker": speaker,
            "unit": unit,
            "lang_id": self.lang_id,
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
