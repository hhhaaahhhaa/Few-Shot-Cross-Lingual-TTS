import numpy as np
from torch.utils.data import Dataset
import json

from dlhlp_lib.utils import segment2duration

import Define
from text import text_to_sequence
from lightning.build import build_id2symbols
from Parsers.parser import DataParser


class FSCLDataset(Dataset):
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

        # Transfer learning module
        segment = self.data_parser.mfa_segment.read_from_query(query)
        if Define.UPSTREAM == "mel":
            avg_frames = self.data_parser.mfa_duration.read_from_query(query)
            mel = self.data_parser.mel.read_from_query(query)
            mel = np.transpose(mel[:, :sum(avg_frames)])
            raw_feat = mel
        else:
            raw_feat = self.data_parser.wav_trim_16000.read_from_query(query)
            avg_frames = segment2duration(segment, fp=0.02)

        sample.update({
            "n_symbols": len(self.id2symbols[self.symbol_id]),
            "raw_feat": raw_feat,
            "avg_frames": avg_frames,
        })

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
