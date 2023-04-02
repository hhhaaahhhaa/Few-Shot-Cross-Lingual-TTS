import numpy as np
from torch.utils.data import Dataset

from dlhlp_lib.utils.tool import segment2duration

import Define
from Parsers.parser import DataParser


class SSLDataset(Dataset):
    """
    Provide feature info for SSL upstream and calculate boundaries in 50Hz.
    """
    def __init__(self, filename, data_parser: DataParser, config=None):
        self.data_parser = data_parser
        self.config = config

        self.name = config["name"]
        self.unit_name = config.get("unit_name", "gt")
        self.lang_id = config["lang_id"]
        self.symbol_id = config["symbol_id"]
        # self.cleaners = config["text_cleaners"]
        # self.id2symbols = build_id2symbols([config])
        # self.use_real_phoneme = config["use_real_phoneme"]

        self.unit_parser = self.data_parser.ssl_units[self.unit_name]
        # if not self.use_real_phoneme:
        #     self.unit2id = {p: i for i, p in enumerate(self.id2symbols[self.symbol_id])}

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
       
        sample = {
            "id": basename,
            "duration": duration,
            "lang_id": self.lang_id,
            "symbol_id": self.symbol_id,
        }

        # For codebook module
        segment = self.unit_parser.segment.read_from_query(query)
        if Define.UPSTREAM == "mel":
            raw_feat = mel
            avg_frames = self.unit_parser.duration.read_from_query(query)
        else:
            raw_feat = self.data_parser.wav_trim_16000.read_from_query(query)
            avg_frames = segment2duration(segment, fp=0.02)

        sample.update({
            "raw_feat": raw_feat,
            "avg_frames": avg_frames,
        })

        pos = 0
        boundary = np.zeros(sum(avg_frames)) 
        for d in avg_frames:
            pos += d
            if pos > 0:
                boundary[pos - 1] = 1
        sample.update({
            "boundary": boundary,
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
