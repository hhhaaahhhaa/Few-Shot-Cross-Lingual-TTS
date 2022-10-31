import numpy as np
from torch.utils.data import Dataset
import json

from dlhlp_lib.utils.tool import segment2duration

from text import text_to_sequence
from text.define import LANG_ID2SYMBOLS
from Parsers.parser import DataParser
from lightning.utils.tool import numpy_exist_nan


class MelPRDataset(Dataset):
    """
    Phoneme recognition dataset, use mel as raw speech representations.
    """
    def __init__(self, filename, data_parser: DataParser, config=None):
        self.data_parser = data_parser

        self.name = config["name"]
        self.lang_id = config["lang_id"]
        self.cleaners = config["text_cleaners"]

        self.basename, self.speaker = self.process_meta(filename)
        with open(self.data_parser.speakers_path, 'r', encoding='utf-8') as f:
            self.speakers = json.load(f)
            self.speaker_map = {spk: i for i, spk in enumerate(self.speakers)}

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        query = {
            "spk": speaker,
            "basename": basename,
        }

        mel = self.data_parser.mel.read_from_query(query)
        duration = self.data_parser.mfa_duration.read_from_query(query)
        phonemes = self.data_parser.phoneme.read_from_query(query)
        raw_text = self.data_parser.text.read_from_query(query)
        mel = np.transpose(mel[:, :sum(duration)])
        phonemes = f"{{{phonemes}}}"

        text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))
        
        assert not numpy_exist_nan(mel)
        assert not numpy_exist_nan(duration)
        try:
            assert len(text) == len(duration)
        except:
            print(query)
            print(text)
            print(len(text), len(phonemes), len(duration))
            raise

        expanded_text = np.repeat(text, duration)
        raw_feat = mel
        avg_frames = duration

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": expanded_text,
            "raw_text": raw_text,
            "mel": mel,
            "duration": duration,
            "lang_id": self.lang_id,
            "n_symbols": len(LANG_ID2SYMBOLS[self.lang_id]),
            "raw-feat": raw_feat,
            "avg-frames": avg_frames,
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


class SSLPRDataset(Dataset):
    """
    Phoneme recognition dataset, use wav as raw speech representations and designed for SSL upstreams.
    """
    def __init__(self, filename, data_parser: DataParser, config=None):
        self.data_parser = data_parser

        self.name = config["name"]
        self.lang_id = config["lang_id"]
        self.cleaners = config["text_cleaners"]

        self.basename, self.speaker = self.process_meta(filename)
        with open(self.data_parser.speakers_path, 'r', encoding='utf-8') as f:
            self.speakers = json.load(f)
            self.speaker_map = {spk: i for i, spk in enumerate(self.speakers)}

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        query = {
            "spk": speaker,
            "basename": basename,
        }

        segment = self.data_parser.mfa_segment.read_from_query(query)
        avg_frames = segment2duration(segment, fp=0.02)
        phonemes = self.data_parser.phoneme.read_from_query(query)
        raw_text = self.data_parser.text.read_from_query(query)
        phonemes = f"{{{phonemes}}}"

        text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))
        duration = np.array(avg_frames)
        
        assert not numpy_exist_nan(duration)
        try:
            assert len(text) == len(duration)
        except:
            print(query)
            print(text)
            print(len(text), len(phonemes), len(duration))
            raise

        expanded_text = np.repeat(text, duration)
        raw_feat = self.data_parser.wav_trim_16000.read_from_query(query)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": text,
            "expanded_text": expanded_text,
            "raw_text": raw_text,
            "wav": raw_feat,
            "duration": duration,
            "lang_id": self.lang_id,
            "n_symbols": len(LANG_ID2SYMBOLS[self.lang_id]),
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
