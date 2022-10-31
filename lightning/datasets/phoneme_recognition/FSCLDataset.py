import numpy as np
from torch.utils.data import Dataset
import json
import pickle

from dlhlp_lib.utils.tool import segment2duration

from text import text_to_sequence
from text.define import LANG_ID2SYMBOLS
from Parsers.parser import DataParser
from lightning.utils.tool import numpy_exist_nan


class FSCLDataset(Dataset):
    """
    Extension of SSLPRDataset, provide raw speech representations.
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

        sample.update({
            # "raw-feat": raw_feat,
            "avg-frames": avg_frames,
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


class SSLUnitFSCLDataset(Dataset):
    """
    SSL Unit version of FSCLDataset, however, this can be an semi-supervised/unsupervised method.
    """
    def __init__(self, filename, data_parser: DataParser, config=None, spk_refer_wav=False, map2phoneme=False):
        self.data_parser = data_parser
        self.spk_refer_wav = spk_refer_wav
        self.map2phoneme = map2phoneme

        self.name = config["name"]
        self.lang_id = config["lang_id"]
        self.unit_name = config["unit_name"]
        self.unit_parser = self.data_parser.ssl_units[self.unit_name]

        try:
            with open(f"{self.unit_parser.root}/centroids.pkl", "rb") as f:
                kmeans_model = pickle.load(f)
                self.n_clusters = kmeans_model.cluster_centers_.shape[0]
        except:
            self.n_clusters = 0
        
        if self.map2phoneme:
            with open(f"{self.unit_parser.root}/centroids2phoneme.pkl", "rb") as f:
                pairs = pickle.load(f)
            self.unit2phoneme = {idx: phn for (idx, phn) in pairs}
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

        segment = self.unit_parser.dp_segment.read_from_query(query)
        avg_frames = segment2duration(segment, fp=0.02)
        phonemes = self.unit_parser.phoneme.read_from_query(query)
        raw_text = self.data_parser.text.read_from_query(query)
        
        if self.map2phoneme:
            phonemes = " ".join([self.unit2phoneme[phn] for phn in phonemes.split(" ")])
            phonemes = f"{{{phonemes}}}"
            text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))
        else:
            text = np.array([int(phn) for phn in phonemes.split(" ")])
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

        sample.update({
            "raw-feat": raw_feat,
            "avg-frames": avg_frames,
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


class SSLUnitPseudoLabelDataset(SSLUnitFSCLDataset):
    """
    SSLUnitFSCLDataset, but units are matched to real phonemes (pseudo labels).
    """
    def __init__(self, filename, data_parser: DataParser, config=None, spk_refer_wav=False):
        super().__init__(filename, data_parser, config, spk_refer_wav, map2phoneme=True)
