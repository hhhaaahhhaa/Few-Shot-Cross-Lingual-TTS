import numpy as np
from torch.utils.data import Dataset
import json

from dlhlp_lib.utils.tool import segment2duration
from dlhlp_lib.utils.numeric import numpy_exist_nan

import Define
from text import text_to_sequence
from lightning.build import build_id2symbols
from Parsers.parser import DataParser


class ContiAEDataset(Dataset):
    def __init__(self, filename, data_parser: DataParser, config=None, spk_refer_wav=False, map2phoneme=False):
        self.data_parser = data_parser
        self.spk_refer_wav = spk_refer_wav
        self.config = config

        self.name = config["name"]
        self.lang_id = config["lang_id"]
        self.symbol_id = config["symbol_id"]
        self.cleaners = config["text_cleaners"]
        self.id2symbols = build_id2symbols([config])

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

        duration = self.data_parser.mfa_duration.read_from_query(query)
        duration = np.array([1] * sum(duration))  # set all durations to 1
        mel = self.data_parser.mel.read_from_query(query)
        mel = np.transpose(mel[:, :sum(duration)])

        # Always use frame level prosody in ContiAESystem
        pitch = self.data_parser.interpolate_pitch.read_from_query(query)
        pitch = pitch[:sum(duration)]

        energy = self.data_parser.energy.read_from_query(query)
        energy = energy[:sum(duration)]

        phonemes = self.data_parser.phoneme.read_from_query(query)
        raw_text = self.data_parser.text.read_from_query(query)

        _, _, global_pitch_mu, global_pitch_std, _, _, global_energy_mu, global_energy_std = Define.ALLSTATS["global"]
        if self.config["pitch"]["normalization"]:
            pitch = (pitch - global_pitch_mu) / global_pitch_std
        if self.config["energy"]["normalization"]:
            energy = (energy - global_energy_mu) / global_energy_std

        text = duration  # text is not important in ContiAESystem
        
        # Sanity check
        assert not numpy_exist_nan(mel)
        assert not numpy_exist_nan(pitch)
        assert not numpy_exist_nan(energy)
        assert not numpy_exist_nan(duration)
        try:
            assert len(text) == len(duration)
            if self.config["pitch"]["feature"] == "phoneme_level":
                assert len(duration) == len(pitch)
            else:
                assert sum(duration) == len(pitch)
            if self.config["energy"]["feature"] == "phoneme_level":
                assert len(duration) == len(energy)
            else:
                assert sum(duration) == len(pitch)
        except:
            print("Length mismatch: ", query)
            print(len(text), len(phonemes), len(duration), len(pitch), len(energy))
            raise

        sample = {
            "id": basename,
            "speaker": speaker,
            "text": text,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "lang_id": self.lang_id,
            "symbol_id": self.symbol_id,
        }

        if self.spk_refer_wav:
            spk_ref_mel_slices = self.data_parser.spk_ref_mel_slices.read_from_query(query)
            sample.update({"spk_ref_mel_slices": spk_ref_mel_slices})

        # For codebook module
        segment = self.data_parser.mfa_segment.read_from_query(query)
        if Define.UPSTREAM == "mel":
            raw_feat = mel
            avg_frames = self.data_parser.mfa_duration.read_from_query(query)
        else:
            raw_feat = self.data_parser.wav_trim_16000.read_from_query(query)
            avg_frames = segment2duration(segment, fp=0.02)

        sample.update({
            "n_symbols": len(self.id2symbols[self.symbol_id]),
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
