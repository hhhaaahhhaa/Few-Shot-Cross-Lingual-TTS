import numpy as np
from torch.utils.data import Dataset
import json
import math

from dlhlp_lib.utils.tool import segment2duration
from dlhlp_lib.utils.numeric import numpy_exist_nan

import Define
from text import text_to_sequence
from lightning.build import build_id2symbols
from Parsers.parser import DataParser


class FSCLDataset(Dataset):
    """
    Extension of FastSpeech2Dataset, provide raw speech representations.
    """
    def __init__(self, filename, data_parser: DataParser, config=None, spk_refer_wav=False):
        self.data_parser = data_parser
        self.spk_refer_wav = spk_refer_wav
        self.config = config

        self.name = config["name"]
        self.lang_id = config["lang_id"]
        self.symbol_id = config["symbol_id"]
        self.cleaners = config["text_cleaners"]
        self.id2symbols = build_id2symbols([config])

        self.basename, self.speaker = self.process_meta(filename)
        with open(self.data_parser.speakers_path, 'r', encoding='utf-8') as f:
            self.speakers = json.load(f)
            self.speaker_map = {spk: i for i, spk in enumerate(self.speakers)}

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
        mel = self.data_parser.mel.read_from_query(query)
        mel = np.transpose(mel[:, :sum(duration)]) * math.log(10)
        if self.config["pitch"]["feature"] == "phoneme_level":
            pitch = self.data_parser.mfa_duration_avg_pitch.read_from_query(query)
        else:
            pitch = self.data_parser.interpolate_pitch.read_from_query(query)
            pitch = pitch[:sum(duration)]
        if self.config["energy"]["feature"] == "phoneme_level":
            energy = self.data_parser.mfa_duration_avg_energy.read_from_query(query)
        else:
            energy = self.data_parser.energy.read_from_query(query)
            energy = energy[:sum(duration)]
        phonemes = self.data_parser.phoneme.read_from_query(query)
        phonemes = f"{{{phonemes}}}"
        raw_text = self.data_parser.text.read_from_query(query)

        _, _, global_pitch_mu, global_pitch_std, _, _, global_energy_mu, global_energy_std = Define.ALLSTATS["global"]
        if self.config["pitch"]["normalization"]:
            pitch = (pitch - global_pitch_mu) / global_pitch_std
        if self.config["energy"]["normalization"]:
            energy = (energy - global_energy_mu) / global_energy_std
        text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))
        
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

        # Addtional speech representations
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


class UnsupFSCLDataset(Dataset):
    """
    Unsupervised version of FSCLDataset.
    """
    def __init__(self, filename, data_parser: DataParser, config=None, spk_refer_wav=False):
        self.oracle = False  # Assume oracle perfect segmentation, will soon be replaced by FSCLDataset.
        self.data_parser = data_parser
        self.spk_refer_wav = spk_refer_wav

        self.name = config["name"]

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
        if not self.oracle:
            pitch = self.data_parser.unsup_duration_avg_pitch.read_from_query(query)
            energy = self.data_parser.unsup_duration_avg_energy.read_from_query(query)
            duration = self.data_parser.unsup_duration.read_from_query(query)
        else:  # Oracle perfect segmentation
            pitch = self.data_parser.mfa_duration_avg_pitch.read_from_query(query)
            energy = self.data_parser.mfa_duration_avg_energy.read_from_query(query)
            duration = self.data_parser.mfa_duration.read_from_query(query)

        mel = np.transpose(mel[:, :sum(duration)])

        _, _, global_pitch_mu, global_pitch_std, _, _, global_energy_mu, global_energy_std = Define.ALLSTATS["global"]
        pitch = (pitch - global_pitch_mu) / global_pitch_std  # normalize
        energy = (energy - global_energy_mu) / global_energy_std  # normalize
        
        assert not numpy_exist_nan(mel)
        assert not numpy_exist_nan(pitch)
        assert not numpy_exist_nan(energy)
        assert not numpy_exist_nan(duration)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": None,
            "raw_text": None,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        if self.spk_refer_wav:
            spk_ref_mel_slices = self.data_parser.spk_ref_mel_slices.read_from_query(query)
            sample.update({"spk_ref_mel_slices": spk_ref_mel_slices})

        # For codebook module
        if not self.oracle:
            segment = self.data_parser.unsup_segment.read_from_query(query)
        else:
            segment = self.data_parser.mfa_segment.read_from_query(query)
        if Define.UPSTREAM == "mel":
            raw_feat = mel
            if not self.oracle:
                avg_frames = self.data_parser.unsup_duration.read_from_query(query)
            else:
                avg_frames = self.data_parser.mfa_duration.read_from_query(query)
        else:
            raw_feat = self.data_parser.wav_trim_16000.read_from_query(query)
            avg_frames = segment2duration(segment, fp=0.02)

        sample.update({
            "lang_id": None,
            "n_symbols": -1,
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


class UnitFSCLDataset(Dataset):
    """
    Unit version of FSCLDataset. This can be an semi-supervised/unsupervised method, depending on
    whether unit is able to mapped to real phoneme or not, e.g. hubert unit is pseudo unit, any pseudo label
    method use real phonemes.
    """
    def __init__(self, filename, data_parser: DataParser, config=None, spk_refer_wav=False, map2phoneme=False):
        self.data_parser = data_parser
        self.spk_refer_wav = spk_refer_wav
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
        mel = np.transpose(mel[:, :sum(duration)]) * math.log(10)
        if self.config["pitch"]["feature"] == "phoneme_level":
            pitch = self.unit_parser.duration_avg_pitch.read_from_query(query)
        else:
            pitch = self.data_parser.interpolate_pitch.read_from_query(query)
            pitch = pitch[:sum(duration)]
        if self.config["energy"]["feature"] == "phoneme_level":
            energy = self.unit_parser.duration_avg_energy.read_from_query(query)
        else:
            energy = self.data_parser.energy.read_from_query(query)
            energy = energy[:sum(duration)]
        phonemes = self.unit_parser.phoneme.read_from_query(query)
        raw_text = self.data_parser.text.read_from_query(query)

        _, _, global_pitch_mu, global_pitch_std, _, _, global_energy_mu, global_energy_std = Define.ALLSTATS["global"]
        if self.config["pitch"]["normalization"]:
            pitch = (pitch - global_pitch_mu) / global_pitch_std
        if self.config["energy"]["normalization"]:
            energy = (energy - global_energy_mu) / global_energy_std
        if self.use_real_phoneme:
            phonemes = f"{{{phonemes}}}"
            text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))
        else:
            text = np.array([self.unit2id[phn] for phn in phonemes.split(" ")])
        
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
            pp = self.unit_parser.phoneme.read_from_query(query)
            print(pp)
            print(len(pp.strip().split(" ")))
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
        segment = self.unit_parser.segment.read_from_query(query)
        if Define.UPSTREAM == "mel":
            raw_feat = mel
            avg_frames = self.unit_parser.duration.read_from_query(query)
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
