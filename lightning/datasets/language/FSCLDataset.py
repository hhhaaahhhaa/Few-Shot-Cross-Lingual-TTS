import numpy as np
from torch.utils.data import Dataset
import json

import Define
from text import text_to_sequence
from Parsers.parser import DataParser
from lightning.utils.tool import numpy_exist_nan


class FSCLDataset(Dataset):
    """
    Extension of FastSpeech2Dataset, provide speech representations.
    """
    def __init__(self, filename, data_parser: DataParser, config=None, sort=False, drop_last=False, spk_refer_wav=False):
        self.data_parser = data_parser
        self.spk_refer_wav = spk_refer_wav
        self.sort = sort
        self.drop_last = drop_last

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
        pitch = self.data_parser.mfa_duration_avg_pitch.read_from_query(query)
        energy = self.data_parser.mfa_duration_avg_energy.read_from_query(query)
        duration = self.data_parser.mfa_duration.read_from_query(query)
        phonemes = self.data_parser.phoneme.read_from_query(query)
        raw_text = self.data_parser.text.read_from_query(query)
        mel = np.transpose(mel[:, :sum(duration)])
        phonemes = f"{{{phonemes}}}"

        _, _, global_pitch_mu, global_pitch_std, _, _, global_energy_mu, global_energy_std = Define.ALLSTATS["global"]
        pitch = (pitch - global_pitch_mu) / global_pitch_std  # normalize
        energy = (energy - global_energy_mu) / global_energy_std  # normalize
        text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))
        
        assert not numpy_exist_nan(mel)
        assert not numpy_exist_nan(pitch)
        assert not numpy_exist_nan(energy)
        assert not numpy_exist_nan(duration)
        try:
            assert len(text) == len(duration) == len(pitch) == len(energy)
        except:
            print(query)
            print(text)
            print(len(text), len(phonemes), len(duration), len(pitch), len(energy))
            raise

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": text,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
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
            avg_frames = []
            for (s, e) in segment:
                avg_frames.append(
                    int(
                        np.round(e * 50)  # All ssl model use 20ms window
                        - np.round(s * 50)
                    )
                )

        sample.update({
            "lang_id": self.lang_id,
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
    def __init__(self, filename, data_parser: DataParser, config=None, sort=False, drop_last=False, spk_refer_wav=False):
        self.data_parser = data_parser
        self.spk_refer_wav = spk_refer_wav
        self.sort = sort
        self.drop_last = drop_last

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
        pitch = self.data_parser.unsup_duration_avg_pitch.read_from_query(query)
        energy = self.data_parser.unsup_duration_avg_energy.read_from_query(query)
        duration = self.data_parser.unsup_duration.read_from_query(query)
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
        segment = self.data_parser.unsup_segment.read_from_query(query)
        if Define.UPSTREAM == "mel":
            raw_feat = mel
            avg_frames = self.data_parser.unsup_duration.read_from_query(query)
        else:
            raw_feat = self.data_parser.wav_trim_16000.read_from_query(query)
            avg_frames = []
            for (s, e) in segment:
                avg_frames.append(
                    int(
                        np.round(e * 50)  # All ssl model use 20ms window
                        - np.round(s * 50)
                    )
                )

        sample.update({
            "lang_id": None,
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
