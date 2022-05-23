import os
import glob
import json

from dlhlp_lib.parsers.Feature import Feature
from dlhlp_lib.parsers.QueryParsers import SFQueryParser, NestSFQueryParser
from dlhlp_lib.parsers.IOObjects import NumpyIO, PickleIO, WavIO, TextGridIO, TextIO


class DataParser(object):
    def __init__(self, root):
        self.root = root
        self.__init_structure()
        self.queries = None

        self.wav_16000 = Feature(
            "wav_16000", root, SFQueryParser(f"{self.root}/wav_16000"), WavIO(sr=16000))
        self.wav_22050 = Feature(
            "wav_22050", root, SFQueryParser(f"{self.root}/wav_22050"), WavIO(sr=22050))
        self.mel = Feature(
            "mel", root, SFQueryParser(f"{self.root}/mel"), NumpyIO())
        self.pitch = Feature(
            "pitch", root, SFQueryParser(f"{self.root}/pitch"), NumpyIO(), enable_cache=True)
        self.interpolate_pitch = Feature(
            "interpolate_pitch", root, SFQueryParser(f"{self.root}/interpolate_pitch"), NumpyIO(), enable_cache=True)
        self.energy = Feature(
            "energy", root, SFQueryParser(f"{self.root}/energy"), NumpyIO(), enable_cache=True)
        self.mfa_duration_avg_pitch = Feature(
            "mfa_duration_avg_pitch", root, SFQueryParser(f"{self.root}/mfa_duration_avg_pitch"), NumpyIO(), enable_cache=True)
        self.unsup_duration_avg_pitch = Feature(
            "unsup_duration_avg_pitch", root, SFQueryParser(f"{self.root}/unsup_duration_avg_pitch"), NumpyIO(), enable_cache=True)
        self.mfa_duration_avg_energy = Feature(
            "mfa_duration_avg_energy", root, SFQueryParser(f"{self.root}/mfa_duration_avg_energy"), NumpyIO(), enable_cache=True)
        self.unsup_duration_avg_energy = Feature(
            "unsup_duration_avg_energy", root, SFQueryParser(f"{self.root}/unsup_duration_avg_energy"), NumpyIO(), enable_cache=True)
        self.wav_trim_22050 = Feature(
            "wav_trim_22050", root, SFQueryParser(f"{self.root}/wav_trim_22050"), NumpyIO())
        self.wav_trim_16000 = Feature(
            "wav_trim_16000", root, SFQueryParser(f"{self.root}/wav_trim_16000"), NumpyIO())
        self.unsup_segment = Feature(
            "unsup_segment", root, SFQueryParser(f"{self.root}/unsup_segment"), PickleIO(), enable_cache=True)
        self.mfa_segment = Feature(
            "mfa_segment", root, SFQueryParser(f"{self.root}/mfa_segment"), PickleIO(), enable_cache=True)
        self.textgrid = Feature(
            "TextGrid", root, NestSFQueryParser(f"{self.root}/TextGrid"), TextGridIO())
        self.phoneme = Feature(
            "phoneme", root, SFQueryParser(f"{self.root}/phoneme"), TextIO(), enable_cache=True)
        self.text = Feature(
            "text", root, SFQueryParser(f"{self.root}/text"), TextIO(), enable_cache=True)
        self.spk_ref_mel_slices = Feature(
            "spk_ref_mel_slices", root, SFQueryParser(f"{self.root}/spk_ref_mel_slices"), NumpyIO())
        self.mfa_duration = Feature(
            "mfa_duration", root, SFQueryParser(f"{self.root}/mfa_duration"), NumpyIO(), enable_cache=True)
        self.unsup_duration = Feature(
            "unsup_duration", root, SFQueryParser(f"{self.root}/unsup_duration"), NumpyIO(), enable_cache=True)
        self.mfa_ssl_duration = Feature(
            "mfa_ssl_duration", root, SFQueryParser(f"{self.root}/mfa_ssl_duration"), NumpyIO(), enable_cache=True)
        self.unsup_ssl_duration = Feature(
            "unsup_ssl_duration", root, SFQueryParser(f"{self.root}/unsup_ssl_duration"), NumpyIO(), enable_cache=True)

        self.stats_path = f"{self.root}/stats.json"
        self.speakers_path = f"{self.root}/speakers.json"

    def __init_structure(self):
        os.makedirs(f"{self.root}/wav_16000", exist_ok=True)
        os.makedirs(f"{self.root}/wav_22050", exist_ok=True)
        os.makedirs(f"{self.root}/mel", exist_ok=True)
        os.makedirs(f"{self.root}/pitch", exist_ok=True)
        os.makedirs(f"{self.root}/interpolate_pitch", exist_ok=True)
        os.makedirs(f"{self.root}/energy", exist_ok=True)
        os.makedirs(f"{self.root}/wav_trim_22050", exist_ok=True)
        os.makedirs(f"{self.root}/wav_trim_16000", exist_ok=True)
        os.makedirs(f"{self.root}/unsup_segment", exist_ok=True)
        os.makedirs(f"{self.root}/mfa_segment", exist_ok=True)
        os.makedirs(f"{self.root}/phoneme", exist_ok=True)
        os.makedirs(f"{self.root}/text", exist_ok=True)
        os.makedirs(f"{self.root}/spk_ref_mel_slices", exist_ok=True)
        os.makedirs(f"{self.root}/mfa_duration", exist_ok=True)
        os.makedirs(f"{self.root}/unsup_duration", exist_ok=True)
        os.makedirs(f"{self.root}/mfa_ssl_duration", exist_ok=True)
        os.makedirs(f"{self.root}/unsup_ssl_duration", exist_ok=True)
        os.makedirs(f"{self.root}/mfa_duration_avg_pitch", exist_ok=True)
        os.makedirs(f"{self.root}/unsup_duration_avg_pitch", exist_ok=True)
        os.makedirs(f"{self.root}/mfa_duration_avg_energy", exist_ok=True)
        os.makedirs(f"{self.root}/unsup_duration_avg_energy", exist_ok=True)
    
    def get_all_queries(self):
        with open(f"{self.root}/data_info.json", "r", encoding="utf-8") as f:
            data_infos = json.load(f)
        res = []
        for data_info in data_infos:
            query = {
                "spk": data_info["spk"],
                "basename": data_info["basename"],
            }
            res.append(query)
        return res
