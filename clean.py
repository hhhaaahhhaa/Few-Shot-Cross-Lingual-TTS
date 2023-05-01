import os
import torch
from tqdm import tqdm
import json
import gc

from dlhlp_lib.s3prl import S3PRLExtractor
from dlhlp_lib.utils.numeric import torch_exist_nan

from Parsers.parser import DataParser


class LengthChecker(object):
    def __init__(self, data_parser: DataParser, mi=1, mx=15):
        self.data_parser = data_parser
        self.data_parser.mfa_segment.read_all()
        self.mi = mi
        self.mx = mx

    def check(self, query) -> bool:
        try:
            segment = self.data_parser.mfa_segment.read_from_query(query)
            l = segment[-1][1] - segment[0][0]
            assert self.mi <= l and l <= self.mx 
        except:
            return False
        return True


class ExistenceChecker(object):
    def __init__(self, data_parser: DataParser):
        self.data_parser = data_parser

    def check(self, query) -> bool:
        try:
            filenames = [
                self.data_parser.mel.read_filename(query, raw=True),
                self.data_parser.interpolate_pitch.read_filename(query, raw=True),
                self.data_parser.energy.read_filename(query, raw=True),
                self.data_parser.mfa_duration.read_filename(query, raw=True),
                self.data_parser.spk_ref_mel_slices.read_filename(query, raw=True)
            ]
            for f in filenames:
                assert os.path.exists(f)
        except:
            return False
        return True


class SSLFeatureChecker(object):
    def __init__(self, s3prl_name: str, data_parser: DataParser):
        self.data_parser = data_parser
        self.extractor = S3PRLExtractor(s3prl_name)

    def check(self, query) -> bool:
        try:
            wav_path = self.data_parser.wav_trim_16000.read_filename(query, raw=True)
            with torch.no_grad():
                repr, _ = self.extractor.extract_from_paths([wav_path])
                assert not torch_exist_nan(repr)
        except:
            return False
        return True


class UnknownTokenChecker(object):
    def __init__(self, data_parser: DataParser):
        self.data_parser = data_parser
        self.data_parser.phoneme.read_all()

    def check(self, query) -> bool:
        try:
            phoneme = self.data_parser.phoneme.read_from_query(query)
            assert "spn" not in phoneme.split(" ")
        except:
            return False
        return True


def clean(root: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_parser = DataParser(root)
    res = data_parser.get_all_queries()

    print("Check existence...")
    filtered = []
    checker = ExistenceChecker(data_parser)
    for query in tqdm(res):
        if checker.check(query):
            filtered.append(query)
    print(f"{len(res)} => {len(filtered)}")
    res = filtered
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)

    print("Check length...")
    filtered = []
    checker = LengthChecker(data_parser)
    for query in tqdm(res):
        if checker.check(query):
            filtered.append(query)
    print(f"{len(res)} => {len(filtered)}")
    res = filtered
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)

    print("Check unknown tokens (spn)...")
    filtered = []
    checker = UnknownTokenChecker(data_parser)
    for query in tqdm(res):
        if checker.check(query):
            filtered.append(query)
    print(f"{len(res)} => {len(filtered)}")
    res = filtered
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)

    # for s3prl_name in ["hubert", "hubert_large_ll60k", 
    #             "wav2vec2", "wav2vec2_large_ll60k",
    #             "xlsr_53"]:
    #     print(f"Check SSL feature({s3prl_name})...")
    #     filtered = []
    #     checker = SSLFeatureChecker(s3prl_name, data_parser)
    #     checker.extractor.cuda()
    #     for query in tqdm(res):
    #         if checker.check(query):
    #             filtered.append(query)
    #     res = filtered
    #     with open(output_path, 'w', encoding='utf-8') as f:
    #         json.dump(res, f, indent=4)
    #     checker.extractor.cpu()
    #     gc.collect()


if __name__ == "__main__":
    clean("./preprocessed_data/LibriTTS", "_data/LibriTTS/clean.json")
    clean("./preprocessed_data/AISHELL-3", "_data/AISHELL-3/clean.json")
    clean("./preprocessed_data/kss", "_data/kss/clean.json")
    clean("./preprocessed_data/JSUT", "_data/JSUT/clean.json")
    clean("./preprocessed_data/CSS10/german", "_data/CSS10/german/clean.json")
    clean("./preprocessed_data/LJSpeech", "_data/LJSpeech/clean.json")
    clean("./preprocessed_data/CSS10/french", "_data/CSS10/french/clean.json")
    clean("./preprocessed_data/CSS10/spanish", "_data/CSS10/spanish/clean.json")
