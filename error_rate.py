import numpy as np
import pickle
from tqdm import tqdm

from Parsers.parser import DataParser


def segment2duration(segment, fp):
    res = []
    for (s, e) in segment:
        res.append(
            int(
                np.round(e * 1 / fp)
                - np.round(s * 1 / fp)
            )
        )
    return res


def expand(seq, dur):
    assert len(seq) == len(dur)
    res = []
    for (x, d) in zip(seq, dur):
        if d > 0:
            res.extend([x] * d)
    return res


class FERCalculator(object):
    def __init__(self):
        pass

    def exec(self,
            data_parser: DataParser, 
            queries,
            phoneme_featname1: str, segment_featname1: str, 
            phoneme_featname2: str, segment_featname2: str,
            symbol_equal_func,
            fp: float
        ) -> float:
        phn_feat1 = data_parser.get_feature(phoneme_featname1)
        seg_feat1 = data_parser.get_feature(segment_featname1)
        phn_feat2 = data_parser.get_feature(phoneme_featname2)
        seg_feat2 = data_parser.get_feature(segment_featname2)

        n_frames, correct = 0, 0
        n_seg1, n_seg2 = 0, 0
        for query in tqdm(queries):
            phoneme1 = phn_feat1.read_from_query(query).strip().split(" ")
            segment1 = seg_feat1.read_from_query(query)
            phoneme2 = phn_feat2.read_from_query(query).strip().split(" ")
            segment2 = seg_feat2.read_from_query(query)

            n_seg1 += len(phoneme1)
            n_seg2 += len(phoneme2)

            duration1, duration2 = segment2duration(segment1, fp), segment2duration(segment2, fp)
            seq1, seq2 = expand(phoneme1, duration1), expand(phoneme2, duration2)
            total_len = min(sum(duration1), sum(duration2))

            for (x1, x2) in zip(seq1, seq2):
                if symbol_equal_func(x1, x2):
                    correct += 1
            n_frames += total_len
        fer = correct / n_frames

        print(f"Segments: {n_seg1}, {n_seg2}.")
        print(f"Frame error rate: {correct}/{n_frames} = {fer * 100:.2f}%")
        return fer


if __name__ == "__main__":
    calculator = FERCalculator()
    data_parser = DataParser("./preprocessed_data/JSUT")
    queries = data_parser.get_all_queries()

    # Construct symbol mapping
    with open("_data/JSUT/hubert-phoneme-4shot.pkl", 'rb') as f:
        table = pickle.load(f)
    mapping = {str(i): p[1:] for i, p in enumerate(table)}
    def equal_func(x1, x2):
        return mapping[x1] == x2
    
    fer = calculator.exec(
        data_parser, queries, 
        "ssl_units/gtcent-4shot-hubert-reg10/phoneme", "ssl_units/gtcent-4shot-hubert-reg10/dp_segment", 
        "phoneme", "mfa_segment",
        equal_func,
        fp=0.02
    )
