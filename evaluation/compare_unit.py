import pickle
import numpy as np
from typing import Dict
from tqdm import tqdm

from dlhlp_lib.metrics.asr import FERCalculator, PERCalculator
from dlhlp_lib.metrics.speech_segmentation import SegmentationEvaluator

from Parsers.parser import DataParser


fer_calculator = FERCalculator()
per_calculator = PERCalculator()
seg_evaluator = SegmentationEvaluator()


def segment2duration(segment, fp):
    res = []
    for (s, e) in segment:
        res.append(
            int(
                round(round(e * 1 / fp, 4))
                - round(round(s * 1 / fp, 4))
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


def evaluate_pl_filter(
    unit_name: str, root: str,
    symbol_ref2unify: Dict, symbol_pred2unify: Dict,
    # thresholds = [0.1 * i for i in range(1, 10)]
    thresholds = [0.01, 0.2, 0.9, 0.95]
):  # unit_name in front for consistency with _dpdp scripts
    fp = 0.02
    data_parser = DataParser(root)
    queries = data_parser.get_all_queries()

    print(f"[{unit_name}]:")
    ref_phn_feat = data_parser.get_feature("phoneme")
    ref_seg_feat = data_parser.get_feature("mfa_segment")
    alignment_matrix_feat = data_parser.get_feature(f"ssl_units/{unit_name}/lp_matrix")
    alignment_matrix_feat.read_all()
    ref_phn_feat.read_all()
    ref_seg_feat.read_all()

    correct, values = [], []
    fail_cnt = 0
    for query in tqdm(queries):
        try:
            mat = alignment_matrix_feat.read_from_query(query)
            ref_phoneme = ref_phn_feat.read_from_query(query).strip().split(" ")
            ref_segment = ref_seg_feat.read_from_query(query)
            ref_duration = segment2duration(ref_segment, fp)
            ref_seq = expand(ref_phoneme, ref_duration)
            assert mat.shape[0] <= len(ref_seq)

            pred_seq = np.argmax(1 - mat, axis=1)
            pred_value = np.max(1 - mat, axis=1)
            for (x1, x2) in zip(ref_seq, pred_seq):
                if symbol_ref2unify[x1] == symbol_pred2unify[str(x2)]:
                    correct.append(1)
                else:
                    correct.append(0)
            values.extend(pred_value.tolist())
        except:
            fail_cnt += 1
            continue
    print("Skipped: ", fail_cnt)

    n_frames = len(correct)
    correct = np.array(correct)
    values = np.array(values)
    print(f"Total: {n_frames}")
    for threshold in thresholds:
        activated = np.sum(values > threshold)
        matched = np.sum(correct[values > threshold])
        print(f"Threshold {threshold}:")
        print(f"Activated: {activated}/{n_frames} = {activated / n_frames * 100:.2f}%")
        print(f"Accuracy: {matched}/{n_frames} = {matched / n_frames * 100:.2f}%")
        print("")


def evaluate_ssl_unit(unit_name: str, root: str, symbol_ref2unify: Dict, symbol_pred2unify: Dict):  # unit_name in front for consistency with _dpdp scripts
    data_parser = DataParser(root)
    queries = data_parser.get_all_queries()

    print(f"[{unit_name}]:")
    fer = fer_calculator.exec(
        data_parser, queries, 
        "phoneme", "mfa_segment",
        f"ssl_units/{unit_name}/phoneme", f"ssl_units/{unit_name}/dp_segment",
        symbol_ref2unify, symbol_pred2unify,
        fp=0.02
    )
    per_dict = per_calculator.exec(
        data_parser, queries, 
        "phoneme",
        f"ssl_units/{unit_name}/phoneme",
        symbol_ref2unify, symbol_pred2unify, return_dict=True
    )
    per = per_dict["wer"]
    seg_result = seg_evaluator.exec(
        data_parser, queries, 
        "mfa_segment",
        f"ssl_units/{unit_name}/dp_segment",
    )
    print(per_dict)
    print(f"Frame error rate: {fer * 100:.2f}%")
    print(f"Phoneme error rate: {per * 100:.2f}%")
    print(f"Recall: {seg_result['recall'] * 100:.2f}%")
    print(f"Precision: {seg_result['precision'] * 100:.2f}%")
    print(f"OS: {seg_result['os'] * 100:.2f}")
    print(f"R-val: {seg_result['r-value'] * 100:.2f}")
    print("")


if __name__ == "__main__":
    from text.define import LANG_ID2SYMBOLS
    symbol_ref2unify, symbol_pred2unify = {}, {}
    for i, p in enumerate(LANG_ID2SYMBOLS[6]):  # jp
        if len(p) < 2:
            symbol_ref2unify[p] = p
        else:
            symbol_ref2unify[p[1:]] = p[1:]
    
    # Construct symbol mapping for pseudo units
    for i, p in enumerate(LANG_ID2SYMBOLS[6]):  # jp
        if len(p) < 2:
            symbol_pred2unify[str(i)] = "none"
        else:
            symbol_pred2unify[str(i)] = p[1:]

    # ssl_baseline_dpdp experiments
    # for shots in ["tune4", "tune16", "tune64", "oracle"]:
    #     for lambd in ["0", "0.3", "1"]:
    #         evaluate_ssl_unit(f"pr-ssl-baseline-{shots}-reg{lambd}", "./preprocessed_data/JSUT", symbol_ref2unify, symbol_pred2unify)
    
    # ssl_fscl_dpdp experiments
    # for shots in ["tune4", "tune16", "tune64"]:
    #     for lambd in ["0", "0.3", "1"]:
    #         evaluate_ssl_unit(f"pr-fscl-{shots}-reg{lambd}", "./preprocessed_data/JSUT", symbol_ref2unify, symbol_pred2unify)

    # ssl_linear_dpdp experiments
    # evaluate_ssl_unit(f"pr-ssl-linear-tune4", "./preprocessed_data/JSUT", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(f"pr-ssl-linear-tune-oracle", "./preprocessed_data/JSUT", symbol_ref2unify, symbol_pred2unify)
    # evaluate_pl_filter(f"pr-ssl-linear-tune4", "./preprocessed_data/JSUT", symbol_ref2unify, symbol_pred2unify)
    # evaluate_pl_filter(f"pr-ssl-linear-tune-oracle", "./preprocessed_data/JSUT", symbol_ref2unify, symbol_pred2unify)
    evaluate_pl_filter(f"pr-ssl-cluster-lp", "./preprocessed_data/JSUT", symbol_ref2unify, symbol_pred2unify)
    
    # ssl_baseline_dpdp (enzh) experiments
    # evaluate_ssl_unit(f"pr-ssl-enzh-baseline-tune4", "./preprocessed_data/JSUT", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(f"pr-ssl-enzh-baseline-tune-oracle", "./preprocessed_data/JSUT", symbol_ref2unify, symbol_pred2unify)
    # evaluate_pl_filter(f"pr-ssl-enzh-baseline-tune4", "./preprocessed_data/JSUT", symbol_ref2unify, symbol_pred2unify)
    # evaluate_pl_filter(f"pr-ssl-enzh-baseline-tune-oracle", "./preprocessed_data/JSUT", symbol_ref2unify, symbol_pred2unify)
    
    # ko
    from text.define import LANG_ID2SYMBOLS
    symbol_ref2unify, symbol_pred2unify = {}, {}
    for i, p in enumerate(LANG_ID2SYMBOLS[8]):  # ko
        if len(p) < 2:
            symbol_ref2unify[p] = p
        else:
            symbol_ref2unify[p[1:]] = p[1:]
    
    # Construct symbol mapping for pseudo units
    for i, p in enumerate(LANG_ID2SYMBOLS[8]):  # ko
        if len(p) < 2:
            symbol_pred2unify[str(i)] = "none"
        else:
            symbol_pred2unify[str(i)] = p[1:]
    
    # ssl_linear_dpdp experiments
    # evaluate_ssl_unit(f"pr-ssl-linear-tune4", "./preprocessed_data/kss", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(f"pr-ssl-linear-tune-oracle", "./preprocessed_data/kss", symbol_ref2unify, symbol_pred2unify)
    # evaluate_pl_filter(f"pr-ssl-linear-tune4", "./preprocessed_data/kss", symbol_ref2unify, symbol_pred2unify)
    # evaluate_pl_filter(f"pr-ssl-linear-tune-oracle", "./preprocessed_data/kss", symbol_ref2unify, symbol_pred2unify)

    # ssl_baseline_dpdp (enzh) experiments
    # evaluate_ssl_unit(f"pr-ssl-enzh-baseline-tune4", "./preprocessed_data/kss", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(f"pr-ssl-enzh-baseline-tune-oracle", "./preprocessed_data/kss", symbol_ref2unify, symbol_pred2unify)
    # evaluate_pl_filter(f"pr-ssl-enzh-baseline-tune4", "./preprocessed_data/kss", symbol_ref2unify, symbol_pred2unify)
    # evaluate_pl_filter(f"pr-ssl-enzh-baseline-tune-oracle", "./preprocessed_data/kss", symbol_ref2unify, symbol_pred2unify)

    
    # unit_ref_segment experiments
    # evaluate_ssl_unit("./preprocessed_data/kss", "pr-ssl-baseline-tune4-seg-oracle", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune16-seg-oracle", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune64-seg-oracle", symbol_ref2unify, symbol_pred2unify)

    # evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune4-seg-gtcent-4shot-hubert-reg10", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune4-seg-gtcent-hubert-reg10", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune4-seg-pr-ssl-baseline-oracle-reg0.3", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune4-seg-pr-ssl-baseline-oracle-reg0", symbol_ref2unify, symbol_pred2unify)

    # ssl-cluster experiments
    # evaluate_ssl_unit(data_parser, f"pr-ssl-cluster-reg0", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, f"pr-ssl-cluster-reg0.3", symbol_ref2unify, symbol_pred2unify)

    # ssl_dpdp experiments
    # with open("_data/JSUT/hubert-phoneme-average.pkl", 'rb') as f:
    #     table = pickle.load(f)
    # symbol_pred2unify = {str(i): p[1:] for i, p in enumerate(table)}
    # evaluate_ssl_unit(data_parser, "gtcent-hubert-reg10", symbol_ref2unify, symbol_pred2unify)

    # with open("_data/JSUT/hubert-phoneme-4shot.pkl", 'rb') as f:
    #     table = pickle.load(f)
    # symbol_pred2unify = {str(i): p[1:] for i, p in enumerate(table)}
    # evaluate_ssl_unit(data_parser, "gtcent-4shot-hubert-reg10", symbol_ref2unify, symbol_pred2unify)

    # with open("_data/JSUT/hubert-phoneme-4shot-debug.pkl", 'rb') as f:
    #     table = pickle.load(f)
    # symbol_pred2unify = {str(i): p[1:] for i, p in enumerate(table)}
    # evaluate_ssl_unit(data_parser, "gtcent-4shot-hubert-reg10-debug", symbol_ref2unify, symbol_pred2unify)

    # ko
    # with open("_data/kss/wav2vec2-l-phoneme-4shot.pkl", 'rb') as f:
    #     table = pickle.load(f)
    # symbol_pred2unify = {str(i): p[1:] for i, p in enumerate(table)}
    # # evaluate_ssl_unit(data_parser, "gtcent-4shot-hubert-reg10", symbol_ref2unify, symbol_pred2unify)
    # # evaluate_ssl_unit(data_parser, "gtcent-4shot-hubert-reg20", symbol_ref2unify, symbol_pred2unify)
    # # evaluate_ssl_unit(data_parser, "gtcent-4shot-hubert-reg30", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "gtcent-4shot-hubert_small-reg10", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "gtcent-4shot-hubert_small-reg20", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "gtcent-4shot-hubert_small-reg30", symbol_ref2unify, symbol_pred2unify)

    # with open("_data/kss/hubert-phoneme-16shot.pkl", 'rb') as f:
    #     table = pickle.load(f)
    # symbol_pred2unify = {str(i): p[1:] for i, p in enumerate(table)}
    # evaluate_ssl_unit(data_parser, "gtcent-16shot-hubert-reg10", symbol_ref2unify, symbol_pred2unify)

    # with open("_data/kss/hubert-phoneme-64shot.pkl", 'rb') as f:
    #     table = pickle.load(f)
    # symbol_pred2unify = {str(i): p[1:] for i, p in enumerate(table)}
    # evaluate_ssl_unit(data_parser, "gtcent-64shot-hubert-reg10", symbol_ref2unify, symbol_pred2unify)
