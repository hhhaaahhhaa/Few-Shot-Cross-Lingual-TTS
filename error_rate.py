import pickle
from typing import Dict

from dlhlp_lib.metrics.asr import FERCalculator, PERCalculator
from dlhlp_lib.metrics.speech_segmentation import SegmentationEvaluator

from Parsers.parser import DataParser


fer_calculator = FERCalculator()
per_calculator = PERCalculator()
seg_evaluator = SegmentationEvaluator()


def evaluate_ssl_unit(data_parser: DataParser, unit_name: str, symbol_ref2unify: Dict, symbol_pred2unify: Dict):
    queries = data_parser.get_all_queries()
    print(f"[{unit_name}]:")
    fer = fer_calculator.exec(
        data_parser, queries, 
        "phoneme", "mfa_segment",
        f"ssl_units/{unit_name}/phoneme", f"ssl_units/{unit_name}/dp_segment",
        symbol_ref2unify, symbol_pred2unify,
        fp=0.02
    )
    per = per_calculator.exec(
        data_parser, queries, 
        "phoneme",
        f"ssl_units/{unit_name}/phoneme",
        symbol_ref2unify, symbol_pred2unify
    )
    seg_result = seg_evaluator.exec(
        data_parser, queries, 
        "mfa_segment",
        f"ssl_units/{unit_name}/dp_segment",
    )
    print(f"Frame error rate: {fer * 100:.2f}%")
    print(f"Phoneme error rate: {per * 100:.2f}%")
    print(f"Recall: {seg_result['recall'] * 100:.2f}%")
    print(f"Precision: {seg_result['precision'] * 100:.2f}%")
    print(f"OS: {seg_result['os'] * 100:.2f}")
    print(f"R-val: {seg_result['r-value'] * 100:.2f}")
    print("")


if __name__ == "__main__":
    data_parser = DataParser("./preprocessed_data/JSUT")

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
    #         evaluate_ssl_unit(data_parser, f"pr-ssl-baseline-{shots}-reg{lambd}", symbol_ref2unify, symbol_pred2unify)
    
    # ssl_fscl_dpdp experiments
    # for shots in ["tune4", "tune16", "tune64"]:
    #     for lambd in ["0", "0.3", "1"]:
    #         evaluate_ssl_unit(data_parser, f"pr-fscl-{shots}-reg{lambd}", symbol_ref2unify, symbol_pred2unify)

    # unit_ref_segment experiments
    # evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune4-seg-oracle", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune4-seg-gtcent-4shot-hubert-reg10", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune4-seg-gtcent-hubert-reg10", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune4-seg-pr-ssl-baseline-oracle-reg0.3", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune4-seg-pr-ssl-baseline-oracle-reg0", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune4-seg-pr-ssl-baseline-tune4-reg0.3", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune16-seg-oracle", symbol_ref2unify, symbol_pred2unify)
    # evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune64-seg-oracle", symbol_ref2unify, symbol_pred2unify)

    evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune4-reg0-unk", symbol_ref2unify, symbol_pred2unify)
    evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune16-reg0-unk", symbol_ref2unify, symbol_pred2unify)
    evaluate_ssl_unit(data_parser, "pr-ssl-baseline-tune64-reg0-unk", symbol_ref2unify, symbol_pred2unify)
    
    # ssl_dpdp experiments
    # with open("_data/JSUT/hubert-phoneme-average.pkl", 'rb') as f:
    #     table = pickle.load(f)
    # symbol_pred2unify = {str(i): p[1:] for i, p in enumerate(table)}
    # evaluate_ssl_unit(data_parser, "gtcent-hubert-reg10", symbol_ref2unify, symbol_pred2unify)

    # with open("_data/JSUT/hubert-phoneme-4shot.pkl", 'rb') as f:
    #     table = pickle.load(f)
    # symbol_pred2unify = {str(i): p[1:] for i, p in enumerate(table)}
    # evaluate_ssl_unit(data_parser, "gtcent-4shot-hubert-reg10", symbol_ref2unify, symbol_pred2unify)
