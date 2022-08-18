import pickle

from dlhlp_lib.metrics.asr import FERCalculator, PERCalculator
from dlhlp_lib.metrics.speech_segmentation import SegmentationEvaluator

from Parsers.parser import DataParser


if __name__ == "__main__":
    calculator = FERCalculator()
    # calculator = PERCalculator()
    # calculator = SegmentationEvaluator()
    data_parser = DataParser("./preprocessed_data/JSUT")
    queries = data_parser.get_all_queries()

    # Construct symbol mapping for pseudo units
    with open("_data/JSUT/hubert-phoneme-4shot.pkl", 'rb') as f:
        table = pickle.load(f)
    unify_map1 = {str(i): p[1:] for i, p in enumerate(table)}
    
    # Construct symbol mapping for pseudo units
    # from text.define import LANG_ID2SYMBOLS
    # unify_map1, unify_map2 = {}, {}
    # for i, p in enumerate(LANG_ID2SYMBOLS[6]):  # jp
    #     if len(p) < 2:
    #         unify_map1[str(i)] = "none"
    #     else:
    #         unify_map1[str(i)] = p[1:]

    from text.define import LANG_ID2SYMBOLS
    unify_map2 = {}
    for i, p in enumerate(LANG_ID2SYMBOLS[6]):  # jp
        if len(p) < 2:
            unify_map2[p] = p
        else:
            unify_map2[p[1:]] = p[1:]
    
    def equal_func(x1, x2):
        return unify_map1[x1] == unify_map2[x2]

    # fer = calculator.exec(
    #     data_parser, queries, 
    #     "ssl_units/gtcent-hubert-reg10/phoneme", "ssl_units/gtcent-hubert-reg10/dp_segment", 
    #     "phoneme", "mfa_segment",
    #     unify_map1, unify_map2,
    #     fp=0.02
    # )

    fer = calculator.exec(
        data_parser, queries, 
        "phoneme", "mfa_segment", 
        "phoneme", "mfa_segment",
        unify_map2, unify_map2,
        fp=0.02
    )
    input()

    result = calculator.exec(
        data_parser, queries, 
        "mfa_segment", "ssl_units/gtcent-hubert-reg10/dp_segment",
    )
    result = calculator.exec(
        data_parser, queries, 
        "mfa_segment", "ssl_units/gtcent-4shot-hubert-reg10/dp_segment",
    )
    result = calculator.exec(
        data_parser, queries, 
        "mfa_segment", "ssl_units/pr-ssl-baseline-oracle-reg0/dp_segment",
    )
    result = calculator.exec(
        data_parser, queries, 
        "mfa_segment", "ssl_units/pr-ssl-baseline-oracle-reg0.3/dp_segment",
    )
    result = calculator.exec(
        data_parser, queries, 
        "mfa_segment", "ssl_units/pr-ssl-baseline-oracle-reg1/dp_segment",
    )
    result = calculator.exec(
        data_parser, queries, 
        "mfa_segment", "ssl_units/pr-ssl-baseline-tune4-reg0/dp_segment",
    )
    result = calculator.exec(
        data_parser, queries, 
        "mfa_segment", "ssl_units/pr-ssl-baseline-tune4-reg0.3/dp_segment",
    )
    result = calculator.exec(
        data_parser, queries, 
        "mfa_segment", "ssl_units/pr-ssl-baseline-tune4-reg1/dp_segment",
    )
    result = calculator.exec(
        data_parser, queries, 
        "mfa_segment", "ssl_units/pr-ssl-baseline-tune16-reg0/dp_segment",
    )
    result = calculator.exec(
        data_parser, queries, 
        "mfa_segment", "ssl_units/pr-ssl-baseline-tune16-reg0.3/dp_segment",
    )
    result = calculator.exec(
        data_parser, queries, 
        "mfa_segment", "ssl_units/pr-ssl-baseline-tune16-reg1/dp_segment",
    )
    result = calculator.exec(
        data_parser, queries, 
        "mfa_segment", "ssl_units/pr-ssl-baseline-tune64-reg0/dp_segment",
    )
    result = calculator.exec(
        data_parser, queries, 
        "mfa_segment", "ssl_units/pr-ssl-baseline-tune64-reg0.3/dp_segment",
    )
    result = calculator.exec(
        data_parser, queries, 
        "mfa_segment", "ssl_units/pr-ssl-baseline-tune64-reg1/dp_segment",
    )
    
    # wer = calculator.exec(
    #     data_parser, queries, 
    #     "phoneme", "ssl_units/gtcent-4shot-hubert-reg10/phoneme",
    #     unify_map2, unify_map1
    # )
    
    # wer = calculator.exec(
    #     data_parser, queries, 
    #     "phoneme", "ssl_units/pr-ssl-baseline-oracle-reg1/phoneme",
    #     unify_map2, unify_map1
    # )
    # wer = calculator.exec(
    #     data_parser, queries, 
    #     "phoneme", "ssl_units/pr-ssl-baseline-oracle-reg0.3/phoneme",
    #     unify_map2, unify_map1
    # )
    # wer = calculator.exec(
    #     data_parser, queries, 
    #     "phoneme", "ssl_units/pr-ssl-baseline-oracle-reg0/phoneme",
    #     unify_map2, unify_map1
    # )

    # wer = calculator.exec(
    #     data_parser, queries, 
    #     "phoneme", "ssl_units/pr-ssl-baseline-tune4-reg1/phoneme",
    #     unify_map2, unify_map1
    # )
    # wer = calculator.exec(
    #     data_parser, queries, 
    #     "phoneme", "ssl_units/pr-ssl-baseline-tune4-reg0.3/phoneme",
    #     unify_map2, unify_map1
    # )
    # wer = calculator.exec(
    #     data_parser, queries, 
    #     "phoneme", "ssl_units/pr-ssl-baseline-tune4-reg0/phoneme",
    #     unify_map2, unify_map1
    # )

    # wer = calculator.exec(
    #     data_parser, queries, 
    #     "phoneme", "ssl_units/pr-ssl-baseline-tune64-reg1/phoneme",
    #     unify_map2, unify_map1
    # )
    # wer = calculator.exec(
    #     data_parser, queries, 
    #     "phoneme", "ssl_units/pr-ssl-baseline-tune64-reg0.3/phoneme",
    #     unify_map2, unify_map1
    # )
    # wer = calculator.exec(
    #     data_parser, queries, 
    #     "phoneme", "ssl_units/pr-ssl-baseline-tune64-reg0/phoneme",
    #     unify_map2, unify_map1
    # )
    

    # fer = calculator.exec(
    #     data_parser, queries, 
    #     "ssl_units/pr-ssl-baseline-tune16-reg1/phoneme", "ssl_units/pr-ssl-baseline-tune16-reg1/dp_segment", 
    #     "phoneme", "mfa_segment",
    #     equal_func,
    #     fp=0.02
    # )

    # fer = calculator.exec(
    #     data_parser, queries, 
    #     "ssl_units/pr-ssl-baseline-tune16-reg0.3/phoneme", "ssl_units/pr-ssl-baseline-tune16-reg0.3/dp_segment", 
    #     "phoneme", "mfa_segment",
    #     equal_func,
    #     fp=0.02
    # )

    # fer = calculator.exec(
    #     data_parser, queries, 
    #     "ssl_units/pr-ssl-baseline-tune16-reg0/phoneme", "ssl_units/pr-ssl-baseline-tune16-reg0/dp_segment", 
    #     "phoneme", "mfa_segment",
    #     equal_func,
    #     fp=0.02
    # )

    # fer = calculator.exec(
    #     data_parser, queries, 
    #     "ssl_units/pr-ssl-baseline-tune4-reg1/phoneme", "ssl_units/pr-ssl-baseline-tune4-reg1/dp_segment", 
    #     "phoneme", "mfa_segment",
    #     equal_func,
    #     fp=0.02
    # )

    # fer = calculator.exec(
    #     data_parser, queries, 
    #     "ssl_units/pr-ssl-baseline-tune4-reg0.3/phoneme", "ssl_units/pr-ssl-baseline-tune4-reg0.3/dp_segment", 
    #     "phoneme", "mfa_segment",
    #     equal_func,
    #     fp=0.02
    # )

    # fer = calculator.exec(
    #     data_parser, queries, 
    #     "ssl_units/pr-ssl-baseline-tune4-reg0/phoneme", "ssl_units/pr-ssl-baseline-tune4-reg0/dp_segment", 
    #     "phoneme", "mfa_segment",
    #     equal_func,
    #     fp=0.02
    # )

    # fer = calculator.exec(
    #     data_parser, queries, 
    #     "ssl_units/pr-ssl-baseline-tune64-reg1/phoneme", "ssl_units/pr-ssl-baseline-tune64-reg1/dp_segment", 
    #     "phoneme", "mfa_segment",
    #     equal_func,
    #     fp=0.02
    # )

    # fer = calculator.exec(
    #     data_parser, queries, 
    #     "ssl_units/pr-ssl-baseline-tune64-reg0.3/phoneme", "ssl_units/pr-ssl-baseline-tune64-reg0.3/dp_segment", 
    #     "phoneme", "mfa_segment",
    #     equal_func,
    #     fp=0.02
    # )

    # fer = calculator.exec(
    #     data_parser, queries, 
    #     "ssl_units/pr-ssl-baseline-tune64-reg0/phoneme", "ssl_units/pr-ssl-baseline-tune64-reg0/dp_segment", 
    #     "phoneme", "mfa_segment",
    #     equal_func,
    #     fp=0.02
    # )

    # fer = calculator.exec(
    #     data_parser, queries, 
    #     "ssl_units/pr-ssl-baseline-oracle-reg1/phoneme", "ssl_units/pr-ssl-baseline-oracle-reg1/dp_segment", 
    #     "phoneme", "mfa_segment",
    #     equal_func,
    #     fp=0.02
    # )

    # fer = calculator.exec(
    #     data_parser, queries, 
    #     "ssl_units/pr-ssl-baseline-oracle-reg0.3/phoneme", "ssl_units/pr-ssl-baseline-oracle-reg0.3/dp_segment", 
    #     "phoneme", "mfa_segment",
    #     equal_func,
    #     fp=0.02
    # )

    # fer = calculator.exec(
    #     data_parser, queries, 
    #     "ssl_units/pr-ssl-baseline-oracle-reg0/phoneme", "ssl_units/pr-ssl-baseline-oracle-reg0/dp_segment", 
    #     "phoneme", "mfa_segment",
    #     equal_func,
    #     fp=0.02
    # )