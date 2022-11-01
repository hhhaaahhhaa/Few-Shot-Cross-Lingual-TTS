"""
Global setup from data_configs.
"""
from torchaudio.models.decoder import ctc_decoder

from text.define import LANG_ID2SYMBOLS
import Define
from Parsers.parser import DataParser


def build_id2symbols(data_configs):
    id2symbols = {}
    
    for data_config in data_configs:
        if data_config["symbol_id"] not in id2symbols:
            if data_config["symbol_id"] in LANG_ID2SYMBOLS:
                id2symbols[data_config["symbol_id"]] = LANG_ID2SYMBOLS[data_config["symbol_id"]]
            else:  # units which are not pseudo labels
                id2symbols[data_config["symbol_id"]] = [str(idx) for idx in range(data_config["n_symbols"])]

    return id2symbols


def build_data_parsers(data_configs):
    for data_config in data_configs:
        Define.DATAPARSERS[data_config["name"]] = DataParser(data_config["data_dir"])


def build_all_speakers(data_configs):
    res = []
    for data_config in data_configs:
        dataset_name = data_config["name"]
        if dataset_name not in Define.DATAPARSERS:
            Define.DATAPARSERS[dataset_name] = DataParser(data_config["data_dir"])
        spks = Define.DATAPARSERS[dataset_name].get_all_speakers()
        res.extend(spks)
    return res


def build_ctc_decoders(data_configs):
    id2symbols = build_id2symbols(data_configs)
    for data_config in data_configs:
        sym_id = data_config["symbol_id"]
        Define.CTC_DECODERS[sym_id] = ctc_decoder(
            lexicon=None,
            tokens=id2symbols[sym_id],
            lm=None,
            nbest=1,
            beam_size=50,
            beam_size_token=30
        )
