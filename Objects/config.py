import yaml
import Define


class LanguageDataConfigReader(object):
    def __init__(self):
        pass

    def read(self, root):
        config = yaml.load(open(f"{root}/config.yaml", "r"), Loader=yaml.FullLoader)
        if "lang_id" not in config:
            config["lang_id"] = "en"
        for k in config['subsets']:
            config['subsets'][k] = f"{root}/{config['subsets'][k]}"
        if "symbol_id" not in config:
            if "n_symbols" in config:
                config["symbol_id"] = config["unit_name"]
                config["use_real_phoneme"] = False
            else:
                config["symbol_id"] = config["lang_id"]
                config["use_real_phoneme"] = True
        
        if Define.TUNET2U:
            config["target"] = {
                "unit_name": "enzhkofres-hubert_large_ll60k-24-512c",
                "n_symbols": 512,
            }
        if "target" in config:
            target = config["target"]
            if "n_symbols" in target:
                target["symbol_id"] = target["unit_name"]
                target["use_real_phoneme"] = False
            else:
                target["symbol_id"] = target["lang_id"]
                target["use_real_phoneme"] = True
        
        return config
