import yaml


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
            else:
                config["symbol_id"] = config["lang_id"]
        return config
