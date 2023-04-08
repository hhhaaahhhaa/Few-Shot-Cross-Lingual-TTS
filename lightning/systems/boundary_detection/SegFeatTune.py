import torch
import torch.nn as nn
import torch.nn.functional as F

from .SegFeat import SegFeatSystem


class SegFeatTuneSystem(SegFeatSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def tune_init(self, data_configs) -> None:
        assert len(data_configs) == 1, f"Currently only support adapting to one language"
        self.target_lang_id = data_configs[0]["lang_id"]
        print(f"Target Language: {self.target_lang_id}.")
