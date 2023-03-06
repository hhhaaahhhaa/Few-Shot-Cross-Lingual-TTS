import numpy as np

from .T2UCollate import T2UCollate
from .LanguageCollate import LanguageCollate


class MixCollate(object):
    def __init__(self, t2u_data_configs, u2s_data_configs):
        self.u2s_collate = LanguageCollate(u2s_data_configs)
        self.t2u_collate = T2UCollate(t2u_data_configs)
    
    def collate_fn(self):
        return self._collate_fn

    def _collate_fn(self, data):
        t2u_data = [d["t2u"] for d in data]
        u2s_data = [d["u2s"] for d in data]

        # Ensure sorted order is text length instead of unit length here, since "text" inside u2s_data dict is
        # in fact "unit".
        len_arr = np.array([d["text"].shape[0] for d in t2u_data])
        idx_arr = np.argsort(-len_arr)
        t2u_data = [t2u_data[idx] for idx in idx_arr]
        u2s_data = [u2s_data[idx] for idx in idx_arr]

        return {
            "t2u": self.t2u_collate._collate_fn(t2u_data, sort=False, re_id=True),  
            "u2s": self.u2s_collate._collate_fn(u2s_data, sort=False, re_id=True)    
        }
