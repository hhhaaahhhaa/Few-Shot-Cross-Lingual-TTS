import torch
import numpy as np
import librosa

import pytorch_lightning as pl
import s3prl.hub as hub

import Define
from text.define import LANG_ID2SYMBOLS


class S3PRLExtractor(pl.LightningModule):
    def __init__(self, name):
        super().__init__()
        self.ssl_extractor = getattr(hub, name)()

    def extract_numpy(self, wav):
        wav = torch.from_numpy(wav).float().cuda()
        representation = self.ssl_extractor([wav])
        representation = torch.stack(representation["hidden_states"], dim=1)
        return representation.detach().cpu().numpy()

    def extract(self, info, norm=False):
        # remap keys
        info["ssl-wav"] = info["raw_feat"]
        for wav in info["ssl-wav"]:
            print(wav.sum())
        info["texts"] = info["phonemes"]
        info["ssl-duration"] = info["avg_frames"]

        # close normalize
        norm = False

        representation_list = []
        # representation = self.ssl_extractor([wav.cuda() for wav in info["ssl-wav"][:16]])
        # representation = torch.stack(representation["hidden_states"], dim=1)  # 16, 25, L, dim
        # if norm:
        #     representation = torch.nn.functional.normalize(representation, dim=3)
        # representation_list.extend([r for r in representation])

        if len(info["ssl-wav"]) == 256:
            for idx in [0, 32, 64, 96, 128, 160, 192, 224]:
                representation = self.ssl_extractor([wav.cuda() for wav in info["ssl-wav"][idx:idx+32]])
                representation = torch.stack(representation["hidden_states"], dim=1)  # 32, 25, L, dim
                if norm:
                    representation = torch.nn.functional.normalize(representation, dim=3)
                representation_list.extend([r for r in representation])

        elif len(info["ssl-wav"]) == 64:
            for idx in [0, 16, 32, 48]:
                representation = self.ssl_extractor([wav.cuda() for wav in info["ssl-wav"][idx:idx+16]])
                representation = torch.stack(representation["hidden_states"], dim=1)  # 16, 25, L, dim
                if norm:
                    representation = torch.nn.functional.normalize(representation, dim=3)
                representation_list.extend([r for r in representation])
        
        elif len(info["ssl-wav"]) <= 32:
            representation = self.ssl_extractor([wav.cuda() for wav in info["ssl-wav"]])
            representation = torch.stack(representation["hidden_states"], dim=1)  # B, 25, L, dim
            if norm:
                representation = torch.nn.functional.normalize(representation, dim=3)
            representation_list.extend([r for r in representation])
        
        print("Check s3prl")
        for r in representation_list:
            print(r[24].sum())

        # SSL representation phoneme-level average
        # lang_id = info["lang_id"]
        n_symbols = info["n_symbols"]
        # n_symbols = len(LANG_ID2SYMBOLS[lang_id])
        texts = info["texts"]
        ssl_durations = info["ssl-duration"]

        table = {i: [] for i in range(n_symbols)}
        for text, duration, repr in zip(texts, ssl_durations, representation_list):
            pos = 0
            for i, (t, d) in enumerate(zip(text, duration)):
                if d > 0:
                    # table[int(t)].append(repr[:, pos: pos + d, :])
                    table[int(t)].append(repr[:, pos: pos + d, :].mean(dim=1, keepdim=True))
                    
                pos += d

        phn_repr = torch.zeros((n_symbols, 25, Define.UPSTREAM_DIM), dtype=float)
        for i in range(n_symbols):
            if len(table[i]) == 0:
                phn_repr[i] = torch.zeros((25, Define.UPSTREAM_DIM))
            else:
                phn_repr[i] = torch.mean(torch.cat(table[i], axis=1), axis=1)

        return phn_repr.float()


class HubertExtractor(S3PRLExtractor):
    def __init__(self):
        super().__init__('hubert_large_ll60k')


class Wav2Vec2Extractor(S3PRLExtractor):
    def __init__(self):
        super().__init__('wav2vec2_large_ll60k')


class XLSR53Extractor(S3PRLExtractor):
    def __init__(self):
        super().__init__('wav2vec2_xlsr')


class MelExtractor(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def extract(self, info, norm=False):
        representation_list = [w for w in info["representation"]]

        # SSL representation phoneme-level average
        lang_id = info["lang_id"]
        n_symbols = len(LANG_ID2SYMBOLS[lang_id])
        texts = info["texts"]
        durations = info["duration"]

        table = {i: [] for i in range(n_symbols)}
        for text, duration, repr in zip(texts, durations, representation_list):
            pos = 0
            for i, (t, d) in enumerate(zip(text, duration)):
                if d > 0:
                    table[int(t)].append(repr[i:i+1, :])
                pos += d

        phn_repr = torch.zeros((n_symbols, Define.UPSTREAM_DIM), dtype=float)
        for i in range(n_symbols):
            if len(table[i]) == 0:
                phn_repr[i] = torch.zeros(Define.UPSTREAM_DIM)
            else:
                phn_repr[i] = torch.mean(torch.cat(table[i], axis=0), axis=0)

        return phn_repr.float()
