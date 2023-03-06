# Deprecated, replaced in DLHLP lib
# import torch
# import numpy as np
# import librosa

# import pytorch_lightning as pl
# import s3prl.hub as hub

# import Define
# from text.define import LANG_ID2SYMBOLS
# from lightning.utils.tool import torch_exist_nan


# class S3PRLExtractor(pl.LightningModule):
#     def __init__(self, name):
#         super().__init__()
#         self.ssl_extractor = getattr(hub, name)()
#         self.repr_reduction = AverageReprModule()
#         # self.repr_reduction = RandomSelectReprModule()

#     def extract(self, info, norm=False, batch_size=32, no_text=False):
#         representation_list = []

#         wavs = []
#         for i in range(len(info["raw-feat"])):
#             wavs.append(info["raw-feat"][i].to(self.device))
#             if (i + 1) % batch_size == 0 or i == len(info["raw-feat"]) - 1:
#                 representation = self.ssl_extractor(wavs)
#                 representation = torch.stack(representation["hidden_states"], dim=1)  # bs, layer, L, dim
#                 if norm:
#                     representation = torch.nn.functional.normalize(representation, dim=3)
#                 representation_list.extend([r.detach().cpu() for r in representation])
#                 wavs = []

#         # SSL representation phoneme-level average
#         avg_frames = info["avg-frames"]

#         if no_text:
#             # repr_lens = torch.LongTensor([len(d_list) for d_list in avg_frames])
#             unsup_repr = []
#             for d_list, repr in zip(avg_frames, representation_list):
#                 pos = 0
#                 for i, d in enumerate(d_list):
#                     if d > 0 and not torch_exist_nan(repr[:, pos: pos + d, :]):
#                         repr[:, i] = torch.mean(repr[:, pos: pos + d, :], axis=1)
#                     else:
#                         repr[:, i] = torch.zeros((Define.UPSTREAM_LAYER, Define.UPSTREAM_DIM))
#                     pos += d
#                 repr = repr[:, :len(d_list)]
#                 unsup_repr.append(repr.transpose(0, 1))
#             unsup_repr = torch.nn.utils.rnn.pad_sequence(unsup_repr, batch_first=True)  # B, L, layer, dim
#             if Define.DEBUG:
#                 self.log("Pseudo phoneme-level average representations shape:")
#                 self.log(unsup_repr.shape)
#             return unsup_repr.to(self.device)
#         else:
#             # lang_id = info["lang_id"]
#             # n_symbols = len(LANG_ID2SYMBOLS[lang_id])
#             n_symbols = info["n_symbols"]
#             texts = info["texts"]
#             table = {i: [] for i in range(n_symbols)}
#             for text, d_list, repr in zip(texts, avg_frames, representation_list):
#                 pos = 0
#                 for i, (t, d) in enumerate(zip(text, d_list)):
#                     if d > 0:
#                         if not torch_exist_nan(repr[:, pos: pos + d, :]):
#                             table[int(t)].append(repr[:, pos: pos + d, :].mean(dim=1))
#                         else:
#                             print("oh, so bad...")
#                     pos += d

#             phn_repr = self.repr_reduction(n_symbols, table, (Define.UPSTREAM_LAYER, Define.UPSTREAM_DIM))
#             phn_repr = phn_repr.unsqueeze(0)  # 1, n_symbols, layer, dim
#             if Define.DEBUG:
#                 self.log("Phoneme-level average representations shape:")
#                 self.log(phn_repr.shape)
#             return phn_repr

#     def log(self, msg):
#         print("[SSL reference extractor]: ", msg)


# class HubertExtractor(S3PRLExtractor):
#     def __init__(self):
#         super().__init__('hubert_large_ll60k')


# class Wav2Vec2Extractor(S3PRLExtractor):
#     def __init__(self):
#         super().__init__('wav2vec2_large_ll60k')


# class XLSR53Extractor(S3PRLExtractor):
#     def __init__(self):
#         super().__init__('wav2vec2_xlsr')


# class MelExtractor(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.repr_reduction = AverageReprModule()
#         # self.repr_reduction = RandomSelectReprModule()

#     def extract(self, info, norm=False, no_text=False):
#         representation_list = [w.detach().cpu() for w in info["raw-feat"]]
#         avg_frames = info["avg-frames"]

#         if no_text:
#             # repr_lens = torch.LongTensor([len(d_list) for d_list in avg_frames])
#             unsup_repr = []
#             for d_list, repr in zip(avg_frames, representation_list):
#                 pos = 0
#                 for i, d in enumerate(d_list):
#                     if d > 0 and not torch_exist_nan(repr[pos: pos + d, :]):
#                         repr[i] = torch.mean(repr[pos: pos + d, :], axis=0)
#                     else:
#                         repr[i] = torch.zeros(Define.UPSTREAM_DIM)
#                     pos += d
#                 repr = repr[:len(d_list)]
#                 unsup_repr.append(repr)
#             unsup_repr = torch.nn.utils.rnn.pad_sequence(unsup_repr, batch_first=True)  # B, L, 80
#             if Define.DEBUG:
#                 self.log("Pseudo phoneme-level average representations shape:")
#                 self.log(unsup_repr.shape)
#             return unsup_repr
#         else:
#             # lang_id = info["lang_id"]
#             # n_symbols = len(LANG_ID2SYMBOLS[lang_id])
#             n_symbols = info["n_symbols"]
#             texts = info["texts"]

#             table = {i: [] for i in range(n_symbols)}
#             for text, d_list, repr in zip(texts, avg_frames, representation_list):
#                 pos = 0
#                 for i, (t, d) in enumerate(zip(text, d_list)):
#                     if d > 0:
#                         if not torch_exist_nan(repr[pos: pos + d, :]):
#                             table[int(t)].append(repr[pos: pos + d, :].mean(dim=0))
#                         else:
#                             print("oh, so bad...")
#                     pos += d

#             phn_repr = self.repr_reduction(n_symbols, table, (Define.UPSTREAM_DIM))
#             phn_repr = phn_repr.unsqueeze(0)  # 1, n_symbols, 80
#             if Define.DEBUG:
#                 self.log("Phoneme-level average representations shape:")
#                 self.log(phn_repr.shape)
#             return phn_repr

#     def log(self, msg):
#         print("[Mel reference extractor]: ", msg)


# class AverageReprModule(pl.LightningModule):
#     """
#     Given K class labels and some representations, average representations classwise.
#     """
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, n_class, table, repr_shape):
#         """
#         Args:
#             n_class: Total number of classes.
#             table: Dictionary with class label as key and list of representations belong to the
#                 label as value.
#             repr_shape: Shape of single representation.
#         Return:
#             Tensor with shape (n_class, *repr_shape).
#         """
#         avg_repr = []
#         for i in range(n_class):
#             if len(table[i]) == 0:
#                 avg_repr.append(torch.zeros(repr_shape))
#             else:
#                 avg_repr.append(torch.mean(torch.stack(table[i], axis=0), axis=0))
#         avg_repr = torch.stack(avg_repr, dim=0).float()

#         return avg_repr.to(self.device)


# import random
# class RandomSelectReprModule(pl.LightningModule):
#     """
#     Given K class labels and some representations, random select one representation for each class.
#     """
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, n_class, table, repr_shape):
#         """
#         Args:
#             n_class: Total number of classes.
#             table: Dictionary with class label as key and list of representations belong to the
#                 label as value.
#             repr_shape: Shape of single representation.
#         Return:
#             Tensor with shape (n_class, *repr_shape).
#         """
#         selected_repr = []
#         for i in range(n_class):
#             if len(table[i]) == 0:
#                 selected_repr.append(torch.zeros(repr_shape))
#             else:
#                 idx = random.randint(0, len(table[i]) - 1)
#                 selected_repr.append(table[i][idx])
#             # print(selected_repr[-1].is_cuda)  # this should always be false
#         selected_repr = torch.stack(selected_repr, dim=0).float()

#         return selected_repr.to(self.device)


# class PoolReprModule(pl.LightningModule):
#     """
#     Given K class labels and some representations, pool one representation for each class.
#     A representation pool is maintained.
#     """
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, n_class, table, repr_shape):
#         """
#         Args:
#             n_class: Total number of classes.
#             table: Dictionary with class label as key and list of representations belong to the
#                 label as value.
#             repr_shape: Shape of single representation.
#         Return:
#             Tensor with shape (n_class, *repr_shape).
#         """
#         selected_repr = []
#         for i in range(n_class):
#             if len(table[i]) == 0:
#                 selected_repr.append(torch.zeros(repr_shape))
#             else:
#                 idx = random.randint(0, len(table[i]) - 1)
#                 selected_repr.append(table[i][idx])
#             # print(selected_repr[-1].is_cuda)  # this should always be false
#         selected_repr = torch.stack(selected_repr, dim=0).float()

#         return selected_repr.to(self.device)


# if __name__ == "__main__":
#     extractor = HubertExtractor()
#     wav, _ = librosa.load("../../jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav", sr=16000)
#     ref = extractor.extract(wav.astype(np.float32))
#     print(ref.shape)
