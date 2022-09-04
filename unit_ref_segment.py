import os
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm

from lightning.systems import get_system

from dlhlp_lib.s3prl import S3PRLExtractor
from dlhlp_lib.utils import torch_exist_nan
from dlhlp_lib.parsers.preprocess import *
from dlhlp_lib.audio import AUDIO_CONFIG
import Define
from Parsers.parser import DataParser


INV_FRAME_PERIOD = AUDIO_CONFIG["audio"]["sampling_rate"] / AUDIO_CONFIG["stft"]["hop_length"]


class SSLPRModel(pl.LightningModule):
    def __init__(self, system_type: str, ckpt_path: str) -> None:
        super().__init__()
        # TODO: Model initialization dependent on global initialization, should improve design
        keys = []
        with open("preprocessed_data/JSUT/stats.json") as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"]
            Define.ALLSTATS["JSUT"] = stats
            keys.append("JSUT")
        Define.ALLSTATS["global"] = Define.merge_stats(Define.ALLSTATS, keys)

        system = get_system(system_type)
        self.pr_model = system.load_from_checkpoint(ckpt_path)
        
    def forward(self, repr, lang_id):
        """
        Pass repr into codebook, input and output have the same size.
        Args:
            repr: Tensor with shape [bs, L, layer, dim].
        """
        repr = repr.to(self.device)
        x = self.pr_model.downstream(repr, [repr.size(1)])
        output = self.pr_model.head(x, lang_id=lang_id)  # bs, L, n_symbols
        output = F.softmax(output, dim=2)
        return output


class PostNetWrapper(pl.LightningModule):
    """
    Postnet wrapper of PR models for DPDP algorithm.
    """
    def __init__(self, model, lang_id: int) -> None:
        super().__init__()
        self.model = model
        self.lang_id = lang_id

    def forward(self, repr):
        return 1 - self.model(repr, lang_id=self.lang_id)  # Generate score for dpdp


class Decoder(object):
    """
    Postnet wrapper of PR models for DPDP algorithm.
    """
    def __init__(self, s3prl_name, layer=0, norm=False, postnet=None):
        self.debug = False
        self._extractor = S3PRLExtractor(s3prl_name)
        self._layer = layer
        self._norm = norm

        self._postnet = lambda x: x[:, :, layer]
        if postnet is not None:
            self._postnet = postnet

    def cuda(self):
        self._extractor.cuda()
    
    def cpu(self):
        self._extractor.cpu()

    def decode(self, wav_path: str, target_segment):
        repr, n_frame = self._extractor.extract_from_paths([wav_path], norm=self._norm) 
        repr = self._postnet(repr)
        sliced_repr = repr[0].detach().cpu()  # L, dim
        try:
            assert not torch_exist_nan(sliced_repr)
        except:
            self.log("NaN in SSL feature:")
            self.log(wav_path)
            raise ValueError
        
        sliced_repr = sliced_repr.numpy()
        L = sliced_repr.shape[0]

        # Perform classification according to target segment
        pos = 0
        boundaries, label_tokens = [], []
        for (s, e) in target_segment:
            d = int(np.round(e * 1000 / self._extractor._fp) - np.round(s * 1000 / self._extractor._fp))
            next_pos = min(L, pos + d)
            d = next_pos - pos
            if d == 0:
                continue
            boundaries.append(next_pos)
            t = np.mean(sliced_repr[pos: next_pos], axis=0).argmin()
            label_tokens.append(t)
            pos = next_pos
            if pos == L:
                break
        
        if self.debug:
            self.log(f"boundaries: {boundaries}")
            self.log(f"label_tokens: {label_tokens}")
            self.log(f"Num of segments = {len(label_tokens)}")
            print()

        foramtted_boundaries = []
        st = 0.0
        for b in boundaries:
            foramtted_boundaries.append((st, b * self._extractor._fp / 1000))
            st = b * self._extractor._fp / 1000
        
        return foramtted_boundaries, label_tokens

    def log(self, msg):
        print(f"[Decoder]: ", msg)


def generate_ssl_units(unit_name: str, root: str, target_segment_featname: str, decoder: Decoder):
    data_parser = DataParser(root)
    data_parser.create_ssl_unit_feature(unit_name=unit_name)
    queries = data_parser.get_all_queries()

    # DPDP
    unit_parser = data_parser.ssl_units[unit_name]
    segment_feat = unit_parser.dp_segment
    phoneme_feat = unit_parser.phoneme
    target_segment_feat = data_parser.get_feature(target_segment_featname)
    
    for query in tqdm(queries):
        try:
            wav_path = data_parser.wav_trim_16000.read_filename(query, raw=True)
            target_segment = target_segment_feat.read_from_query(query)
            segment, phoneme = decoder.decode(wav_path, target_segment)
            segment_feat.save(segment, query)
            phoneme = [str(phn) for phn in phoneme]
            phoneme_feat.save(" ".join(phoneme), query)
        except:
            print(query)
            raise

    # Other preprocessing
    segment2duration_mp(unit_parser, queries, "dp_segment", "dp_duration", INV_FRAME_PERIOD, n_workers=os.cpu_count() // 2, refresh=True)
    duration_avg_pitch_and_energy_mp(data_parser, queries, f"ssl_units/{unit_name}/dp_duration", n_workers=os.cpu_count() // 2, refresh=True)


if __name__ == "__main__":
    if Define.CUDA_LAUNCH_BLOCKING:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    Define.set_upstream("hubert")
    pr_model = SSLPRModel(
        system_type="pr-ssl-baseline-tune",
        ckpt_path="output/ckpt/tune/fscl/98e354d98ed448d7a6e406968e8dd93b/checkpoints/epoch=3-step=1000.ckpt"  # 4shot
        # ckpt_path="output/ckpt/tune/fscl/3d8e23d06e404de7921ccce324ab697b/checkpoints/epoch=9-step=2500.ckpt"  # 16shot
        # ckpt_path="output/ckpt/tune/fscl/72acfc01c03a43b3a0844d07b2a26f01/checkpoints/epoch=39-step=10000.ckpt"  # 64shot
        # ckpt_path="output/ckpt/tune/fscl/3939f48365ab4773bab43ce7b5b0ead3/checkpoints/epoch=39-step=10000.ckpt"  # 3000shot
    )
    pr_model = PostNetWrapper(pr_model, lang_id=6).eval().cuda()
    decoder = Decoder('hubert_large_ll60k', postnet=pr_model)
    decoder.cuda()
    
    generate_ssl_units("pr-ssl-baseline-tune4-reg0-seg-oracle", "./preprocessed_data/JSUT", f"mfa_segment", decoder)
    generate_ssl_units("pr-ssl-baseline-tune4-reg0-seg-gtcent-4shot-hubert-reg10", "./preprocessed_data/JSUT", f"ssl_units/gtcent-4shot-hubert-reg10/dp_segment", decoder)
