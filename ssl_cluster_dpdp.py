import os
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import pickle

from dlhlp_lib.algorithm.dpdp import DPDPSSLUnit
from dlhlp_lib.parsers.preprocess import *
from dlhlp_lib.audio import AUDIO_CONFIG

from lightning.systems import get_system
import Define
from Parsers.parser import DataParser
from text.define import LANG_ID2SYMBOLS


INV_FRAME_PERIOD = AUDIO_CONFIG["audio"]["sampling_rate"] / AUDIO_CONFIG["stft"]["hop_length"]


class SSLClusterPRModel(pl.LightningModule):
    def __init__(self, system_type: str, ckpt_path: str, cluster_path: str) -> None:
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

        # init clusters
        with open(cluster_path, 'rb') as f:
            info = pickle.load(f)
        
        self.missing_idxs = []
        exist_phonemes = list(info.keys())
        for i, phn in enumerate(LANG_ID2SYMBOLS[6]):
            if phn not in exist_phonemes:
                self.missing_idxs.append(i)

        clusters = torch.randn(len(LANG_ID2SYMBOLS[6]), 256)
        for phn, repr in info.items():
            clusters[LANG_ID2SYMBOLS[6].index(phn)] = torch.from_numpy(repr).float()
        self.pr_model.head.clusters[f"head-6"].data = clusters
        
    def forward(self, repr, lang_id):
        """
        Pass repr into codebook, input and output have the same size.
        Args:
            repr: Tensor with shape [bs, L, layer, dim].
        """
        repr = repr.to(self.device)
        x = self.pr_model.downstream(repr, [repr.size(1)])
        output = self.pr_model.head(x, lang_id=lang_id)  # bs, L, n_symbols
        output[:, :, self.missing_idxs] = -1000
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


def generate_ssl_units(unit_name: str, root: str, dpdp: DPDPSSLUnit, lambd=1):
    data_parser = DataParser(root)
    data_parser.create_ssl_unit_feature(unit_name=unit_name)
    queries = data_parser.get_all_queries()

    # DPDP
    unit_parser = data_parser.ssl_units[unit_name]
    segment_feat = unit_parser.dp_segment
    phoneme_feat = unit_parser.phoneme
    
    for query in tqdm(queries):
        try:
            wav_path = data_parser.wav_trim_16000.read_filename(query, raw=True)
            segment, phoneme = dpdp.segment_by_dist(wav_path, lambd=lambd)
            segment_feat.save(segment, query)
            phoneme = [str(phn) for phn in phoneme]
            phoneme_feat.save(" ".join(phoneme), query)
        except:
            print(query)
            continue

    # Other preprocessing
    segment2duration_mp(unit_parser, queries, "dp_segment", "dp_duration", INV_FRAME_PERIOD, n_workers=os.cpu_count() // 2, refresh=True)
    duration_avg_pitch_and_energy_mp(data_parser, queries, f"ssl_units/{unit_name}/dp_duration", n_workers=os.cpu_count() // 2, refresh=True)


if __name__ == "__main__":
    if Define.CUDA_LAUNCH_BLOCKING:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    Define.set_upstream("hubert")

    pr_model = SSLClusterPRModel(
        system_type="pr-ssl-cluster",
        ckpt_path="output/ckpt/pr/d67d47b67d674d8ba25c26425b521c6f/checkpoints/epoch=6-step=17500.ckpt",  # 
        cluster_path="_data/JSUT/hubert_large_ll60k-cluster-phoneme-4shot.pkl"
    )
    pr_model = PostNetWrapper(pr_model, lang_id=6).eval().cuda()
    dpdp = DPDPSSLUnit('hubert_large_ll60k', postnet=pr_model)
    dpdp.cuda()
    generate_ssl_units("pr-ssl-cluster-reg0", "./preprocessed_data/JSUT", dpdp, lambd=0)
    generate_ssl_units("pr-ssl-cluster-reg0.3", "./preprocessed_data/JSUT", dpdp, lambd=0.3)
