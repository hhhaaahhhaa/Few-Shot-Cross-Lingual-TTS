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
        x = self.pr_model.downstream(repr, dim=2)
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
    # pr_model = SSLPRModel(
    #     system_type="pr-ssl-linear-tune",
    #     ckpt_path="output/ckpt/tune/fscl/89204084d8974b5d8308abf134b4b82d/checkpoints/epoch=39-step=10000.ckpt"  # oracle
    # )
    # pr_model = PostNetWrapper(pr_model, lang_id=6).eval().cuda()
    # dpdp = DPDPSSLUnit('hubert_large_ll60k', postnet=pr_model)
    # dpdp.cuda()
    # generate_ssl_units("pr-ssl-linear-tune-oracle", "./preprocessed_data/JSUT", dpdp, lambd=0)

    pr_model = SSLPRModel(
        system_type="pr-ssl-linear-tune",
        ckpt_path="output/ckpt/tune/fscl/52231b421d184f998395552e9294c2ca/checkpoints/epoch=5-step=1500.ckpt"  # oracle
    )
    pr_model = PostNetWrapper(pr_model, lang_id=6).eval().cuda()
    dpdp = DPDPSSLUnit('hubert_large_ll60k', postnet=pr_model)
    dpdp.cuda()
    generate_ssl_units("pr-ssl-linear-tune4", "./preprocessed_data/JSUT", dpdp, lambd=0)

    # pr_model = SSLPRModel(
    #     system_type="pr-ssl-linear-tune",
    #     ckpt_path="output/ckpt/tune/fscl/0501b335ac894236bcccaf8580d4d4f8/checkpoints/epoch=39-step=10000.ckpt"  # oracle
    # )
    # pr_model = PostNetWrapper(pr_model, lang_id=8).eval().cuda()
    # dpdp = DPDPSSLUnit('hubert_large_ll60k', postnet=pr_model)
    # dpdp.cuda()
    # generate_ssl_units("pr-ssl-linear-tune-oracle", "./preprocessed_data/kss", dpdp, lambd=0)

    pr_model = SSLPRModel(
        system_type="pr-ssl-linear-tune",
        ckpt_path="output/ckpt/tune/fscl/f3a15db80e7d46ed8dcba25093777862/checkpoints/epoch=5-step=1500.ckpt"  # oracle
    )
    pr_model = PostNetWrapper(pr_model, lang_id=8).eval().cuda()
    dpdp = DPDPSSLUnit('hubert_large_ll60k', postnet=pr_model)
    dpdp.cuda()
    generate_ssl_units("pr-ssl-linear-tune4", "./preprocessed_data/kss", dpdp, lambd=0)
