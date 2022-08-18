import os
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import s3prl.hub as hub
import time

from lightning.systems import get_system

from dlhlp_lib.algorithm.dpdp import DPDPSSLUnit
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
        x = self.pr_model.downstream(repr)
        output = self.pr_model.head(x, lang_id=lang_id)  # bs, L, n_symbols
        output = F.softmax(output, dim=2)
        return output


class SSLPRPostNet(pl.LightningModule):
    """
    Postnet wrapper of SSLPRModel for DPDP algorithm.
    """
    def __init__(self, model: SSLPRModel, lang_id: int) -> None:
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
            raise
            print(query)

    # Other preprocessing
    segment2duration_mp(unit_parser, queries, "dp_segment", "dp_duration", INV_FRAME_PERIOD, n_workers=os.cpu_count() // 2, refresh=True)
    duration_avg_pitch_and_energy_mp(data_parser, queries, f"ssl_units/{unit_name}/dp_duration", n_workers=os.cpu_count() // 2, refresh=True)


if __name__ == "__main__":
    if Define.CUDA_LAUNCH_BLOCKING:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    Define.set_upstream("hubert")
    pr_model = SSLPRModel(
        system_type="pr-ssl-baseline-tune",
        # ckpt_path="output/ckpt/tune/fscl/72acfc01c03a43b3a0844d07b2a26f01/checkpoints/epoch=39-step=10000.ckpt"  # 64shot
        # ckpt_path="output/ckpt/tune/fscl/3939f48365ab4773bab43ce7b5b0ead3/checkpoints/epoch=39-step=10000.ckpt"  # 3000shot
        # ckpt_path="output/ckpt/tune/fscl/98e354d98ed448d7a6e406968e8dd93b/checkpoints/epoch=3-step=1000.ckpt"  # 4shot
        ckpt_path="output/ckpt/tune/fscl/3d8e23d06e404de7921ccce324ab697b/checkpoints/epoch=9-step=2500.ckpt"  # 16shot
    )
    pr_model = SSLPRPostNet(pr_model, lang_id=6).eval().cuda()
    dpdp = DPDPSSLUnit('hubert_large_ll60k', postnet=pr_model)
    dpdp.cuda()

    # ========= debug section =============
    # dpdp.debug = True
    # generate_ssl_units("pr-debug", "./preprocessed_data/JSUT", dpdp)
    # dpdp.debug = False
    # ========= debug section =============
    
    # dpdp.debug = True
    generate_ssl_units("pr-ssl-baseline-tune16-reg1", "./preprocessed_data/JSUT", dpdp, lambd=1)
    generate_ssl_units("pr-ssl-baseline-tune16-reg0.3", "./preprocessed_data/JSUT", dpdp, lambd=0.3)
    generate_ssl_units("pr-ssl-baseline-tune16-reg0", "./preprocessed_data/JSUT", dpdp, lambd=0)
