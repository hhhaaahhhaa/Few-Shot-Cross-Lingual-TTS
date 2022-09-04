import os
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm

from lightning.systems import get_system

from dlhlp_lib.algorithm.dpdp import DPDPSSLUnit
from dlhlp_lib.parsers.preprocess import *
from dlhlp_lib.audio import AUDIO_CONFIG
import Define
from Parsers.parser import DataParser


INV_FRAME_PERIOD = AUDIO_CONFIG["audio"]["sampling_rate"] / AUDIO_CONFIG["stft"]["hop_length"]


class TransHeadModel(pl.LightningModule):
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
        
    def forward(self, repr):
        """
        Pass repr into codebook, input and output have the same size.
        Args:
            repr: Tensor with shape [bs, L, layer, dim].
        """
        repr = repr.to(self.device)
        x = self.pr_model.downstream(repr, [repr.size(1)])
        output = self.pr_model.head(x)  # bs, L, n_symbols
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
        return 1 - self.model(repr)  # Generate score for dpdp


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
            raise

    # Other preprocessing
    segment2duration_mp(unit_parser, queries, "dp_segment", "dp_duration", INV_FRAME_PERIOD, n_workers=os.cpu_count() // 2, refresh=True)
    duration_avg_pitch_and_energy_mp(data_parser, queries, f"ssl_units/{unit_name}/dp_duration", n_workers=os.cpu_count() // 2, refresh=True)


if __name__ == "__main__":
    if Define.CUDA_LAUNCH_BLOCKING:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    Define.set_upstream("hubert")
    pr_model = TransHeadModel(
        system_type="pr-fscl-tune",
        # ckpt_path="output/ckpt/tune/fscl/da94dfabfc394e1694471ba5625b0b57/checkpoints/epoch=5-step=1500.ckpt"  # 4shot
        # ckpt_path="output/ckpt/tune/fscl/353592528b6f457a813230ace92c905a/checkpoints/epoch=5-step=1500.ckpt"  # 16shot
        ckpt_path="output/ckpt/tune/fscl/06dcc1ee75dd422f9e4d8f1b254ca299/checkpoints/epoch=5-step=1500.ckpt"  # 64shot
    )
    pr_model = PostNetWrapper(pr_model, lang_id=6).eval().cuda()
    dpdp = DPDPSSLUnit('hubert_large_ll60k', postnet=pr_model)
    dpdp.cuda()

    # generate_ssl_units("pr-ssl-baseline-tune4-reg1", "./preprocessed_data/JSUT", dpdp, lambd=1)
    # generate_ssl_units("pr-ssl-baseline-tune4-reg0.3", "./preprocessed_data/JSUT", dpdp, lambd=0.3)
    # generate_ssl_units("pr-ssl-baseline-tune4-reg0", "./preprocessed_data/JSUT", dpdp, lambd=0)

    # generate_ssl_units("pr-ssl-baseline-tune16-reg1", "./preprocessed_data/JSUT", dpdp, lambd=1)
    # generate_ssl_units("pr-ssl-baseline-tune16-reg0.3", "./preprocessed_data/JSUT", dpdp, lambd=0.3)
    # generate_ssl_units("pr-ssl-baseline-tune16-reg0", "./preprocessed_data/JSUT", dpdp, lambd=0)

    generate_ssl_units("pr-fscl-tune64-reg1", "./preprocessed_data/JSUT", dpdp, lambd=1)
    generate_ssl_units("pr-fscl-tune64-reg0.3", "./preprocessed_data/JSUT", dpdp, lambd=0.3)
    generate_ssl_units("pr-fscl-tune64-reg0", "./preprocessed_data/JSUT", dpdp, lambd=0)
