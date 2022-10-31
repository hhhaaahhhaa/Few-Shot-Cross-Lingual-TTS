import os
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm

from dlhlp_lib.algorithm.dpdp import DPDPSSLUnit
from dlhlp_lib.parsers.preprocess import *

import Define
from text.define import LANG_ID2SYMBOLS
from lightning.systems import get_system
from lightning.utils.tool import read_queries_from_txt
from Parsers.parser import DataParser
from Objects.config import LanguageDataConfigReader


config_reader = LanguageDataConfigReader()


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
        return -torch.log_softmax(self.model(repr, lang_id=self.lang_id), dim=2)  # Generate score for dpdp


def evaluate(
    system_type: str, ckpt_path: str,
    data_parser: DataParser,
    task_path: str,
    output_path: str,
    lang_id: int
):
    pr_model = SSLPRModel(system_type, ckpt_path)
    pr_model = PostNetWrapper(pr_model, lang_id=lang_id).eval().cuda()
    dpdp = DPDPSSLUnit('hubert_large_ll60k', postnet=pr_model)
    dpdp.cuda()

    # construct symbol mapping
    mapping = {}
    for i, p in enumerate(LANG_ID2SYMBOLS[lang_id]):
        if p[0] != '@':
            mapping[str(i)] = "none"
        else:
            mapping[str(i)] = p[1:]
    
    # read query from task
    task = config_reader.read(task_path)
    sup_txt, qry_txt = task["subsets"]["train"], task["subsets"]["test"]

    # execution
    res = []
    queries = read_queries_from_txt(qry_txt)
    for query in tqdm(queries):
        wav_path = data_parser.wav_trim_16000.read_filename(query, raw=True)
        segment, phoneme = dpdp.segment_by_dist(wav_path, lambd=0)
        phoneme = [mapping[str(phn)] for phn in phoneme]
        phoneme = ' '.join(phoneme)
        res.append({
            "pred_segment": segment,
            "pred": phoneme,
            "gt_segment": data_parser.mfa_segment.read_from_query(query),
            "gt": data_parser.phoneme.read_from_query(query),
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f)

    dpdp.cpu()
    del dpdp  # currently unknown what variable hold reference so memory can't be automatically freed...


def main_jp():
    Define.set_upstream(Define.UPSTREAM)
    system_type = "pr-ssl-linear-tune"
    exp_name = "linear"
    output_exp_name = "linear"  # maybe use difference decoding such as dpdp or lp
    data_parser = DataParser("preprocessed_data/JSUT")

    for s in [4, 8, 16]:
        os.makedirs(f"evaluation/output/{output_exp_name}/jp/{s}-shot", exist_ok=True)
        for i in range(20):
            print(f"Evaluate {s} shot, task {i}...")
            evaluate(
                system_type,
                f"output/{exp_name}/jp/jp-{s}-{i}/ckpt/epoch=2-step=1500.ckpt",
                data_parser,
                f"_data/JSUT/few-shot/{s}-shot/task-{i}",
                f"evaluation/output/{output_exp_name}/jp/{s}-shot/task-{i}.json",
                lang_id=6
            )


def main_ko():
    Define.set_upstream(Define.UPSTREAM)
    system_type = "pr-ssl-linear-tune"
    exp_name = "linear"
    output_exp_name = "linear"  # maybe use difference decoding such as dpdp or lp
    data_parser = DataParser("preprocessed_data/kss")

    for s in [4, 8, 16]:
        os.makedirs(f"evaluation/output/{output_exp_name}/ko/{s}-shot", exist_ok=True)
        for i in range(20):
            print(f"Evaluate {s} shot, task {i}...")
            evaluate(
                system_type,
                f"output/{exp_name}/ko/ko-{s}-{i}/ckpt/epoch=2-step=1500.ckpt",
                data_parser,
                f"_data/kss/few-shot/{s}-shot/task-{i}",
                f"evaluation/output/{output_exp_name}/ko/{s}-shot/task-{i}.json",
                lang_id=8
            )


if __name__ == "__main__":
    if Define.CUDA_LAUNCH_BLOCKING:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main_ko()
