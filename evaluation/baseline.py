import os
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import scipy
from tqdm import tqdm
import pickle
import json
import gc

from dlhlp_lib.algorithm.dpdp import DPDPDecoder
from dlhlp_lib.utils.numeric import torch_exist_nan

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
        
    def forward(self, wav_path: str, lang_id):  # Since we can not get the correct length, batch inference is not implemented. 
        """
        Args:
            wav_path: Input wav raw path.
        Return:
            logits: Tensor with shape [L, n_symbols]. Client applies different decoding algorithms on it.
        """
        repr, _ = self.pr_model.upstream.extract_from_paths([wav_path]) 
        x = self.pr_model.downstream(repr, lengths=torch.LongTensor([repr.shape[1]]))
        x = self.pr_model.head(x, lang_id=lang_id)
        logits = x[0]  # L, dim
        return logits


def save_logits(
    pr_model,
    data_parser: DataParser,
    task_path: str,
    output_path: str,
    lang_id: int
) -> None:
    # read query from task
    task = config_reader.read(task_path)
    sup_txt, qry_txt = task["subsets"]["train"], task["subsets"]["test"]

    res = {}
    queries = read_queries_from_txt(sup_txt) + read_queries_from_txt(qry_txt)
    for query in tqdm(queries, leave=False):
        wav_path = data_parser.wav_trim_16000.read_filename(query, raw=True)
        with torch.no_grad():
            logits = pr_model(wav_path, lang_id=lang_id)
            try:
                assert not torch_exist_nan(logits)
            except:
                print(f"NaN in logits: {wav_path}.")
                raise ValueError
            logits = logits.detach().cpu().numpy()
        res[query["basename"]] = logits

    with open(output_path, 'wb') as f:
        pickle.dump(res, f)


def inference(
    data_parser: DataParser,
    task_path: str,
    logits_path: str,
    output_path: str,
    lang_id: int
):
    decoder = DPDPDecoder(lambd=0)

    # construct symbol mapping
    mapping = {}
    for i, p in enumerate(LANG_ID2SYMBOLS[lang_id]):
        if p[0] != '@':
            mapping[str(i)] = "none"
        else:
            mapping[str(i)] = p[1:]
    
    # read logits
    with open(logits_path, 'rb') as f:
        all_logits = pickle.load(f)

    # read query from task
    task = config_reader.read(task_path)
    sup_txt, qry_txt = task["subsets"]["train"], task["subsets"]["test"]

    # execution
    res = []
    queries = read_queries_from_txt(qry_txt)
    for query in tqdm(queries, leave=False):
        logits = all_logits[query["basename"]]
        score = -scipy.special.log_softmax(logits, axis=1)
        segment, phoneme = decoder.decode(score, fp=20)
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


def main_jp():
    Define.set_upstream(Define.UPSTREAM)
    system_type = "pr-ssl-baseline"
    exp_name = "baseline"
    output_exp_name = "baseline"  # maybe use difference decoding such as dpdp or lp
    data_parser = DataParser("preprocessed_data/JSUT")

    # Save logits (Only need to run once!)
    for s in [4, 8, 16]:
        os.makedirs(f"evaluation/logits/{exp_name}/jp/{s}-shot", exist_ok=True)
        for i in tqdm(range(20), desc=f"Save logits ({s} shot)"):
            ckpt_path = f"output/{exp_name}/jp/jp-{s}-{i}/ckpt/epoch=2-step=1500.ckpt"
            task_path = f"_data/JSUT/few-shot/{s}-shot/task-{i}"
            output_path = f"evaluation/logits/{exp_name}/jp/{s}-shot/task-{i}.pkl"
            pr_model = SSLPRModel(system_type, ckpt_path)
            pr_model.eval().cuda()
            save_logits(pr_model, data_parser, task_path, output_path, lang_id=6)
            pr_model.cpu()
            gc.collect()

    for s in [4, 8, 16]:
        os.makedirs(f"evaluation/output/{output_exp_name}/jp/{s}-shot", exist_ok=True)
        for i in tqdm(range(20), desc=f"Inference ({s} shot)"):
            task_path = f"_data/JSUT/few-shot/{s}-shot/task-{i}"
            logits_path = f"evaluation/logits/{exp_name}/jp/{s}-shot/task-{i}.pkl"
            output_path = f"evaluation/output/{output_exp_name}/jp/{s}-shot/task-{i}.json"
            inference(data_parser, task_path, logits_path, output_path, lang_id=6)


def main_ko():
    Define.set_upstream(Define.UPSTREAM)
    system_type = "pr-ssl-baseline"
    exp_name = "baseline"
    output_exp_name = "baseline"  # maybe use difference decoding such as dpdp or lp
    data_parser = DataParser("preprocessed_data/kss")

    # Save logits (Only need to run once!)
    for s in [4, 8, 16]:
        os.makedirs(f"evaluation/logits/{exp_name}/ko/{s}-shot", exist_ok=True)
        for i in tqdm(range(20), desc=f"Save logits ({s} shot)"):
            ckpt_path = f"output/{exp_name}/ko/ko-{s}-{i}/ckpt/epoch=2-step=1500.ckpt"
            task_path = f"_data/kss/few-shot/{s}-shot/task-{i}"
            output_path = f"evaluation/logits/{exp_name}/ko/{s}-shot/task-{i}.pkl"
            pr_model = SSLPRModel(system_type, ckpt_path)
            pr_model.eval().cuda()
            save_logits(pr_model, data_parser, task_path, output_path, lang_id=8)
            pr_model.cpu()
            gc.collect()

    for s in [4, 8, 16]:
        os.makedirs(f"evaluation/output/{output_exp_name}/ko/{s}-shot", exist_ok=True)
        for i in tqdm(range(20), desc=f"Inference ({s} shot)"):
            task_path = f"_data/kss/few-shot/{s}-shot/task-{i}"
            logits_path = f"evaluation/logits/{exp_name}/ko/{s}-shot/task-{i}.pkl"
            output_path = f"evaluation/output/{output_exp_name}/ko/{s}-shot/task-{i}.json"
            inference(data_parser, task_path, logits_path, output_path, lang_id=8)


if __name__ == "__main__":
    if Define.CUDA_LAUNCH_BLOCKING:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main_ko()
    main_jp()
