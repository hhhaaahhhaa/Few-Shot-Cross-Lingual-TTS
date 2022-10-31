import numpy as np
import glob
import json
import jiwer
from tqdm import tqdm

from dlhlp_lib.utils.tool import segment2duration, expand


def fer(dir):
    errs = []
    for filename in tqdm(glob.glob(f"{dir}/*json")):
        with open(filename, 'r', encoding='utf-8') as f:
            infos = json.load(f)
        err = []
        for info in infos:
            ref_segment, pred_segment = info["gt_segment"], info["pred_segment"]
            ref_phoneme, pred_phoneme = info["gt"].strip().split(" "), info["pred"].strip().split(" ")
            ref_duration, pred_duration = segment2duration(ref_segment, 0.02), segment2duration(pred_segment, 0.02)
            ref_seq, pred_seq = expand(ref_phoneme, ref_duration), expand(pred_phoneme, pred_duration) 
                
            if len(pred_seq) >= len(ref_seq):
                pred_seq = pred_seq[:len(ref_seq)]
            else:
                padding = [pred_seq[-1]] * (len(ref_seq) - len(pred_seq))
                pred_seq.extend(padding)
            assert len(pred_seq) == len(ref_seq)
            correct = 0
            for (x1, x2) in zip(ref_seq, pred_seq):
                if x1 == x2:
                    correct += 1
            err.append(1 - correct / len(ref_seq))    
        errs.append(sum(err) / len(err))
    
    result = sum(errs) / len(errs)
    std = np.std(errs)
    print(f"[{dir}] FER: {result * 100:.2f}%, std {std * 100:.2f}%.")


def per(dir):
    errs = []
    for filename in tqdm(glob.glob(f"{dir}/*json")):
        with open(filename, 'r', encoding='utf-8') as f:
            infos = json.load(f)
        err = []
        for info in infos:
            err.append(jiwer.wer(info["gt"], info["pred"]))
        errs.append(sum(err) / len(err))

    result = sum(errs) / len(errs)
    std = np.std(errs)
    print(f"[{dir}] PER: {result * 100:.2f}%, std {std * 100:.2f}%.")


if __name__ == "__main__":
    # for s in [4, 8, 16]:
    #     per(f"evaluation/output/linear/ko/{s}-shot")
    #     fer(f"evaluation/output/linear/ko/{s}-shot")
    # for s in [4, 8, 16]:
    #     per(f"evaluation/output/linear/jp/{s}-shot")
    #     fer(f"evaluation/output/linear/jp/{s}-shot")
    
    # for s in [4, 8, 16]:
    #     per(f"evaluation/output/baseline/ko/{s}-shot")
    #     fer(f"evaluation/output/baseline/ko/{s}-shot")
    # for s in [4, 8, 16]:
    #     per(f"evaluation/output/baseline/jp/{s}-shot")
    #     fer(f"evaluation/output/baseline/jp/{s}-shot")

    for s in [4, 8, 16]:
        per(f"evaluation/output/protonet-zs/ko/{s}-shot")
        fer(f"evaluation/output/protonet-zs/ko/{s}-shot")
    for s in [4, 8, 16]:
        per(f"evaluation/output/protonet-zs/jp/{s}-shot")
        fer(f"evaluation/output/protonet-zs/jp/{s}-shot")
