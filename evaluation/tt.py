import os
import glob
import shutil
from tqdm import tqdm


os.makedirs(f"evaluation/output/linear/jp/4-shot", exist_ok=True)
os.makedirs(f"evaluation/output/linear/jp/8-shot", exist_ok=True)
os.makedirs(f"evaluation/output/linear/jp/16-shot", exist_ok=True)
for filename in tqdm(glob.glob(f"evaluation/output/linear/jp/*json")):
    _, s, idx = os.path.basename(filename)[:-5].split('-')
    shutil.move(filename, f"evaluation/output/linear/jp/{s}-shot/task-{idx}.json")
