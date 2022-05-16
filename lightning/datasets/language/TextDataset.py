import numpy as np
import torch
from torch.utils.data import Dataset

from lightning.utils.tool import pad_1D
from text import text_to_sequence


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        print("filepath: ", filepath)
        self.basename, self.text, self.raw_text = self.process_meta(
            filepath
        )
        self.lang_id = preprocess_config.get("lang_id", 0)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners, self.lang_id))

        return (basename, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                text.append(t)
                raw_text.append(r)
            return name, text, raw_text
    
    def collate_fn(self, data):
        ids = [d[0] for d in data]
        texts = [d[1] for d in data]
        raw_texts = [d[2] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return (ids, raw_texts, torch.from_numpy(texts).long(), torch.from_numpy(text_lens), max(text_lens))
