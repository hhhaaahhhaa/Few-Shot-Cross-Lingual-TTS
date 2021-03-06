import json
import torch
import os

import numpy as np
from torch.utils.data import Dataset
import resemblyzer
from resemblyzer.audio import preprocess_wav, wav_to_mel_spectrogram

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D
from preprocessor.reference_extractor import HubertExtractor, Wav2Vec2Extractor, XLSR53Extractor
from preprocessor.utils import representation_average
import Define


class TTSDataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, spk_refer_wav=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.spk_refer_wav = spk_refer_wav
        # if spk_refer_wav:
            # dset = filename.split('.')[0]
            # self.raw_path = os.path.join(preprocess_config["path"]["raw_path"], dset)

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.lang_id = 0

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners, self.lang_id))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        _, _, pitch_mu, pitch_std, _, _, energy_mu, energy_std = Define.ALLSTATS[self.lang_id]
        _, _, global_pitch_mu, global_pitch_std, _, _, global_energy_mu, global_energy_std = Define.ALLSTATS["global"]
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        pitch = (pitch * pitch_std + pitch_mu - global_pitch_mu) / global_pitch_std  # renormalize
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        energy = (energy * energy_std + energy_mu - global_energy_mu) / global_energy_std  # renormalize
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)
        assert not np.any(np.isnan(mel))
        assert not np.any(np.isnan(pitch))
        assert not np.any(np.isnan(energy))
        assert not np.any(np.isnan(duration))

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        if self.spk_refer_wav:
            spk_ref_mel_slices_path = os.path.join(
                self.preprocessed_path,
                "spk_ref_mel_slices",
                "{}-mel-{}.npy".format(speaker, basename),
            )
            spk_ref_mel_slices = np.load(spk_ref_mel_slices_path)

            sample.update({"spk_ref_mel_slices": spk_ref_mel_slices})

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text


class MonolingualTTSDataset(TTSDataset):

    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, spk_refer_wav=False
    ):
        super().__init__(filename, preprocess_config, train_config, sort, drop_last, spk_refer_wav)
        self.lang_id = preprocess_config["lang_id"]

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        basename = self.basename[idx]
        speaker = self.speaker[idx]

        representation, ssl_wav, ssl_duration = None, None, None

        if Define.UPSTREAM == "mel" or None:
            representation_path = f"{self.preprocessed_path}/mel-representation/{speaker}-mel-representation-{basename}.npy"
            representation = np.load(representation_path)
        else:  # Runtime loading
            ssl_wav_path = f"{self.preprocessed_path}/ssl-wav/{speaker}-ssl-wav-{basename}.npy"
            ssl_wav = np.load(ssl_wav_path)

            import pickle
            raw_dur_filename = f"{self.preprocessed_path}/raw-duration/{speaker}-raw-duration-{basename}.npy"
            with open(raw_dur_filename, 'rb') as f:
                raw_durations = pickle.load(f)

            ssl_duration = []
            for (s, e) in raw_durations:
                ssl_duration.append(
                    int(
                        np.round(e * 50)  # All ssl model use 20ms window
                        - np.round(s * 50)
                    )
                )

        sample.update({
            "language": self.lang_id,
            "representation": representation,
            "ssl-wav": ssl_wav,
            "ssl-duration": ssl_duration,
        })

        return sample


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)


class TextDataset2(Dataset):
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


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = TTSDataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = TTSDataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
