import torch
import numpy as np
import tgt
import resemblyzer
from resemblyzer import preprocess_wav, wav_to_mel_spectrogram


def check_nan(x):
    return np.any(np.isnan(x))


def check_nan2(x):
    return (x != x).any()


def torch_exist_nan(x):
    return np.any(np.isnan(x))


def torch_exist_nan(x):
    return (x != x).any()


def representation_average(representation, duration, pad=np.zeros(1024)):
    pos = 0
    for i, d in enumerate(duration):
        if d > 0:
            representation[i] = np.mean(
                representation[pos: pos + d], axis=0)
        else:
            representation[i] = pad
        pos += d
    return representation[: len(duration)]
    

def extract_spk_ref_mel_slices(wav_path):
    # speaker d-vector reference
    # Settings are slightly different from above, so should start again
    wav = preprocess_wav(wav_path)

    # Compute where to split the utterance into partials and pad the waveform
    # with zeros if the partial utterances cover a larger range.
    wav_slices, mel_slices = resemblyzer.VoiceEncoder.compute_partial_slices(
        len(wav), rate=1.3, min_coverage=0.75
    )
    max_wave_length = wav_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
    # Split the utterance into partials and forward them through the model
    spk_ref_mel = wav_to_mel_spectrogram(wav)
    spk_ref_mel_slices = [spk_ref_mel[s] for s in mel_slices]
    
    return spk_ref_mel_slices


class TextGridReader(object):

    SILENCE = ["sil", "sp", "spn"]

    def __init__(self, textgrid_path):
        self.textgrid = tgt.io.read_textgrid(textgrid_path)
        self.parse_info()

    def parse_info(self):
        tier = self.textgrid.get_tier_by_name("phones")
        
        phones = []
        durations = []
        start_time, end_time = 0, 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in self.SILENCE:
                    continue
                else:
                    start_time = s

            phones.append(p)
            durations.append((s, e))
            if p not in self.SILENCE:
                end_time = e
                end_idx = len(phones)

        self.phones = phones[:end_idx]
        self.durations = durations[:end_idx]
        self.start, self.end = start_time, end_time
