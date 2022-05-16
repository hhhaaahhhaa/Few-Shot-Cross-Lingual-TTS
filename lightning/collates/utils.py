import numpy as np
import torch

from lightning.utils.tool import pad_1D, pad_2D


def reprocess(data, idxs):
    """
    Pad data and calculate length of data. Unsupervised version has no text-related data.
    """
    unsup = False
    if data[idxs[0]]["text"] is None:
        unsup = True
    ids = [data[idx]["id"] for idx in idxs]
    speakers = [data[idx]["speaker"] for idx in idxs]
    if not unsup:
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
    mels = [data[idx]["mel"] for idx in idxs]
    pitches = [data[idx]["pitch"] for idx in idxs]
    energies = [data[idx]["energy"] for idx in idxs]
    durations = [data[idx]["duration"] for idx in idxs]

    if not unsup:
        text_lens = np.array([text.shape[0] for text in texts])
    mel_lens = np.array([mel.shape[0] for mel in mels])

    speakers = np.array(speakers)
    if not unsup:
        texts = pad_1D(texts)
    mels = pad_2D(mels)
    pitches = pad_1D(pitches)
    energies = pad_1D(energies)
    durations = pad_1D(durations)

    if "spk_ref_mel_slices" in data[0]:
        spk_ref_mels = [data[idx]["spk_ref_mel_slices"] for idx in idxs]
        # spk_ref_mel_lens = np.array([len(spk_ref_mel) for spk_ref_mel in spk_ref_mels])
        start = 0
        spk_ref_slices = []
        for spk_ref_mel in spk_ref_mels:
            end = start + spk_ref_mel.shape[0]
            spk_ref_slices.append(slice(start, end))
            start = end

        spk_ref_mels = np.concatenate(spk_ref_mels, axis=0)
        speaker_args = (
            torch.from_numpy(spk_ref_mels).float(),
            spk_ref_slices
        )
    else:
        speaker_args = torch.from_numpy(speakers).long()

    if not unsup:
        return (
            ids,
            raw_texts,
            speaker_args,
            torch.from_numpy(texts).long(),
            torch.from_numpy(text_lens),
            max(text_lens),
            torch.from_numpy(mels).float(),
            torch.from_numpy(mel_lens),
            max(mel_lens),
            torch.from_numpy(pitches).float(),
            torch.from_numpy(energies),
            torch.from_numpy(durations).long(),
        )
    
    else:
        return (
            ids,
            None,
            speaker_args,
            None,
            None,
            None,
            torch.from_numpy(mels).float(),
            torch.from_numpy(mel_lens),
            max(mel_lens),
            torch.from_numpy(pitches).float(),
            torch.from_numpy(energies),
            torch.from_numpy(durations).long(),
        )
