import numpy as np
import torch

from lightning.utils.tool import pad_1D, pad_2D


def reprocess(data, idxs, mode="sup"):
    """
    Pad data and calculate length of data. Unsupervised version has no text-related data.
    Inference version has only text and speaker data.
    Args:
        mode: "sup", "unsup", or "inference".
    """
    ids = [data[idx]["id"] for idx in idxs]
    speakers = [data[idx]["speaker"] for idx in idxs]
    speakers = np.array(speakers)

    if mode in ["sup", "inference"]:
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        text_lens = np.array([text.shape[0] for text in texts])
        texts = pad_1D(texts)

    if mode in ["sup", "unsup"]:
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        mel_lens = np.array([mel.shape[0] for mel in mels])

    if mode in ["unsup"]:  # Duration has same length with text, which is equal to the number of segments.
        text_lens = np.array([len(duration) for duration in durations])

    if mode in ["sup", "unsup"]:
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

    if mode == "sup":
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
    elif mode == "unsup":
        return (
            ids,
            None,
            speaker_args,
            None,
            torch.from_numpy(text_lens),
            max(text_lens),
            torch.from_numpy(mels).float(),
            torch.from_numpy(mel_lens),
            max(mel_lens),
            torch.from_numpy(pitches).float(),
            torch.from_numpy(energies),
            torch.from_numpy(durations).long(),
        )
    elif mode == "inference":
        return (
            ids,
            raw_texts,
            speaker_args,
            torch.from_numpy(texts).long(),
            torch.from_numpy(text_lens),
            max(text_lens),
        )
    else:
        raise NotImplementedError


def reprocess_pr(data, idxs, mode="sup"):
    """
    Pad data and calculate length of data. Inference version has no text-related data.
    Args:
        mode: "sup" or "inference". "inference" mode is not done yet
    """
    ids = [data[idx]["id"] for idx in idxs]
    speakers = [data[idx]["speaker"] for idx in idxs]
    speakers = np.array(speakers)

    texts = [data[idx]["expanded_text"] for idx in idxs]
    raw_texts = [data[idx]["raw_text"] for idx in idxs]
    text_lens = np.array([text.shape[0] for text in texts])
    texts = pad_1D(texts)

    durations = [data[idx]["duration"] for idx in idxs]
    durations = pad_1D(durations)

    speaker_args = torch.from_numpy(speakers).long()

    if mode == "sup":
        return (
            ids,
            raw_texts,
            speaker_args,
            torch.from_numpy(texts).long(),
            torch.from_numpy(text_lens),
            max(text_lens),
            torch.from_numpy(durations).long(),
        )
    # elif mode == "inference":
    #     return (
    #         ids,
    #         raw_texts,
    #         speaker_args,
    #         torch.from_numpy(texts).long(),
    #         torch.from_numpy(text_lens),
    #         max(text_lens),
    #     )
    else:
        raise NotImplementedError
