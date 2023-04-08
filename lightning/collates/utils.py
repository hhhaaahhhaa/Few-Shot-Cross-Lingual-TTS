import numpy as np
import torch

from text.define import LANG_NAME2ID
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
    lang_ids = [data[idx]["lang_id"] for idx in idxs]
    speakers = np.array(speakers)
    lang_ids = np.array(lang_ids)

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

    # print(len(ids))
    # print(len(raw_texts))
    # print(texts.shape)
    # print(len(text_lens))
    # print(mels.shape)
    # print(pitches.shape)
    # print(energies.shape)
    # print(durations.shape)

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
            torch.from_numpy(lang_ids).long(),
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
            torch.from_numpy(lang_ids).long(),
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

    texts = [data[idx]["text"] for idx in idxs]
    expanded_texts = [data[idx]["expanded_text"] for idx in idxs]
    raw_texts = [data[idx]["raw_text"] for idx in idxs]
    text_lens = np.array([text.shape[0] for text in texts])
    expanded_text_lens = np.array([expanded_text.shape[0] for expanded_text in expanded_texts])
    texts = pad_1D(texts)
    expanded_texts = pad_1D(expanded_texts)

    durations = [data[idx]["duration"] for idx in idxs]
    durations = pad_1D(durations)

    speaker_args = torch.from_numpy(speakers).long()

    if mode == "sup":
        return (
            ids,
            raw_texts,
            speaker_args,
            torch.from_numpy(expanded_texts).long(),
            torch.from_numpy(expanded_text_lens),
            max(expanded_text_lens),
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


def reprocess_bd(data, idxs):
    ids = [data[idx]["id"] for idx in idxs]

    durations = [data[idx]["duration"] for idx in idxs]
    durations = pad_1D(durations)
    boundaries = [data[idx]["boundary"] for idx in idxs]
    boundaries = pad_1D(boundaries)

    return (
        ids,
        torch.from_numpy(durations).long(),
        torch.from_numpy(boundaries).float(),
    )


def reprocess_bd2(data, idxs):
    ids = [data[idx]["id"] for idx in idxs]

    mels = [data[idx]["mel"] for idx in idxs]
    durations = [data[idx]["duration"] for idx in idxs]
    boundaries = durations2boundaries(durations)
    mel_lens = np.array([mel.shape[0] for mel in mels])
    texts = [data[idx]["text"] for idx in idxs]
    text_lens = np.array([text.shape[0] for text in texts])
    texts = pad_1D(texts)

    mels = pad_2D(mels)

    segs = []
    for duration in durations:
        seg = [0]
        pos = 0
        for d in duration:
            pos += d
            if d > 0:
                seg.append(pos)
        segs.append(seg)

    return (
        ids,
        torch.from_numpy(texts).long(),
        torch.from_numpy(text_lens),
        max(text_lens),
        torch.from_numpy(mels).float(),
        torch.from_numpy(mel_lens),
        max(mel_lens),
        segs,
        boundaries,
    )


def durations2boundaries(durations) -> torch.FloatTensor:
    boundaries = []
    for duration in durations:
        if isinstance(duration, torch.Tensor):
            duration = duration.tolist()
        pos = 0
        boundary = np.zeros(sum(duration)) 
        for d in duration:
            pos += d
            if d > 0:
                boundary[pos - 1] = 1
        boundaries.append(boundary)
    boundaries = pad_1D(boundaries)
    return torch.from_numpy(boundaries).float()
