import os
import json
import random

from torch.utils.data import ConcatDataset
from learn2learn.data import MetaDataset, TaskDataset
from learn2learn.data.transforms import FusedNWaysKShots, LoadData
from learn2learn.data.task_dataset import DataDescription
from learn2learn.utils.lightning import EpisodicBatcher


def load_descriptions(tasks, filename):
    with open(filename, 'r') as f:
        loaded_descriptions = json.load(f)
    assert len(tasks.datasets) == len(loaded_descriptions), "TaskDataset count mismatch"

    for i, _tasks in enumerate(tasks.datasets):
        descriptions = loaded_descriptions[i]
        assert len(descriptions) == _tasks.num_tasks, "num_tasks mismatch"
        for j in descriptions:
            data_descriptions = [DataDescription(index) for index in descriptions[j]]
            task_descriptions = _tasks.task_transforms[-1](data_descriptions)
            _tasks.sampled_descriptions[int(j)] = task_descriptions


def write_descriptions(tasks, filename):
    descriptions = []
    for ds in tasks.datasets:
        data_descriptions = {}
        for i in ds.sampled_descriptions:
            data_descriptions[i] = [desc.index for desc in ds.sampled_descriptions[i]]
        descriptions.append(data_descriptions)

    with open(filename, 'w') as f:
        json.dump(descriptions, f, indent=4)


def load_SQids2Tid(SQids_filename, tag):
    with open(SQids_filename, 'r') as f:
        SQids = json.load(f)
    SQids2Tid = {}
    for i, SQids_dict in enumerate(SQids):
        sup_ids, qry_ids = SQids_dict['sup_id'], SQids_dict['qry_id']
        SQids2Tid[f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"] = f"{tag}_{i:03d}"
    return SQids, SQids2Tid


def get_SQids2Tid(tasks, tag):
    SQids = []
    SQids2Tid = {}
    for i, task in enumerate(tasks):
        sup_ids, qry_ids = task[0][0][0], task[1][0][0]
        SQids.append({'sup_id': sup_ids, 'qry_id': qry_ids})
        SQids2Tid[f"{'-'.join(sup_ids)}.{'-'.join(qry_ids)}"] = f"{tag}_{i:03d}"
    return SQids, SQids2Tid


def prefetch_tasks(tasks, tag='val', log_dir=''):
    if (os.path.exists(os.path.join(log_dir, f'{tag}_descriptions.json'))
            and os.path.exists(os.path.join(log_dir, f'{tag}_SQids.json'))):
        # print(os.path.join(log_dir, f'{tag}_descriptions.json'))
        # print("heyheyhey2-1, why the path exist??")
        # Recover descriptions
        load_descriptions(tasks, os.path.join(log_dir, f'{tag}_descriptions.json'))
        SQids, SQids2Tid = load_SQids2Tid(os.path.join(log_dir, f'{tag}_SQids.json'), tag)

    else:
        os.makedirs(log_dir, exist_ok=True)

        # Run through tasks to get descriptions
        SQids, SQids2Tid = get_SQids2Tid(tasks, tag)
        with open(os.path.join(log_dir, f"{tag}_SQids.json"), 'w') as f:
            json.dump(SQids, f, indent=4)
        write_descriptions(tasks, os.path.join(log_dir, f"{tag}_descriptions.json"))

    return SQids2Tid


def get_multispeaker_id2lb(datasets):
    id2lb = {}
    total = 0
    for dataset in datasets:
        l = len(dataset)
        id2lb.update({k: f"corpus_{dataset.lang_id}-spk_{dataset.speaker[k - total]}"
                     for k in range(total, total + l)})
        total += l

    return id2lb


def get_multilingual_id2lb(datasets):
    id2lb = {}
    total = 0
    for dataset in datasets:
        l = len(dataset)
        id2lb.update({k: dataset.lang_id for k in range(total, total + l)})
        total += l

    return id2lb


class EpisodicInfiniteWrapper:
    def __init__(self, dataset, epoch_length, weights=None):
        self.dataset = dataset
        self.epoch_length = epoch_length
        self.idxs = list(range(len(self.dataset)))
        self.ws = [1] * len(self.dataset)
        if weights is not None:
            assert isinstance(weights, list) and len(weights) == len(dataset)
            self.ws = weights

    def __getitem__(self, idx):
        idx = random.choices(self.idxs, weights=self.ws, k=1)[0]
        return self.dataset[idx]

    def __len__(self):
        return self.epoch_length
