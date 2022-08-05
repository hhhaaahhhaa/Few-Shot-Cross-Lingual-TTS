from typing import List
from torch.utils.data import ConcatDataset, Sampler, Dataset
import random

from dlhlp_lib.utils import batchify


class MultiTaskSampler(Sampler):
    """
    Sample from ConcatDataset such that every batch is from the same dataset.
    If n_batches are specified (usually for training stage), perform random sampling.
    If not (usually for validation / test stage), simply return data batches in order so that every sample occurs once.
    """
    def __init__(self, data_source: ConcatDataset, batch_size: int, n_batches: int=None) -> None:
        super().__init__(data_source=data_source)
        self.dataset = data_source
        self.batch_size = batch_size
        self.number_of_datasets = len(data_source.datasets)
        self.n_batches = n_batches

        self.group_indices = []
        total = 0
        for d in data_source.datasets:
            l = len(d)
            self.group_indices.append(list(range(total, total + l)))
            total += l

        # Map style
        self.batches = []
        for i in range(self.number_of_datasets):
            self.batches.extend(list(batchify(self.group_indices[i], batch_size=self.batch_size)))
    
    def __iter__(self):
        if self.n_batches is not None:
            return RandomMultiTaskIterator(self.number_of_datasets, self.group_indices, self.batch_size, self.n_batches)
        else:
            return iter(self.batches)


class RandomMultiTaskIterator(object):
    def __init__(self, n_tasks: int, task_indices: List[List[int]], batch_size: int, total_len: int=-1) -> None:
        self.n_tasks = n_tasks
        self.task_indices = task_indices
        self.batch_size = batch_size
        self.__cnt = 0
        self.__total_len = total_len
        self.__weights = [len(idxs) for idxs in task_indices]

    def __iter__(self):
        return self

    def __next__(self):
        if self.__cnt >= self.__total_len and self.__total_len > 0:
            raise StopIteration
        task_idx = random.choices(list(range(self.n_tasks)), weights=self.__weights, k=1)[0]
        res = random.choices(self.task_indices[task_idx], k=self.batch_size)
        self.__cnt += 1

        return res


class CustomSamplerDataset(Dataset):
    def __init__(self, dataset, sampler) -> None:
        super().__init__()
        self.dataset = dataset
        self.sampler = sampler

    def __len__(self):
        return len(self.sampler.batches)

    def __getitem__(self, idx):
        return [self.dataset[id] for id in self.sampler.batches[idx]]
