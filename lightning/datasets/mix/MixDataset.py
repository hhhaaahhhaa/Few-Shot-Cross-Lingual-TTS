from torch.utils.data import Dataset

from ..language.FSCLDataset import UnitFSCLDataset
from ..t2u.t2udataset import T2UDataset


class T2U2SDataset(Dataset):
    """
    T2UDataset + UnitFSCLDataset
    """
    def __init__(self, t2u_args, t2u_kwargs, u2s_args, u2s_kwargs) -> None:
        super().__init__()
        self.t2u_dataset = T2UDataset(*t2u_args, **t2u_kwargs)
        self.u2s_dataset = UnitFSCLDataset(*u2s_args, **u2s_kwargs)

    def __len__(self):
        return len(self.t2u_dataset)
    
    def __getitem__(self, index):
        return {
            "t2u": self.t2u_dataset.__getitem__(index),
            "u2s": self.u2s_dataset.__getitem__(index)
        }
