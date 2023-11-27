import torch
from torch.utils.data import Dataset
import numpy as np

class ImagesWithSaliency(Dataset):
    def __init__(self, npy_path, dtype=None):
        self.dtype = dtype
        self.datas = np.load(npy_path, allow_pickle = True)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        if self.dtype:
            self.datas[idx][0] = self.datas[idx][0].type(self.dtype)
            self.datas[idx][3] = self.datas[idx][3].type(self.dtype)

        return self.datas[idx]
