import torch
from torch.utils.data import Dataset
import numpy as np

class ImagesWithSaliency(Dataset):
    def __init__(self, npy_path):
        self.datas = np.load(npy_path, allow_pickle = True)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]
