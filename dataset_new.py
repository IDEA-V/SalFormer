import torch
from torch.utils.data import Dataset
import numpy as np

class ImagesWithSaliency(Dataset):
    def __init__(self, npy_path, dtype):
        self.datas = []
        np_data = np.load(npy_path, allow_pickle = True)
        for i in range(len(np_data)):
            img, q, fixation, hm, fix = np_data[i]
            self.datas.append([img.type(dtype), q, fixation, hm.type(dtype), fix])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]
