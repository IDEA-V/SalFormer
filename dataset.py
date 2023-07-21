import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import re

class ImagesWithSaliency(Dataset):
    def __init__(self, img_folder, heat_map_folder, tokenizer, img_transform=None, hm_transform=None):
        self.img_transform = img_transform
        self.hm_transform = hm_transform
        self.datas = []
        self.img_folder = img_folder
        self.heat_map_folder = heat_map_folder
        self.tokenizer = tokenizer

        imgs = os.listdir(img_folder)
        maps = os.listdir(heat_map_folder)
        for img in imgs:
            id = img.split(".")[0]
            for heat_map in maps:
                if heat_map.startswith(id):
                    self.datas.append([img, heat_map, heat_map.replace(f"{id}_", "").replace(".png", "")])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        img, hm, q  = self.datas[idx]
        img = read_image(f"{self.img_folder}/{img}")
        hm = read_image(f"{self.heat_map_folder}/{hm}")
        
        if self.img_transform:
            img = self.img_transform(img)
        if self.hm_transform:
            hm = self.hm_transform(hm)
        
        return img, self.tokenizer(q, return_tensors="pt"), hm
