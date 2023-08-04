import torch
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import re
from PIL import Image

class ImagesWithSaliency(Dataset):
    def __init__(self, img_folder, fixation_map_folder, heat_map_folder, img_transform=None, hm_transform=None):
        self.img_transform = img_transform
        self.hm_transform = hm_transform
        self.datas = []
        self.img_folder = img_folder
        self.fix_folder = fixation_map_folder
        self.heat_map_folder = heat_map_folder

        imgs = os.listdir(img_folder)
        maps = os.listdir(heat_map_folder)
        for img in imgs:
            id = img.split(".")[0]
            for heat_map in maps:
                if heat_map.startswith(id):
                    self.datas.append([img, heat_map, heat_map, heat_map.replace(f"{id}_", "").replace(".png", "")])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        # idx = 0
        img, fix, hm, q  = self.datas[idx]
        # img = read_image(f"{self.img_folder}/{img}")
        img = Image.open(f"{self.img_folder}/{img}").convert("RGB")
        fixation = read_image(f"{self.fix_folder}/{fix}")
        hm = read_image(f"{self.heat_map_folder}/{hm}")
        
        # if self.img_transform:
        #     img = self.img_transform(img)
        if self.hm_transform:
            fixation = self.hm_transform(fixation)
            hm = self.hm_transform(hm)
        
        return img, q, fixation, hm
    
