import torch
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import re
from PIL import Image
import json

class ImagesWithSaliency(Dataset):
    def __init__(self, img_folder, fixation_map_folder, heat_map_folder, fix_transform=None, hm_transform=None):
        self.fix_transform = fix_transform
        self.hm_transform = hm_transform
        self.datas = []
        self.img_folder = img_folder
        self.fix_folder = fixation_map_folder
        self.heat_map_folder = heat_map_folder

        imgs = os.listdir(img_folder)
        maps = os.listdir(heat_map_folder)
        qa = json.load(open('./SalChartQA/image_questions.json'))
        for img in imgs:
            id = img.split(".")[0]
            q0 = qa[img]['Q0']
            self.datas.append([img, f"{id}_Q0.png", f"{id}_Q0.png", q0])
            q1 = qa[img]['Q1']
            self.datas.append([img, f"{id}_Q1.png", f"{id}_Q1.png", q1])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        # idx = 0
        img, fix, hm, q  = self.datas[idx]
        # img = read_image(f"{self.img_folder}/{img}")
        img = Image.open(f"{self.img_folder}/{img}").convert("RGB")
        fixation = read_image(f"{self.fix_folder}/{fix}")
        hm = read_image(f"{self.heat_map_folder}/{hm}")
        
        if self.fix_transform:
            fixation = self.fix_transform(fixation)
        if self.hm_transform:
            hm = self.hm_transform(hm)
        
        return img, q, fixation, hm