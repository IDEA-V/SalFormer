import torch
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import re
from PIL import Image
import json
from torchvision import transforms
import numpy as np
from transformers import AutoImageProcessor

class ImagesWithSaliency1(Dataset):
    def __init__(self, img_folder, fixation_map_folder, heat_map_folder, img_transforms=None, fix_transform=None, hm_transform=None):
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
            self.datas.append([img, id, q0])
        
        self.img_transform = img_transforms

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        # idx = 0
        img, id, q  = self.datas[idx]
        # img = read_image(f"{self.img_folder}/{img}")
        img = Image.open(f"{self.img_folder}/{img}").convert("RGB")
        fixation0 = read_image(f"{self.fix_folder}/{id}_Q0.png")
        fixation1 = read_image(f"{self.fix_folder}/{id}_Q1.png")
        fixation = torch.where(fixation1>0, fixation1, fixation0)
        hm0 = read_image(f"{self.heat_map_folder}/{id}_Q0.png").float()
        hm1 = read_image(f"{self.heat_map_folder}/{id}_Q1.png").float()
        hm = hm0+hm1
        hm /= hm.max()
        
        if self.img_transform:
            img = self.img_transform(img)
        if self.fix_transform:
            fixation = self.fix_transform(fixation)
        if self.hm_transform:
            hm = self.hm_transform(hm)
        
        return img, q, fixation, hm, img

    def get_resized_imgs(self):
        imgs = []
        for i in range(len(self.datas)):
            img, fix, hm, q  = self.datas[i]
            img = Image.open(f"{self.img_folder}/{img}").convert("RGB")
            img = self.img_transform(img)
            imgs.append(img)
        return imgs