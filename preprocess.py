import random
random.seed(666)
from env import *
from torchvision import transforms
import json
from PIL import Image
import numpy as np
from torchvision.io import read_image
from tqdm import tqdm

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224), antialias=True),
    transforms.Lambda(lambda x: x[:3]),
    transforms.Normalize([0.8801, 0.8827, 0.8840], [0.2523, 0.2321, 0.2400]),
    transforms.RandomPerspective()
])

img_transform_no_augment = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224), antialias=True),
    transforms.Lambda(lambda x: x[:3]),
    transforms.Normalize([0.8801, 0.8827, 0.8840], [0.2523, 0.2321, 0.2400])
])

fix_transform = transforms.Compose([
    transforms.Resize((128,128), antialias=None)
])
hm_transform = transforms.Compose([
    transforms.Resize((128,128), antialias=None),
    transforms.Lambda(lambda x: x/255)
])



def preprocess_dataset(img_folder, fixation_map_folder, heat_map_folder):
    datas = []
    imgs = os.listdir(img_folder)
    qa = json.load(open(f'{dataset_path}/image_questions.json'))
    for i in tqdm(range(len(imgs))):
        img_name = imgs[i]
        id = img_name.split(".")[0]

        q0 = qa[img_name]['Q0']
        img = img_transform_no_augment(Image.open(f"{img_folder}/{img_name}").convert("RGB"))
        fixation = fix_transform(read_image(f"{fixation_map_folder}/{id}_Q0.png"))
        hm = hm_transform(read_image(f"{heat_map_folder}/{id}_Q0.png"))
        datas.append([img, q0, fixation, hm, f"{id}_Q0.png"])

        q1 = qa[img_name]['Q1']
        img = img_transform_no_augment(Image.open(f"{img_folder}/{img_name}").convert("RGB"))
        fixation = fix_transform(read_image(f"{fixation_map_folder}/{id}_Q1.png"))
        hm = hm_transform(read_image(f"{heat_map_folder}/{id}_Q1.png"))
        datas.append([img, q1, fixation, hm, f"{id}_Q1.png"])
    np.save(f"data/{img_folder.split('/')[-3]}.npy", np.array(datas, dtype=object))

if __name__ == '__main__':
    # dataset_path = './SalChartQA'
    dataset_path = '/datasets/internal/datasets_wang/SalChartQA/SalChartQA-split'

    preprocess_dataset(f"{dataset_path}/train/raw_img/", f"{dataset_path}/train/saliency_all/fix_maps/", f"{dataset_path}/train/saliency_all/heatmaps/")
    preprocess_dataset(f"{dataset_path}/val/raw_img/", f"{dataset_path}/val/saliency_all/fix_maps/", f"{dataset_path}/val/saliency_all/heatmaps/")
    preprocess_dataset(f"{dataset_path}/test/raw_img/", f"{dataset_path}/test/saliency_all/fix_maps/", f"{dataset_path}/test/saliency_all/heatmaps/")
