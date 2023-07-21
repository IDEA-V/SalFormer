from PIL import Image
from torchviz import make_dot

import torch
from torchvision import transforms
from transformers import AutoTokenizer, BertModel, ViTConfig

from dataset import ImagesWithSaliency
from model import SalFormer

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Lambda(lambda x: x/255),
    transforms.Lambda(lambda x: x[:-1])
    ])
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = ImagesWithSaliency("./SaliencyChartQA/raw_img/", "./SaliencyChartQA/heatmaps/", tokenizer, transform, transform)


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

config = ViTConfig()

model = SalFormer(bert, config)

input = dataset[0][:-1]
y = model(torch.unsqueeze(input[0], 0), input[1])

print("===================")
make_dot(y.mean(), params=dict(model.named_parameters())).render("attached", format="png")