from PIL import Image
from torchviz import make_dot

import torch
from torchvision import transforms
from torchvision.utils import save_image
from transformers import AutoImageProcessor, AutoTokenizer, BertModel, ViTConfig
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from dataset import ImagesWithSaliency
from model import SalFormer

# torch.set_default_device(device)
device = 'cuda'
number_epoch = 50


img_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Lambda(lambda x: x/255),
    transforms.Lambda(lambda x: x[:3])
])
hm_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Lambda(lambda x: x/255)
])

dataset = ImagesWithSaliency("./SaliencyChartQA/raw_img/", "./SaliencyChartQA/fix_maps/", "./SaliencyChartQA/heatmaps/", img_transform, hm_transform)


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.train()
config = ViTConfig()

model = SalFormer(bert, config).to(device)

input = dataset[0][:-1]

def padding_fn(data):
    img, q, fix, hm = zip(*data)

    input_ids = tokenizer(q, return_tensors="pt", padding=True)

    return torch.stack(img), input_ids, torch.stack(fix), torch.stack(hm)

train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=padding_fn)

normalize = transforms.Normalize(0, 1)
kl_loss = torch.nn.KLDivLoss(reduction="mean", log_target=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train(True)

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

def log_softmax2d(x):
    logits = torch.log_softmax(x.flatten(), 0)
    return logits.reshape(x.shape)

def softmax2d(x):
    logits = torch.softmax(x.flatten(), 0)
    return logits.reshape(x.shape)

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
for epoch in range(number_epoch):
    for batch, (img, input_ids, fix, hm) in enumerate(train_dataloader):
        img = img.to(device)
        input_ids = input_ids.to(device)
        fix = fix.to(device)
        hm = hm.to(device)

        # y = model(image_processor(img, return_tensors="pt"), input_ids)
        y = model(img, input_ids)

        y = y - torch.min(y)
        y = y / torch.max(y)
        y = y + 0.0001
        y = y / 1.0001

        hm = hm + 0.0001
        hm = hm/ 1.0001

        nss = torch.sum(normalize(y)*fix)
        kl = kl_loss(torch.log(torch.stack([y, 1.0001-y])), torch.log(torch.stack([hm, 1.0001-hm])))

        vy = y - torch.mean(y)
        vhm = hm - torch.mean(hm)
        cc = torch.sum(vy * vhm) / (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vhm ** 2)))

        if epoch >= 0:
            for i in range(3):
                save_image(y[i], f'./results/epoch{epoch}_batch{batch}_{i}.png')
                save_image(hm[i], f'./results/epoch{epoch}_batch{batch}_{i}_truth.png')

        print("kl ", kl, "cc ", cc, "nss ", nss)
        loss = 60*kl - cc - 2*nss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}/10: ", loss.item())
        optimizer.zero_grad()

# print("===================")
# make_dot(y.mean(), params=dict(model.named_parameters())).render("attached", format="png")
