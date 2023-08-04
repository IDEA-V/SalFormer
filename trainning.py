import matplotlib.pyplot as plt
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
import random

# torch.set_default_device(device)
device = 'cuda'
number_epoch = 150
eps=1e-10

img_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Lambda(lambda x: x/255),
    transforms.Lambda(lambda x: x[:3]),
    transforms.Normalize(0.5,0.5)
])
hm_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.Lambda(lambda x: x/255),
    # transforms.Lambda(lambda x: 0.0001 if x== 0 else x)
])

dataset = ImagesWithSaliency("./SaliencyChartQA/raw_img/", "./SaliencyChartQA/fix_maps/", "./SaliencyChartQA/heatmaps/",img_transform, hm_transform)


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

model = SalFormer(bert).to(device)

input = dataset[0][:-1]

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
def padding_fn(data):
    img, q, fix, hm = zip(*data)

    input_ids = tokenizer(q, return_tensors="pt", padding=True)

    return image_processor(img), input_ids, torch.stack(fix), torch.stack(hm)

train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=padding_fn)

normalize = transforms.Normalize(0, 1)
kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer =torch.optim.Adam(model.parameters(), lr=0.0001)

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
bert.eval()
for epoch in range(number_epoch):
    for batch, (img, input_ids, fix, hm) in enumerate(train_dataloader):
        img = img.convert_to_tensors('pt').to(device)
        input_ids = input_ids.to(device)
        fix = fix.to(device)
        hm = hm.to(device)

        # y = model(image_processor(img, return_tensors="pt"), input_ids)
        y = model(img, input_ids)
        y_sum = y.view(y.shape[0], -1).sum(1, keepdim=True)
        y_distribution = y / (y_sum[:, :, None, None] + eps)

        hm_sum = hm.view(y.shape[0], -1).sum(1, keepdim=True)
        hm_distribution = hm / (hm_sum[:, :, None, None] + eps)
        hm_distribution = hm_distribution + eps
        hm_distribution = hm_distribution / (1+eps)

        nss = torch.sum(normalize(y)*fix)
        kl = kl_loss(torch.log(y_distribution), torch.log(hm_distribution))

        vy = y - torch.mean(y)
        vhm = hm - torch.mean(hm)
        cc = torch.sum(vy * vhm) / (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vhm ** 2)))

        if epoch >= 0:
            for i in random.sample(range(1, y.shape[0]), 3):
                # save_image(y[i], f'./results/epoch{epoch}_batch{batch}_{i}.png')
                plt.imsave(f'./results/epoch{epoch}_batch{batch}_{i}.png', y[i, 0, :, :].squeeze().detach().cpu().numpy(), vmin=0.0, vmax=1.0, cmap='gray')
                save_image(hm[i], f'./results/epoch{epoch}_batch{batch}_{i}_truth.png')

        loss = 2*kl - cc - 2*(nss-10)

        # loss = -nss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch == len(train_dataloader)-1:
            print("kl ", kl.item(), "cc ", cc.item(), "nss ", nss.item())
            print(f"Epoch {epoch}/{number_epoch} batch {batch}: ", loss.item())

# print("===================")
# make_dot(y.mean(), params=dict(model.named_parameters())).render("attached", format="png")
