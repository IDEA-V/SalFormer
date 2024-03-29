import random
random.seed(42)
from env import *

import torch
from torchvision.utils import save_image
from transformers import BertModel, SwinModel
from torch.utils.data import DataLoader
from dataset_new import ImagesWithSaliency
from model_swin import SalFormer

from torch.utils.tensorboard import SummaryWriter
from utils import padding_fn, inference
from pathlib import Path

writer = SummaryWriter()

device = 'cuda:0'
number_epoch = 200
eps=1e-10
batch_size = 32

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224), antialias=True),
    transforms.Lambda(lambda x: x[:3]),
    transforms.Normalize([0.8801, 0.8827, 0.8840], [0.2523, 0.2321, 0.2400]),
    transforms.RandomPerspective()
])

img_transform_no_augment = transforms.Compose([
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

# vit = timm.create_model('xception41p.ra3_in1k', pretrained=True)
# data_config = timm.data.resolve_model_data_config(vit)
# img_transform_no_augment = timm.data.create_transform(**data_config, is_training=True)


# dataset_path = './SalChartQA'
dataset_path = '/datasets/internal/datasets_wang/SalChartQA/SalChartQA-split'

train_set = ImagesWithSaliency("data/train.npy")
val_set = ImagesWithSaliency("data/val.npy")
test_set = ImagesWithSaliency("data/test.npy")

# vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
# vit = SwinModel.from_pretrained("microsoft/swin-base-patch4-window12-384")

bert = BertModel.from_pretrained("bert-base-uncased")

for param in bert.parameters(): 
    param.requires_grad = False

model = SalFormer(vit, bert).to(device)

Path('./results/train').mkdir(parents=True, exist_ok=True)
Path('./results/val').mkdir(parents=True, exist_ok=True)
Path('./results/test').mkdir(parents=True, exist_ok=True)

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=padding_fn, num_workers=8)
vali_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=padding_fn, num_workers=8)



optimizer = torch.optim.Adam(model.parameters(), lr=0.00006, weight_decay=0.0001)


n_iter = 0
for epoch in range(number_epoch):
    for batch, (img, input_ids, fix, hm, name) in enumerate(train_dataloader):

        y, kl, cc, nss = inference(model, device, eps, img, input_ids, fix, hm)

        if torch.isnan(kl):
            kl = torch.Tensor([0.0]).to(device)
            print(max([p.norm() for p in model.parameters()]))
            print("kl is nan!")
        if torch.isnan(cc):
            cc = torch.Tensor([0.0]).to(device)
            print(max([p.norm() for p in model.parameters()]))
            print("cc is nan!")
        if torch.isnan(nss):
            nss = torch.Tensor([0.0]).to(device)
            print(max([p.norm() for p in model.parameters()]))
            print("nss is nan!")
        
        loss = 10*kl - cc - 2*nss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch == len(train_dataloader) - 2:
            for i in random.sample(range(0, y.shape[0]), 1):
                save_image(y[i], f'./results/train/epoch{epoch}_batch{batch}_{i}.png')
                save_image(hm[i], f'./results/train/epoch{epoch}_batch{batch}_{i}_truth.png')

        writer.add_scalar('Loss/train', loss.item(), n_iter)
        writer.add_scalar('Loss/train/kl', kl.item(), n_iter)
        writer.add_scalar('Loss/train/cc', cc.item(), n_iter)
        writer.add_scalar('Loss/train/nss', nss.item(), n_iter)

        if batch == len(train_dataloader)-1:
            print(f"Epoch {epoch}/{number_epoch} batch {batch}: ")
            print("Training: loss ", loss.item(), "kl ", kl.item(), "cc ", cc.item(), "nss ", nss.item())
            if epoch % 3 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, f'./ckpt/model_new_{epoch}.tar')


            model.eval()
            test_loss = 0
            test_kl, test_cc, test_nss = 0,0,0 
            for batch, (img, input_ids, fix, hm, name) in enumerate(vali_dataloader):    
                with torch.no_grad():
                    y, kl, cc, nss = inference(model, device, eps, img, input_ids, fix, hm)
                    loss = 10*kl - cc - 2*nss
                    test_loss += loss.item()/len(vali_dataloader)

                    if y.shape[0] == batch_size:
                        for i in random.sample(range(0, y.shape[0]), 3):
                            save_image(y[i], f'./results/test/epoch{epoch}_batch{batch}_{i}.png')
                            save_image(hm[i], f'./results/test/epoch{epoch}_batch{batch}_{i}_truth.png')
                    
                    test_kl += kl.item()/len(vali_dataloader)
                    test_cc += cc.item()/len(vali_dataloader)
                    test_nss += nss.item()/len(vali_dataloader)
            model.train(True)
            print("Testing: loss ", test_loss, "kl ", test_kl, "cc ", test_cc, "nss ", test_nss)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Loss/test/kl', test_kl, epoch)
            writer.add_scalar('Loss/test/cc', test_cc, epoch)
            writer.add_scalar('Loss/test/nss', test_nss, epoch)
        n_iter += 1
