import random
random.seed(42)
import numpy as np

import os
os.environ['TORCH_HOME'] = '/projects/wang/.cache/torch'
os.environ['TRANSFORMERS_CACHE'] = '/projects/wang/.cache'

my_variable = os.environ.get('TORCH_HOME')

import torch
from torchvision import transforms
from torchvision.utils import save_image
from transformers import SwinModel, BloomModel, LlamaModel
from torch.utils.data import DataLoader
from utils import padding_fn, inference

from dataset_new import ImagesWithSaliency

from model_llama import SalFormer

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = 'cuda:1'
number_epoch = 300
eps=1e-6
batch_size = 16


# vit = timm.create_model('xception41p.ra3_in1k', pretrained=True)
# data_config = timm.data.resolve_model_data_config(vit)
# img_transform_no_augment = timm.data.create_transform(**data_config, is_training=True)

train_set = ImagesWithSaliency("data/train.npy", dtype=torch.float32)
val_set = ImagesWithSaliency("data/val.npy", dtype=torch.float32)
test_set = ImagesWithSaliency("data/test.npy", dtype=torch.float32)

# vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
# vit = CvtModel.from_pretrained("microsoft/cvt-13")
# vit = SwinModel.from_pretrained("microsoft/swin-base-patch4-window12-384")
print('SwinModel loaded')

# bert = BertModel.from_pretrained("bert-base-uncased")
# bert = RobertaModel.from_pretrained("roberta-base")

llama = LlamaModel.from_pretrained("Enoch/llama-7b-hf", low_cpu_mem_usage=True)
# # llama = LlamaModel.from_pretrained("Enoch/llama-7b-hf")
print("llama loaded")

# bloom = BloomModel.from_pretrained("bigscience/bloom-3b")
# print('BloomModel loaded')

for param in llama.parameters(): 
    param.requires_grad = False


model = SalFormer(vit, llama).to(device)

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=padding_fn)
vali_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=padding_fn)

optimizer =torch.optim.Adam(model.parameters(), lr=0.00002, weight_decay=0.0001)


n_iter = 0

for epoch in range(number_epoch):
    for batch, (img, input_ids, fix, hm, name) in enumerate(train_dataloader):
        optimizer.zero_grad()

        y, kl, cc, nss = inference(model, device, eps, img, input_ids, fix, hm)

        if torch.isnan(kl):
            print(np.mean([ p.norm().cpu().detach().numpy() for p in model.parameters()]))
            print(kl)
            kl = torch.Tensor([0.0]).to(device)
            print("kl is nan!")
        if torch.isnan(cc):
            print(np.mean([ p.norm().cpu().detach().numpy() for p in model.parameters()]))
            print(cc)
            cc = torch.Tensor([0.0]).to(device)
            print("cc is nan!")
        if torch.isnan(nss):
            print(np.mean([ p.norm().cpu().detach().numpy() for p in model.parameters()]))
            print(nss)
            nss = torch.Tensor([0.0]).to(device)
            print("nss is nan!")

        loss = 10*kl - cc - 2*nss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if batch == len(train_dataloader) - 2:
            for i in random.sample(range(0, y.shape[0]), 1):
                save_image(y[i].type(torch.float32), f'./results_llm/train/epoch{epoch}_batch{batch}_{i}.png')
                save_image(hm[i].type(torch.float32), f'./results_llm/train/epoch{epoch}_batch{batch}_{i}_truth.png')

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
                }, f'./ckpt/model_llama_{epoch}.tar')

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
                            save_image(y[i].type(torch.float32), f'./results_llm/test/epoch{epoch}_batch{batch}_{i}.png')
                            save_image(hm[i].type(torch.float32), f'./results_llm/test/epoch{epoch}_batch{batch}_{i}_truth.png')

                    test_kl += kl.item()/len(vali_dataloader)
                    test_cc += cc.item()/len(vali_dataloader)
                    test_nss += nss.item()/len(vali_dataloader)
            model.train(True)
            print("Testing: loss ", test_loss, "kl ", test_kl, "cc ", test_cc, "nss ", test_nss)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Loss/test/kl', test_kl, epoch)
            writer.add_scalar('Loss/test/cc', test_cc, epoch)
            writer.add_scalar('Loss/test/nss', test_nss, epoch)
        # scheduler.step()
        n_iter += 1
