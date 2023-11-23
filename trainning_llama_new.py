import random
random.seed(666)

import os
os.environ['TORCH_HOME'] = '/projects/wang/.cache/torch'
os.environ['TRANSFORMERS_CACHE'] = '/projects/wang/.cache'

import torch
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.utils import save_image
from transformers import AutoTokenizer, SwinModel, BloomModel
from torch.utils.data import DataLoader

from dataset_new import ImagesWithSaliency
from model_llama import SalFormer

from utils import padding_fn, inference
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


device = 'cuda:1'
number_epoch = 200
eps=1e-6
batch_size = 32
# torch.set_default_dtype(torch.float16)


train_set = ImagesWithSaliency("data/train.npy")
val_set = ImagesWithSaliency("data/val.npy")
test_set = ImagesWithSaliency("data/test.npy")

# vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
# vit = CvtModel.from_pretrained("microsoft/cvt-13")
# vit = SwinModel.from_pretrained("microsoft/swin-base-patch4-window12-384")
print('SwinModel loaded')

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# bert = BertModel.from_pretrained("bert-base-uncased")
# bert = RobertaModel.from_pretrained("roberta-base")

# tokenizer = LlamaTokenizer.from_pretrained("Enoch/llama-7b-hf")
# tokenizer.pad_token = tokenizer.eos_token
# llama = LlamaModel.from_pretrained("Enoch/llama-7b-hf", torch_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b")
print('tokenizer loaded')
bloom = BloomModel.from_pretrained("bigscience/bloom-3b")
print('BloomModel loaded')

model = SalFormer(vit, bloom).to(device)
Path('./results/train').mkdir(parents=True, exist_ok=True)
Path('./results/val').mkdir(parents=True, exist_ok=True)
Path('./results/test').mkdir(parents=True, exist_ok=True)

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=padding_fn)
vali_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=padding_fn)



optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.1, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.6)
#optimizer =torch.optim.Adam(model.parameters(), lr=0.00006, weight_decay=0.0001)


n_iter = 0
n_test_iter = 0
for epoch in range(number_epoch):
    for batch, (img, input_ids, fix, hm) in enumerate(train_dataloader):
        optimizer.zero_grad()

        y, kl, cc, nss = inference(model, device, eps, img, input_ids, fix, hm)

        if torch.isnan(kl):
            kl = torch.Tensor([0.0]).to(device)
            print(max([ p.norm() for p in model.parameters()]))
            print("kl is nan!")
        if torch.isnan(cc):
            cc = torch.Tensor([0.0]).to(device)
            print(max([ p.norm() for p in model.parameters()]))
            print("cc is nan!")
        if torch.isnan(nss):
            nss = torch.Tensor([0.0]).to(device)
            print(max([ p.norm() for p in model.parameters()]))
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, './model_llama.tar')

            model.eval()
            test_loss = 0
            test_kl, test_cc, test_nss = 0,0,0 
            for batch, (img, input_ids, fix, hm) in enumerate(vali_dataloader):    
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
        scheduler.step()
        n_iter += 1
