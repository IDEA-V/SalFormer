import matplotlib.pyplot as plt
from PIL import Image
from torchviz import make_dot
import numpy as np

import torch
from torchvision import transforms
from torchvision.utils import save_image
from transformers import AutoImageProcessor, AutoTokenizer, BertModel, ViTConfig, ViTMAEModel, ViTModel, SwinModel

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from dataset import ImagesWithSaliency
from model import SalFormer
import random
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
# torch.set_default_device(device)
device = 'cuda'
number_epoch = 600
eps=1e-10

img_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Lambda(lambda x: x/255),
    transforms.Lambda(lambda x: x[:3]),
    transforms.Normalize(0.5,0.5)
])
fix_transform = transforms.Compose([
    transforms.Resize((128,128))
])
hm_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.Lambda(lambda x: x/255)
])

train_set = ImagesWithSaliency("./SalChartQA/train/raw_img/", "./SalChartQA/train/saliency_all/fix_maps/", "./SalChartQA/train/saliency_all/heatmaps/", fix_transform, hm_transform)
val_set = ImagesWithSaliency("./SalChartQA/val/raw_img/", "./SalChartQA/val/saliency_all/fix_maps/", "./SalChartQA/val/saliency_all/heatmaps/", fix_transform, hm_transform)
test_set = ImagesWithSaliency("./SalChartQA/test/raw_img/", "./SalChartQA/test/saliency_all/fix_maps/", "./SalChartQA/test/saliency_all/heatmaps/", fix_transform, hm_transform)

# image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
# image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window12-384")
# image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
# vit = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
# vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
# vit = SwinModel.from_pretrained("microsoft/swin-base-patch4-window12-384")


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

model = SalFormer(vit, bert).to(device)

def padding_fn(data):
    img, q, fix, hm = zip(*data)

    input_ids = tokenizer(q, return_tensors="pt", padding=True)

    return image_processor(img), input_ids, torch.stack(fix), torch.stack(hm)

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=padding_fn)
vali_dataloader = DataLoader(val_set, batch_size=64, shuffle=True, collate_fn=padding_fn)


normalize = transforms.Normalize(0, 1)
kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer =torch.optim.Adam(model.parameters(), lr=0.0001)
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

def log_softmax2d(x):
    logits = torch.log_softmax(x.flatten(), 0)
    return logits.reshape(x.shape)

def softmax2d(x):
    logits = torch.softmax(x.flatten(), 0)
    return logits.reshape(x.shape)

def nss_npy(gt_locs, predicted_map):
    assert gt_locs.shape == predicted_map.shape, \
    'dim missmatch in nss_npy: %s vs %s' % (gt_locs.shape, predicted_map.shape)
    predicted_map_norm = (predicted_map - np.mean(predicted_map))/np.std(predicted_map)
    dot = predicted_map_norm * gt_locs
    N = np.sum(gt_locs)
    ret = np.sum(dot)/N
    return ret

def inference(img, input_ids, fix, hm):
    img = img.convert_to_tensors('pt').to(device)
    input_ids = input_ids.to(device)
    fix = fix.to(device)
    hm = hm.to(device)

    # if batch == 0:
    #     writer.add_graph(model, [img['pixel_values'], input_ids['input_ids']])
    y = model(img, input_ids) 
    y_sum = y.view(y.shape[0], -1).sum(1, keepdim=True)
    y_distribution = y / (y_sum[:, :, None, None] + eps)

    hm_sum = hm.view(y.shape[0], -1).sum(1, keepdim=True)
    hm_distribution = hm / (hm_sum[:, :, None, None] + eps)
    hm_distribution = hm_distribution + eps
    hm_distribution = hm_distribution / (1+eps)

    if fix.sum() != 0:
        normal_y = (y-y.mean())/y.std()
        nss = torch.sum(normal_y*fix)/fix.sum()
    else:
        nss = torch.Tensor([0.0]).to(device)
    kl = kl_loss(torch.log(y_distribution), torch.log(hm_distribution))

    vy = y - torch.mean(y)
    vhm = hm - torch.mean(hm)  

    if (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vhm ** 2))) != 0:
        cc = torch.sum(vy * vhm) / (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vhm ** 2)))
    else: 
        cc = torch.Tensor([0.0]).to(device)

    return y, kl, cc, nss

n_iter = 0
n_test_iter = 0
for epoch in range(number_epoch):
    for batch, (img, input_ids, fix, hm) in enumerate(train_dataloader):

        y, kl, cc, nss = inference(img, input_ids, fix, hm)

        if torch.isnan(kl):
            kl = torch.Tensor([0.0]).to(device)
            print("kl is nan!")
        if torch.isnan(cc):
            cc = torch.Tensor([0.0]).to(device)
            print("cc is nan!")
        if torch.isnan(nss):
            nss = torch.Tensor([0.0]).to(device)
            print("nss is nan!")
        
        loss = 10*kl - cc - 2*nss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        for i in random.sample(range(1, y.shape[0]), 1):
            save_image(y[i], f'./results/train/epoch{epoch}_batch{batch}_{i}.png')
            # plt.imsave(f'./results/train/epoch{epoch}_batch{batch}_{i}.png', y[i, 0, :, :].squeeze().detach().cpu().numpy(), vmin=0.0, vmax=1.0, cmap='gray')
            save_image(hm[i], f'./results/train/epoch{epoch}_batch{batch}_{i}_truth.png')

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
            }, './model.tar')

            model.eval()
            test_loss = 0
            test_kl, test_cc, test_nss = 0,0,0 
            for batch, (img, input_ids, fix, hm) in enumerate(vali_dataloader):    
                with torch.no_grad():
                    y, kl, cc, nss = inference(img, input_ids, fix, hm)
                    loss = 10*kl - cc - 2*nss
                    test_loss += loss.item()/len(vali_dataloader)

                    if y.shape[0] == 64:
                        for i in random.sample(range(1, y.shape[0]), 3):
                            save_image(y[i], f'./results/test/epoch{epoch}_batch{batch}_{i}.png')
                            # plt.imsave(f'./results/test/epoch{epoch}_batch{batch}_{i}.png', y[i, 0, :, :].squeeze().detach().cpu().numpy(), vmin=0.0, vmax=1.0, cmap='gray')
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

# print("===================")
# make_dot(y.mean(), params=dict(model.named_parameters())).render("attached", format="png")
