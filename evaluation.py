import torch
from torch.utils.data import DataLoader

import os
os.environ['TORCH_HOME'] = '/projects/wang/.cache/torch'
os.environ['TRANSFORMERS_CACHE'] = '/projects/wang/.cache'

from transformers import AutoTokenizer, BertModel, RobertaModel, ViTModel, SwinModel
from model_swin import SalFormer
# from dataset import ImagesWithSaliency
from dataset_new import ImagesWithSaliency
from torchvision import transforms
from torchvision.utils import save_image
from tokenizer_bert import padding_fn

from pathlib import Path


device = 'cuda:3'
eps=1e-10

test_set = ImagesWithSaliency("data/test.npy")

Path('./eval_results').mkdir(parents=True, exist_ok=True)

# vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
# vit = timm.create_model('xception41p.ra3_in1k', pretrained=True)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# bert = RobertaModel.from_pretrained("roberta-base")

model = SalFormer(vit, bert).to(device)
checkpoint = torch.load('./ckpt/model_bert_10kl_1cc_2nss_174.tar')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


test_dataloader = DataLoader(test_set, batch_size=16, shuffle=True, collate_fn=padding_fn, num_workers=8)
kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

test_kl, test_cc, test_nss = 0,0,0 
for batch, (img, input_ids, fix, hm, name) in enumerate(test_dataloader):
    img = img.to(device)
    input_ids = input_ids.to(device)
    fix = fix.to(device)
    hm = hm.to(device)

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
    
    test_kl += kl.item()/len(test_dataloader)
    test_cc += cc.item()/len(test_dataloader)
    test_nss += nss.item()/len(test_dataloader)

    for i in range(0, y.shape[0]):
        save_image(y[i], f"./eval_results/{name[i]}")

print("kl:", test_kl, "cc", test_cc, "nss", test_nss)
