import torch
from torch.utils.data import DataLoader
from env import *

from transformers import SwinModel, BloomModel
from model_llama import SalFormer
from dataset_new import ImagesWithSaliency
from torchvision.utils import save_image
from tokenizer_bloom import padding_fn
from pathlib import Path
from tqdm import tqdm

device = 'cuda:4'
eps=1e-10

test_set = ImagesWithSaliency("data/test.npy", dtype=torch.float32)

output_path = './eval_results'
Path(output_path).mkdir(parents=True, exist_ok=True)
neuron_n = 2560

vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

llm = BloomModel.from_pretrained("bigscience/bloom-3b")
for param in llm.parameters(): 
    param.requires_grad = False


model = SalFormer(vit, llm, neuron_n=neuron_n).to(device)
checkpoint = torch.load('./ckpt/model_bloom_freeze_10kl_5cc_2nss.tar')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

test_dataloader = DataLoader(test_set, batch_size=16, shuffle=False, collate_fn=padding_fn, num_workers=4)
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

    for i in tqdm(range(0, y.shape[0])):
        save_image(y[i], f"{output_path}/{name[i]}")

print("kl:", test_kl, "cc", test_cc, "nss", test_nss)
