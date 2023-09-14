import torch
from torch.utils.data import DataLoader
# from model_swin import SalFormer
# from model_vit import SalFormer
# from model_swin_pure import SalFormer
from model_mask import SalFormer
# from model_wo_cross_attn import SalFormer
from transformers import ViTModel

from transformers import AutoTokenizer, BertModel, SwinModel
from dataset import ImagesWithSaliency
from torchvision import transforms
from torchvision.utils import save_image

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

device = 'cuda'
eps=1e-10

test_set = ImagesWithSaliency("./SalChartQA/test/raw_img/", "./SalChartQA/test/saliency_all/fix_maps/", "./SalChartQA/test/saliency_all/heatmaps/", img_transform_no_augment, fix_transform, hm_transform)

# vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
model = SalFormer(vit, bert).to(device)
checkpoint = torch.load('./model_wo_cross.tar')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def padding_fn(data):
    img, q, fix, hm, name = zip(*data)

    input_ids = tokenizer(q, return_tensors="pt", padding=True)

    return torch.stack(img), input_ids, torch.stack(fix), torch.stack(hm), name

test_dataloader = DataLoader(test_set, batch_size=32, shuffle=True, collate_fn=padding_fn, num_workers=8)
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