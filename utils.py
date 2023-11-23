import torch
from transformers import AutoTokenizer

kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print('bert-base-uncased tokenizer loaded')
# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b")
# print('bloom-3b tokenizer loaded')

def padding_fn(data):
    img, q, fix, hm, name = zip(*data)

    input_ids = tokenizer(q, return_tensors="pt", padding=True)

    return torch.stack(img), input_ids, torch.stack(fix), torch.stack(hm), name

def log_softmax2d(x):
    logits = torch.log_softmax(x.flatten(), 0)
    return logits.reshape(x.shape)

def softmax2d(x):
    logits = torch.softmax(x.flatten(), 0)
    return logits.reshape(x.shape)

def inference(model, device, eps, img, input_ids, fix, hm):
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

    return y, kl, cc, nss
