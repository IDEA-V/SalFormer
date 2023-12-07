import torch
from torch.utils.data import DataLoader
from env import *

import argparse
from dataset_new import ImagesWithSaliency
from torchvision.utils import save_image
from transformers import SwinModel
from pathlib import Path

def evaluation(Model:str, ckpt: str, device, batch_size:int):
    eps=1e-10

    if Model == 'llama':
        from model_llama import SalFormer
        from transformers import LlamaModel
        from tokenizer_llama import padding_fn
        # llm = LlamaModel.from_pretrained("Enoch/llama-7b-hf", low_cpu_mem_usage=True)
        llm = LlamaModel.from_pretrained("daryl149/Llama-2-7b-chat-hf", low_cpu_mem_usage=True)
        neuron_n = 4096
        print("llama loaded")
    elif Model == 'bloom':
        from model_llama import SalFormer
        from transformers import BloomModel
        from tokenizer_bloom import padding_fn
        llm = BloomModel.from_pretrained("bigscience/bloom-3b")
        neuron_n = 2560
        print('BloomModel loaded')
    elif Model == 'bert':
        from model_swin import SalFormer
        from transformers import BertModel
        from tokenizer_bert import padding_fn
        llm = BertModel.from_pretrained("bert-base-uncased")
        print('BertModel loaded')
    else:
        print('model not available, possiblilities: llama, bloom, bert')
        return

    test_set = ImagesWithSaliency("data/test.npy")

    Path('./eval_results').mkdir(parents=True, exist_ok=True)

    # vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    # vit = timm.create_model('xception41p.ra3_in1k', pretrained=True)

    if Model == 'bert':
        model = SalFormer(vit, llm).to(device)
    else:
        model = SalFormer(vit, llm, neuron_n = neuron_n).to(device)

    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=padding_fn, num_workers=8)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='bert')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default='./ckpt/model_bert_freeze_10kl_5cc_2nss.tar')
    args = vars(parser.parse_args())

    evaluation(Model = args['model'], device = args['device'], ckpt = args['ckpt'], batch_size = args['batch_size'])
