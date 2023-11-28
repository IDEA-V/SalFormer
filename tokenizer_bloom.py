import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b")
print('bloom-3b tokenizer loaded')

def padding_fn(data):
    img, q, fix, hm, name = zip(*data)

    input_ids = tokenizer(q, return_tensors="pt", padding=True)

    return torch.stack(img), input_ids, torch.stack(fix), torch.stack(hm), name
