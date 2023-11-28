import torch
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print('bert-base-uncased tokenizer loaded')

def padding_fn(data):
    img, q, fix, hm, name = zip(*data)

    input_ids = tokenizer(q, return_tensors="pt", padding=True)

    return torch.stack(img), input_ids, torch.stack(fix), torch.stack(hm), name


