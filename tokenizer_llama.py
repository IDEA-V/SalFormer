import torch
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("Enoch/llama-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
print('LLAMA tokenizer loaded')

def padding_fn(data):
    img, q, fix, hm, name = zip(*data)

    input_ids = tokenizer(q, return_tensors="pt", padding=True)

    return torch.stack(img), input_ids, torch.stack(fix), torch.stack(hm), name