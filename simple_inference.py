import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
from glob import glob
import torch
torch.manual_seed(42)
from PIL import Image
import numpy as np
np.random.seed(42)
from typing import List
from model_swin import SalFormer
from transformers import AutoImageProcessor, AutoTokenizer, BertModel, SwinModel

DEVICE = 'cuda'

def predict(ques: str, img_path: str) -> List:
    """
    Execute the prediction.

    Args:
        ques: a question string to feed into VisSalFormer

    Returns: [list]
        - heatmap from VisSalFormer (np.array)
        - Average WAVE score across pixels (float, [0, 1))
    """
    image = Image.open(img_path).convert("RGB")
    img_pt = image_processor(image, return_tensors="pt").to(DEVICE)
    inputs = tokenizer(ques, return_tensors="pt").to(DEVICE)

    mask = model(img_pt['pixel_values'], inputs)
    mask = mask.detach().cpu().squeeze().numpy()
    heatmap = (mask * 255).astype(np.uint8)
    im_grey = image.convert('L')

    heatmap = np.resize(heatmap, (image.size[1], image.size[0]))
    return [heatmap, image, np.array(im_grey)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="/netpool/homes/wangyo/Projects/chi2025_scanpath/evaluation/images/economist_daily_chart_85.png")
    parser.add_argument("--query", type=str, default="type your query")
    args = vars(parser.parse_args())
    print(args["query"])

    image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")
    model = SalFormer(vit, bert).to(DEVICE)
    checkpoint = torch.load('./model/model_lr6e-5_wd1e-4.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    predictions = predict(args['query'], args['img_path'])
    np.save(f'predictions/{args["img_path"].split("/")[-1].strip(".png")}_{args["query"][:10]}.npy', predictions[0]/255.)
