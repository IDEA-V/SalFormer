import torch
import math
from transformers import ViTModel

class SalFormer(torch.nn.Module):
    def __init__(self, bert, vit_config):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        
        self.vit = ViTModel(vit_config)
        self.bert = bert
        
        self.cross_attention = torch.nn.MultiheadAttention(bert.config.hidden_size, 12, kdim=vit_config.hidden_size, vdim=vit_config.hidden_size, batch_first=True)
        self. ln1 = torch.nn.LayerNorm(vit_config.hidden_size)
        self.self_attetion = torch.nn.MultiheadAttention(vit_config.hidden_size, 12, batch_first=True)
        self.up = torch.nn.Upsample(8*8, mode='linear')

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(vit_config.hidden_size, 512, 1),
            torch.nn.Upsample((16,16), mode='bilinear'),
            torch.nn.Conv2d(512, 256, 3),
            torch.nn.Upsample((32,32), mode='bilinear'),
            torch.nn.Conv2d(256, 128, 3),
            torch.nn.Upsample((64,64), mode='bilinear'),
            torch.nn.Conv2d(128, 64, 3),
            torch.nn.Upsample((128,128), mode='bilinear'),
            torch.nn.Conv2d(64, 1, 3),
            torch.nn.Upsample((256,256), mode='bilinear'),
            torch.nn.Conv2d(1, 1, 7),
        )


    def forward(self, img, q_inputs):

        img_features =  self.vit.forward(img, return_dict =True)["last_hidden_state"]
        text_features =  self.bert(**q_inputs)["last_hidden_state"]

        out = self.cross_attention.forward(text_features, img_features, img_features, need_weights=False)[0]
        out = self.ln1(out)
        out = self.self_attetion.forward(out, out, out, need_weights=False)[0]
        out = out.permute(0,2,1)
        out = torch.reshape(self.up(out), (-1,768, 8, 8))
        out = self.decoder(out)

        return out
        