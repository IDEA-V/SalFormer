import torch
import math
from transformers import ViTModel
from positional_encodings.torch_encodings import PositionalEncoding2D 

class SalFormer(torch.nn.Module):
    def __init__(self, vision_encoder, bert):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        
        
        self.vit = vision_encoder
        self.feature_dim = 768
        self.bert = bert
        
        self.vision_head = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU()
        )

        self.text_head = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU()
        )

        self.cross_attention = torch.nn.MultiheadAttention(self.feature_dim, 16, kdim=self.feature_dim, vdim=self.feature_dim, batch_first=True)
        self.cross_attention1 = torch.nn.MultiheadAttention(self.feature_dim, 16, kdim=self.feature_dim, vdim=self.feature_dim, batch_first=True)

        self.ln1 = torch.nn.LayerNorm(self.feature_dim)
        self.ln2 = torch.nn.LayerNorm(self.feature_dim)

        self.self_attetion = torch.nn.MultiheadAttention(self.feature_dim, 16, batch_first=True)

        self.img_positional_embedding = torch.nn.Parameter(torch.zeros(49, self.feature_dim))


        self.dense1 = torch.nn.Linear(self.feature_dim, self.feature_dim)
        self.relu1 = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.sigmoid = torch.nn.Sigmoid()

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(self.feature_dim, 512, 3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Conv2d(512, 512, 3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Upsample((16,16), mode='bilinear'),
            torch.nn.Conv2d(512, 256, 3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Conv2d(256, 256, 3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Upsample((32,32), mode='bilinear'),
            torch.nn.Conv2d(256, 128, 3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Conv2d(128, 128, 3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Upsample((130,130), mode='bilinear'),
            torch.nn.Conv2d(128, 1, 3),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid(),
        )

        self.train(True)

    def eval(self):
        super().eval()
        self.vit.eval()
        self.bert.eval()

    def train(self, mode=True):
        super().train(mode)
        self.vit.train(mode)
        self.bert.train(mode)

    def forward(self, img, q_inputs):

        img_features =  self.vit.forward(img, return_dict =True)["last_hidden_state"]
        img_features = img_features + self.img_positional_embedding

        text_features =  self.bert(**q_inputs)["last_hidden_state"]
    
        fused_features = torch.concat((self.vision_head(img_features), self.text_head(text_features)), 1)
        att_fused_features = self.self_attetion.forward(fused_features, fused_features, fused_features, need_weights=False)[0]
        fused_features = fused_features + att_fused_features
        fused_features = self.ln1(fused_features)
        fused_features = fused_features.permute(0,2,1)
        fused_features = self.pool(fused_features)
        fused_features = self.sigmoid(fused_features)

        img_features = img_features.permute(0,2,1)
        fused_img_features = img_features * fused_features.repeat([1,1,49])

        out = fused_img_features.contiguous().view(fused_img_features.shape[0], self.feature_dim, 7, 7)
        out = self.decoder(out)

        return out

  