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
        
        self.bert_head = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU()
        )

        self.cross_attention = torch.nn.MultiheadAttention(self.feature_dim, 16, kdim=self.feature_dim, vdim=self.feature_dim, batch_first=True)
        self.ln1 = torch.nn.LayerNorm(self.feature_dim)
        self.ln2 = torch.nn.LayerNorm(self.feature_dim)
        self.ln3 = torch.nn.LayerNorm(self.feature_dim)
        self.relu1 = torch.nn.ReLU()

        self.self_attetion = torch.nn.MultiheadAttention(self.feature_dim, 16, batch_first=True)
        # self.up = torch.nn.Upsample(8*8, mode='linear')

        query = torch.randn(8, 8, self.feature_dim).unsqueeze(0)
        pos_encode2d = PositionalEncoding2D(self.feature_dim)
        query_2d = pos_encode2d(query)
        self.query = torch.nn.Parameter(torch.reshape(query_2d, (1, 64, self.feature_dim)))
        self.text_feature_query = torch.nn.Parameter(torch.randn(50, self.feature_dim).unsqueeze(0)/2)

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(self.feature_dim, 512, 3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Conv2d(512, 512, 3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Upsample((16,16), mode='bilinear'),
            torch.nn.Conv2d(512, 256, 3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Conv2d(256, 256, 3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Upsample((32,32), mode='bilinear'),
            torch.nn.Conv2d(256, 128, 3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Conv2d(128, 128, 3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Upsample((64,64), mode='bilinear'),
            torch.nn.Conv2d(128, 64, 3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Upsample((130,130), mode='bilinear'),
            torch.nn.Conv2d(64, 1, 3),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid(),
        )

        self.train(True)

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, torch.nn.MultiheadAttention):
    #         for n, p in module.named_parameters():
    #             if 'weight' in n:
    #                 torch.nn.init.normal_(p.data)
    #     if isinstance(module, torch.nn.Conv2d):
    #         torch.nn.init.normal_(module.weight)
    #         print("")

    def eval(self):
        super().eval()
        self.vit.eval()
        self.bert.eval()

    def train(self, mode=True):
        super().train(mode)
        self.vit.train(mode)
        self.bert.train(mode)

    def forward(self, img, q_inputs):

        img_features =  self.vit.forward(**img, return_dict =True)["last_hidden_state"]
        text_features =  self.bert(**q_inputs)["last_hidden_state"]
        text_features = self.cross_attention.forward(self.text_feature_query.repeat([text_features.shape[0], 1, 1]), text_features, text_features, need_weights=False)[0]
        text_features = self.ln1(text_features)

        features = torch.concat((img_features, text_features), 1)
        self_att_features = self.self_attetion.forward(features, features, features, need_weights=False)[0]
        features = features + self_att_features
        self_att_features = self.ln2(self_att_features)
        # features = self.relu1(features)
        
        latent_features = self.cross_attention.forward(self.query.repeat([features.shape[0], 1, 1]), features, features, need_weights=False)[0]
        latent_features = self.ln3(latent_features)

        latent_features = latent_features.permute(0,2,1)
        out = torch.reshape(latent_features, (features.shape[0], self.feature_dim, 8, 8))
        out = self.decoder(out)

        return out
        