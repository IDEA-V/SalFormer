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
            torch.nn.GELU(),
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.GELU()
        )

        self.text_head = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.GELU()
        )

        self.cross_attention = torch.nn.MultiheadAttention(self.feature_dim, 16, kdim=self.feature_dim, vdim=self.feature_dim, batch_first=True)
        self.cross_attention1 = torch.nn.MultiheadAttention(self.feature_dim, 16, kdim=self.feature_dim, vdim=self.feature_dim, batch_first=True)

        self.ln1 = torch.nn.LayerNorm(self.feature_dim)
        self.ln2 = torch.nn.LayerNorm(self.feature_dim)

        self.self_attetion = torch.nn.MultiheadAttention(self.feature_dim, 16, batch_first=True)

        # query = torch.randn(8, 8, self.feature_dim).unsqueeze(0)
        # pos_encode2d = PositionalEncoding2D(self.feature_dim)
        # query_2d = pos_encode2d(query)
        # self.query = torch.nn.Parameter(torch.reshape(query_2d, (1, 64, self.feature_dim)))
        self.text_feature_query = torch.nn.Parameter(torch.randn(10, self.feature_dim).unsqueeze(0)/2)
        self.img_positional_embedding = torch.nn.Parameter(torch.zeros(49, self.feature_dim))
        self.text_positional_embedding = torch.nn.Parameter(torch.zeros(10, self.feature_dim))


        self.dense1 = torch.nn.Linear(self.feature_dim, self.feature_dim)
        self.relu1 = torch.nn.ReLU()

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

        self.vit.eval()
        self.bert.eval()
        self.train(True)

    # def eval(self):
    #     super().eval()
    #     self.vit.eval()
    #     self.bert.eval()

    # def train(self, mode=True):
    #     super().train(mode)
    #     self.vit.train(mode)
    #     self.bert.train(mode)

    def forward(self, img, q_inputs):

        img_features =  self.vit.forward(img, return_dict =True)["last_hidden_state"]
        latent_features = self.vision_head(img_features)+self.img_positional_embedding

        latent_features = latent_features.permute(0,2,1)
        # out = torch.reshape(latent_features, (features.shape[0], self.feature_dim, 8, 8))
        out = torch.reshape(latent_features, (latent_features.shape[0], self.feature_dim, 7, 7))
        out = self.decoder(out)

        return out
        