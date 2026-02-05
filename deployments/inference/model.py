import torch.nn as nn
from encoder import FeatureEncoder
from decoder import RecoloringDecoder


class RecolorizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FeatureEncoder()
        self.decoder = RecoloringDecoder()

    def forward(self, ori_img, tgt_palette, illu):
        c1, c2, c3, c4 = self.encoder(ori_img)
        # print(c1, c2, c3, c4)
        out = self.decoder(c1, c2, c3, c4, tgt_palette, illu)
        return out
    

def get_model():
    model = RecolorizerModel()
    return model