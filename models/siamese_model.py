import torch.nn as nn
from models.encoder import Encoder
from models.projection_head import ProjectionHead

class SiameseChangeModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.encoder = Encoder(embed_dim)
        self.projector = ProjectionHead(embed_dim)

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return z1, z2
