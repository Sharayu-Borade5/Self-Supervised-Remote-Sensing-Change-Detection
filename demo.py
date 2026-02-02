import torch
from models.siamese_model import SiameseChangeModel

model = SiameseChangeModel(embed_dim=256)

before = torch.randn(3, 256, 256)
after  = torch.randn(3, 256, 256)

score = torch.norm(
    model.encoder(before.unsqueeze(0)) -
    model.encoder(after.unsqueeze(0))
)

print("Change score:", score.item())
