import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)
