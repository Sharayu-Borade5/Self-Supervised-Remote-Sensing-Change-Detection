import torch
import numpy as np

def compute_change_map(model, before, after):
    model.eval()
    with torch.no_grad():
        f1 = model.encoder(before.unsqueeze(0))
        f2 = model.encoder(after.unsqueeze(0))

    diff = torch.norm(f1 - f2, dim=1)
    return diff.item()
