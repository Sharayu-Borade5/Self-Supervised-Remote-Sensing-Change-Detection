import torch
from training.losses import contrastive_loss

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for before, after in loader:
        before, after = before.to(device), after.to(device)
        z1, z2 = model(before, after)
        loss = contrastive_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)
