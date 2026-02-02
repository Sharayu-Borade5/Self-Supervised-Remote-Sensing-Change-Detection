import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class SatellitePairDataset(Dataset):
    def __init__(self, root):
        self.pairs = sorted(os.listdir(root))
        self.root = root
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        before_path = os.path.join(self.root, self.pairs[idx], "before.png")
        after_path  = os.path.join(self.root, self.pairs[idx], "after.png")

        before = self.transform(Image.open(before_path).convert("RGB"))
        after  = self.transform(Image.open(after_path).convert("RGB"))

        return before, after
