import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from pathlib import Path
import torch

class SkinLesionDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.transform = transform or v2.Compose([
            v2.Resize((512, 512)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        self.filenames = sorted([f.name for f in self.image_dir.glob("*.jpg")])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_name = self.filenames[index]
        mask_name = img_name.replace(".jpg", "_segmentation.png")

        img_path = self.image_dir / img_name
        mask_path = self.mask_dir / mask_name

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        return self.transform(img, mask)