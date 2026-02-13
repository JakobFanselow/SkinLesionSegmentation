import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import config
from pathlib import Path


class SkinLesionSet(Dataset):
    def __init__(self, images, masks):
        self.image_paths = images
        self.mask_paths = masks
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,index):
        image = torch.from_numpy(np.array(Image.open(self.image_paths[index])))
        mask = torch.from_numpy(np.array(Image.open(self.mask_paths[index])))

        return image, mask


def loadData(images_path, masks_path):
    dataset = SkinLesionSet(images_path,masks_path)
    config_loader = config.ConfigLoader("config.yaml")
    train_size = int(config_loader.traing_percentage()*len(dataset))
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size,test_size])
    return train,test


if __name__ == "__main__":
    path = Path("ISIC2018_Task1-2_Training_Input")
    files = list(path.glob("*.jpg"))
    train,test = loadData(files, files)
    print(len(train))