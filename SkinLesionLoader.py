import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import config
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import v2


config_loader = config.ConfigLoader("config.yaml")

class SkinLesionSet(Dataset):
    def __init__(self, images, masks):
        self.image_paths = images
        self.mask_paths = masks

        self.transform = v2.Compose([
            v2.ToImage(),            
            v2.Resize((config_loader.resolution(), config_loader.resolution())),   
            v2.ToDtype(torch.float32, scale=True), 
        ])
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")
        image, mask = self.transform(image,mask)
        return image, mask


def loadData(images_path, masks_path):
    dataset = SkinLesionSet(images_path,masks_path)
    train_size = int(config_loader.traing_percentage()*len(dataset))
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size,test_size])
    return (
        DataLoader(train, batch_size=config_loader.batch_size(), num_workers=config_loader.num_load_workers(),drop_last=True),
        DataLoader(test,batch_size=config_loader.batch_size(), num_workers=config_loader.num_load_workers(),drop_last=True)
        )


if __name__ == "__main__":
    path = Path("ISIC2018_Task1-2_Training_Input")
    files = list(path.glob("*.jpg"))
    train,test = loadData(files, files)
    for batch_idx, (data, target) in enumerate(train):
        #print(data[0].shape)
        #print(target[0].shape)
        print(len(data))
        #break
    