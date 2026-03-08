import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
from pathlib import Path
import wandb
import yaml

from unet.unet import UNet, AttentionUNet, ResUNet
from skin_lesion_dataset import SkinLesionDataset
from config import ConfigLoader
from vnet.vnet import VNet2D
from dice_bce_loss import DiceBCELoss

def identity(x):
    return x

def rotate_90(x):
    return F.rotate(x, 90)

def rotate_180(x):
    return F.rotate(x, 180)

def rotate_270(x):
    return F.rotate(x, 270)

def train() -> None:
    with open("config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    with wandb.init(config=config_dict) as run:
        config = ConfigLoader("config.yaml")

        BASE_DIR = Path(__file__).resolve().parent
        MODEL_DIR = BASE_DIR / "models"
        MODEL_DIR.mkdir(exist_ok=True)

        EPOCHS = config.epochs()
        SEED = config.manual_seed()
        BATCH_SIZE = config.batch_size()
        LR = config.learning_rate()
        MAX_LR = config.max_learning_rate()
        WEIGHT_DECAY = config.weight_decay()
        NUM_WORKERS = config.num_load_workers()

        MAX_NORM = config.max_norm()
        DROP_LAST = config.drop_last()

        DICE_WEIGHT = config.dice_weight()
        BCE_WEIGHT = config.bce_weight()

        TEST_PERC = config.test_percentage()
        VAL_PERC = config.validation_percentage()
        TRAIN_PERC = config.train_percentage()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)

        dataset = SkinLesionDataset(
            BASE_DIR / "isic2018-challenge-task1-data-segmentation" / "ISIC2018_Task1-2_Training_Input",
            BASE_DIR / "isic2018-challenge-task1-data-segmentation" / "ISIC2018_Task1_Training_GroundTruth"
        )

        generator = torch.Generator().manual_seed(SEED)
        train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[TRAIN_PERC, VAL_PERC, TEST_PERC], generator=generator)

        loader_args = {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pin_memory": True
        }

        train_transforms_90_multiples = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomChoice([
                transforms.Lambda(identity),
                transforms.Lambda(rotate_90),
                transforms.Lambda(rotate_180),
                transforms.Lambda(rotate_270)
            ]),
            transforms.ToTensor()
        ])

        train_dataset.transform = train_transforms_90_multiples

        train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, **loader_args, drop_last=DROP_LAST)
        val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, **loader_args)
        test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, **loader_args)

        # model = UNet(in_channels=3, out_channels=1).to(device)
        # model = AttentionUNet(in_channels=3, out_channels=1).to(device)
        # model = ResUNet(in_channels=3, out_channels=1).to(device)
        model = VNet2D().to(device)


        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=MAX_LR, 
            steps_per_epoch=len(train_dataloader), 
            epochs=EPOCHS
        )
        # criterion = nn.BCEWithLogitsLoss()
        criterion = DiceBCELoss(dice_weight=DICE_WEIGHT, bce_weight=BCE_WEIGHT)

        for epoch in range(EPOCHS):
            model.train()
            train_running_loss = 0
            clipped = 0

            for img, mask in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
                img, mask = img.to(device), mask.to(device)

                optimizer.zero_grad()
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                loss.backward()
                old_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)      
                if old_norm > MAX_NORM:
                    clipped += 1

                optimizer.step()
                scheduler.step()

                train_running_loss += loss.item()
            
            clipped_percentage = clipped / len(train_dataloader)
            train_loss = train_running_loss / len(train_dataloader)

            model.eval()
            val_running_loss = 0
            with torch.no_grad():
                for img, mask in val_dataloader:
                    img, mask = img.to(device), mask.to(device)
                    y_pred = model(img)
                    val_running_loss += criterion(y_pred, mask).item()

            val_loss = val_running_loss / len(val_dataloader)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "clipped_percentage": clipped_percentage
            })
        
        save_path = f"models/{run.id}.pth"
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train()
