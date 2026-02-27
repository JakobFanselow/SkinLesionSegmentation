import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
import wandb

from unet.unet import UNet, AttentionUNet, ResUNet
from skin_lesion_dataset import SkinLesionDataset
from config import ConfigLoader

def train() -> None:
    with wandb.init():
        config = ConfigLoader("config.yaml")

        BASE_DIR = Path(__file__).resolve().parent
        MODEL_DIR = BASE_DIR / "models"
        MODEL_DIR.mkdir(exist_ok=True)

        EPOCHS = config.epochs()
        SEED = config.manual_seed()
        BATCH_SIZE = config.batch_size()
        LR = config.learning_rate()
        NUM_WORKERS = config.num_load_workers()

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

        train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, **loader_args)
        val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, **loader_args)
        test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, **loader_args)

        # model = UNet(in_channels=3, out_channels=1).to(device)
        # model = AttentionUNet(in_channels=3, out_channels=1).to(device)
        model = ResUNet(in_channels=3, out_channels=1).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(EPOCHS):
            model.train()
            train_running_loss = 0

            for img, mask in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
                img, mask = img.to(device), mask.to(device)

                optimizer.zero_grad()
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                loss.backward()
                optimizer.step()

                train_running_loss += loss.item()
            
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
                "val_loss": val_loss
            })

if __name__ == "__main__":
    train()