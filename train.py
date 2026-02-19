import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
import wandb

from unet.unet import UNet
from skin_lesion_dataset import SkinLesionDataset
from config import ConfigLoader

if __name__ == "__main__":
    config_vals = ConfigLoader("config.yaml")

    BASE_DIR = Path(__file__).resolve().parent
    
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    MODEL_SAVE_PATH = MODEL_DIR / "unet.pth"

    BATCH_SIZE = config_vals.batch_size()
    LEARNING_RATE = config_vals.learning_rate()
    EPOCHS = config_vals.epochs()
    TRAINING_PERC = config_vals.traing_percentage()
    SEED = config_vals.manual_seed()

    wandb.init(
        project="skin-lesion-segmentation",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "architecture": "UNet",
            "dataset": "ISIC2018"
        }
    )
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    train_dataset = SkinLesionDataset(BASE_DIR / "isic2018-challenge-task1-data-segmentation" / "ISIC2018_Task1-2_Training_Input",
                                BASE_DIR / "isic2018-challenge-task1-data-segmentation"/ "ISIC2018_Task1_Training_GroundTruth")

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(train_dataset, [TRAINING_PERC, 1 - TRAINING_PERC], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    wandb.watch(model, log_freq=100)

    for epoch in range(EPOCHS): 
        model.train()
        train_running_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for idx, (img, mask) in enumerate(progress_bar):
            img = img.to(device)
            mask = mask.to(device)

            y_pred = model(img)
            optimizer.zero_grad()
            loss = criterion(y_pred, mask)
            
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            
            wandb.log({"batch_loss": loss.item()})

        train_loss = train_running_loss / len(train_dataloader)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for img, mask in val_dataloader:
                img = img.to(device)
                mask = mask.to(device)
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                val_running_loss += loss.item()
        
        val_loss = val_running_loss / len(val_dataloader)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]['lr']
        })

        print(f"\nEpoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    wandb.finish()