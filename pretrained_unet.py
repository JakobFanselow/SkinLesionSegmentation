import os
import random
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import wandb
import yaml
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from config import ConfigLoader
from dice_bce_loss import DiceBCELoss
from dice_loss import DiceLoss


DEFAULT_IMAGE_DIR = (
    Path(__file__).resolve().parent
    / "isic2018-challenge-task1-data-segmentation"
    / "ISIC2018_Task1-2_Training_Input"
)
DEFAULT_MASK_DIR = (
    Path(__file__).resolve().parent
    / "isic2018-challenge-task1-data-segmentation"
    / "ISIC2018_Task1_Training_GroundTruth"
)


class ISIC2018SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, resolution, augment=False, filenames=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.resolution = resolution
        self.augment = augment
        self.filenames = filenames or sorted(path.name for path in self.image_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image_name = self.filenames[index]
        mask_name = image_name.replace(".jpg", "_segmentation.png")

        image = Image.open(self.image_dir / image_name).convert("RGB")
        mask = Image.open(self.mask_dir / mask_name).convert("L")

        image = TF.resize(
            image,
            [self.resolution, self.resolution],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            [self.resolution, self.resolution],
            interpolation=InterpolationMode.NEAREST,
        )

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            rotations = random.randint(0, 3)
            if rotations:
                angle = rotations * 90
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        image_tensor = TF.to_tensor(image)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()

        return image_tensor, mask_tensor


def resolve_data_dir(configured_path, default_path):
    return Path(configured_path) if configured_path else default_path


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_env_int(name, default):
    value = os.getenv(name)
    return int(value) if value is not None else default


def get_env_float(name, default):
    value = os.getenv(name)
    return float(value) if value is not None else default


def mean_iou_from_logits(logits, targets, threshold=0.5, smooth=1e-6):
    predictions = (torch.sigmoid(logits) > threshold).float()
    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    return ((intersection + smooth) / (union + smooth)).mean()


def evaluate(model, dataloader, device, criterion, dice_metric, bce_metric):
    model.eval()
    totals = {
        "loss": 0.0,
        "dice_loss": 0.0,
        "bce_loss": 0.0,
        "iou": 0.0,
    }

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            totals["loss"] += criterion(logits, masks).item()
            totals["dice_loss"] += dice_metric(logits, masks).item()
            totals["bce_loss"] += bce_metric(logits, masks).item()
            totals["iou"] += mean_iou_from_logits(logits, masks).item()

    num_batches = len(dataloader)
    return {name: value / num_batches for name, value in totals.items()}


def train() -> None:
    with open("config.yaml", "r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle)

    config = ConfigLoader("config.yaml")
    set_seed(config.manual_seed())

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)
    torch_home = base_dir / ".torch"
    torch_home.mkdir(exist_ok=True)
    torch.hub.set_dir(str(torch_home))
    batch_size = get_env_int("SMP_UNET_BATCH_SIZE", config.batch_size())
    num_workers = get_env_int("SMP_UNET_NUM_WORKERS", config.num_load_workers())
    epochs = get_env_int("SMP_UNET_EPOCHS", config.epochs())
    resolution = get_env_int("SMP_UNET_RESOLUTION", config.resolution())
    max_samples = get_env_int("SMP_UNET_MAX_SAMPLES", 0)
    encoder_weights = os.getenv("SMP_UNET_ENCODER_WEIGHTS", "imagenet")
    if encoder_weights.lower() == "none":
        encoder_weights = None
    learning_rate = get_env_float("SMP_UNET_LR", config.learning_rate())
    max_learning_rate = get_env_float("SMP_UNET_MAX_LR", config.max_learning_rate())
    wandb_mode = os.getenv("WANDB_MODE", "online")

    image_dir = resolve_data_dir(
        config_dict.get("paths", {}).get("image_dir"),
        DEFAULT_IMAGE_DIR,
    )
    mask_dir = resolve_data_dir(
        config_dict.get("paths", {}).get("mask_dir"),
        DEFAULT_MASK_DIR,
    )

    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"ISIC2018 dataset not found. image_dir={image_dir} mask_dir={mask_dir}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = torch.Generator().manual_seed(config.manual_seed())
    filenames = sorted(path.name for path in image_dir.glob("*.jpg"))
    if max_samples > 0:
        filenames = filenames[:max_samples]
    train_subset, val_subset, test_subset = random_split(
        dataset=filenames,
        lengths=[
            config.train_percentage(),
            config.validation_percentage(),
            config.test_percentage(),
        ],
        generator=generator,
    )
    train_dataset = ISIC2018SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        resolution=resolution,
        augment=True,
        filenames=[filenames[index] for index in train_subset.indices],
    )
    val_dataset = ISIC2018SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        resolution=resolution,
        augment=False,
        filenames=[filenames[index] for index in val_subset.indices],
    )
    test_dataset = ISIC2018SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        resolution=resolution,
        augment=False,
        filenames=[filenames[index] for index in test_subset.indices],
    )

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        drop_last=config.drop_last(),
        **loader_args,
    )
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, **loader_args)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    ).to(device)

    criterion = DiceBCELoss(
        dice_weight=config.dice_weight(),
        bce_weight=config.bce_weight(),
    )
    dice_metric = DiceLoss()
    bce_metric = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay(),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )

    wandb_config = {
        **config_dict,
        "model": {
            "name": "smp.Unet",
            "encoder_name": "resnet34",
            "encoder_weights": "imagenet",
        },
        "dataset": "ISIC2018",
        "device": device,
        "runtime_overrides": {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "epochs": epochs,
            "resolution": resolution,
            "max_samples": max_samples,
            "encoder_weights": encoder_weights,
            "learning_rate": learning_rate,
            "max_learning_rate": max_learning_rate,
            "wandb_mode": wandb_mode,
        },
    }

    with wandb.init(
        project=os.getenv("WANDB_PROJECT", "SkinLesionSegmentation"),
        config=wandb_config,
        mode=wandb_mode,
        settings=wandb.Settings(
            root_dir=str(base_dir),
            start_method="thread",
            x_disable_stats=True,
        ),
    ) as run:
        best_val_iou = float("-inf")
        best_model_path = model_dir / f"{run.id}_best.pth"

        for epoch in range(epochs):
            model.train()
            train_totals = {
                "loss": 0.0,
                "dice_loss": 0.0,
                "bce_loss": 0.0,
                "iou": 0.0,
            }
            clipped = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for images, masks in progress:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                logits = model(images)

                loss = criterion(logits, masks)
                dice_loss = dice_metric(logits, masks)
                bce_loss = bce_metric(logits, masks)
                iou = mean_iou_from_logits(logits, masks)

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.max_norm(),
                )
                if grad_norm > config.max_norm():
                    clipped += 1

                optimizer.step()
                scheduler.step()

                train_totals["loss"] += loss.item()
                train_totals["dice_loss"] += dice_loss.item()
                train_totals["bce_loss"] += bce_loss.item()
                train_totals["iou"] += iou.item()

            train_metrics = {
                name: value / len(train_loader) for name, value in train_totals.items()
            }
            val_metrics = evaluate(model, val_loader, device, criterion, dice_metric, bce_metric)

            clipped_percentage = clipped / len(train_loader)
            current_lr = scheduler.get_last_lr()[0]

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "learning_rate": current_lr,
                    "train_loss": train_metrics["loss"],
                    "train_dice_loss": train_metrics["dice_loss"],
                    "train_bce_loss": train_metrics["bce_loss"],
                    "train_iou": train_metrics["iou"],
                    "val_loss": val_metrics["loss"],
                    "val_dice_loss": val_metrics["dice_loss"],
                    "val_bce_loss": val_metrics["bce_loss"],
                    "val_iou": val_metrics["iou"],
                    "clipped_percentage": clipped_percentage,
                }
            )
            print(
                "epoch="
                f"{epoch + 1} "
                f"train_dice_loss={train_metrics['dice_loss']:.4f} "
                f"train_iou={train_metrics['iou']:.4f} "
                f"val_dice_loss={val_metrics['dice_loss']:.4f} "
                f"val_iou={val_metrics['iou']:.4f}"
            )

            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                torch.save(model.state_dict(), best_model_path)

        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_metrics = evaluate(model, test_loader, device, criterion, dice_metric, bce_metric)
        wandb.log(
            {
                "test_loss": test_metrics["loss"],
                "test_dice_loss": test_metrics["dice_loss"],
                "test_bce_loss": test_metrics["bce_loss"],
                "test_iou": test_metrics["iou"],
            }
        )
        print(
            "test "
            f"dice_loss={test_metrics['dice_loss']:.4f} "
            f"iou={test_metrics['iou']:.4f}"
        )

        torch.save(model.state_dict(), model_dir / f"{run.id}_last.pth")


if __name__ == "__main__":
    train()
