import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
from pathlib import Path
import wandb
import yaml
import sys

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

def calc_test_loss(run_id) -> None:
    api = wandb.Api()
    wandb_path = f"jakob-fanselow-hasso-plattner-institut/SkinLesionSegmentation/{run_id}"
    print(f"Wandb path: {wandb_path}")
    run = api.run(wandb_path)


    config = ConfigLoader(run.config)

    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    MODEL = config.model()

    EPOCHS = config.epochs()
    SEED = config.manual_seed()
    BATCH_SIZE = config.batch_size()
    LR = config.learning_rate()
    MAX_LR = max(config.max_learning_rate(), LR * 1.5)
    WEIGHT_DECAY = config.weight_decay()
    NUM_WORKERS = config.num_load_workers()

    MAX_NORM = config.max_norm()
    DROP_LAST = config.drop_last()

    DICE_WEIGHT = config.dice_weight()
    BCE_WEIGHT = config.bce_weight()

    TEST_PERC = config.test_percentage()
    VAL_PERC = config.validation_percentage()
    TRAIN_PERC = config.train_percentage()

    KERNEL_SIZE = config.kernel_size()
    UNET_EXCLUDE_BOTTLENECK = config.exclude_bottleneck()

    RES = config.resolution()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    dataset = SkinLesionDataset(
        BASE_DIR / "isic2018-challenge-task1-data-segmentation" / "ISIC2018_Task1-2_Training_Input",
        BASE_DIR / "isic2018-challenge-task1-data-segmentation" / "ISIC2018_Task1_Training_GroundTruth",
        RES
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

    match MODEL:
        case "UNET":
            if KERNEL_SIZE is None:
                KERNEL_SIZE = 3
            if UNET_EXCLUDE_BOTTLENECK is None:
                UNET_EXCLUDE_BOTTLENECK = False
            model = UNet(in_channels=3, out_channels=1, exclude_bottleneck=UNET_EXCLUDE_BOTTLENECK, conv_kernel_size=KERNEL_SIZE).to(device)
        case "AUNET":
            model = AttentionUNet(in_channels=3, out_channels=1).to(device)
        case "RESNET":
            model = ResUNet(in_channels=3, out_channels=1).to(device)
        case "VNET":
            if KERNEL_SIZE is None:
                KERNEL_SIZE = 5
            model = VNet2D(kernel_size=KERNEL_SIZE).to(device)
        case _:
            print("No correct model name provided. Using UNet as fallback.")
            model = UNet(in_channels=3, out_channels=1).to(device)

    model_path = MODEL_DIR / f"{run_id}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))

    # criterion = nn.BCEWithLogitsLoss()
    criterion = DiceBCELoss(dice_weight=DICE_WEIGHT, bce_weight=BCE_WEIGHT)



    model.eval()
    test_running_loss = 0.0
    with torch.no_grad():
        for img, mask in test_dataloader:
            img, mask = img.to(device), mask.to(device)
            y_pred = model(img)
            test_running_loss += criterion(y_pred, mask).item()

    test_loss = test_running_loss / len(test_dataloader)
    return test_loss

       


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_id_arg = sys.argv[1]
        print(calc_test_loss(run_id_arg))
