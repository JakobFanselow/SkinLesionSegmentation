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

def get_test_dataset_wandb(run_wandb_path):
    api = wandb.Api()
    run = api.run(run_wandb_path)
    config = ConfigLoader(run.config)
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    NUM_WORKERS = config.num_load_workers()
    BATCH_SIZE = config.batch_size()
    SEED = config.manual_seed()
    TEST_PERC = config.test_percentage()
    VAL_PERC = config.validation_percentage()
    TRAIN_PERC = config.train_percentage()
    RES = config.resolution()
    dataset = SkinLesionDataset(
        BASE_DIR / "isic2018-challenge-task1-data-segmentation" / "ISIC2018_Task1-2_Training_Input",
        BASE_DIR / "isic2018-challenge-task1-data-segmentation" / "ISIC2018_Task1_Training_GroundTruth",
        RES
    )

    loader_args = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True
    }


    generator = torch.Generator().manual_seed(SEED)
    _, _, test_dataset = random_split(dataset=dataset, lengths=[TRAIN_PERC, VAL_PERC, TEST_PERC], generator=generator)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, **loader_args)

    return (test_dataset, test_dataloader)

def load_model(run_wandb_path):
    api = wandb.Api()
    run = api.run(run_wandb_path)
    config = ConfigLoader(run.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = BASE_DIR / "models"
    model_path = MODEL_DIR / f"{Path(run_wandb_path).name}.pth"

    MODEL = config.model()
    UNET_EXCLUDE_BOTTLENECK = config.exclude_bottleneck()
    KERNEL_SIZE = config.kernel_size()

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
    
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model
