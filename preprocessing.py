import os
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import config

config_loader = config.ConfigLoader("config.yaml")

def resize_image(image_info):
    img_path, save_path, size = image_info
    try:
        with Image.open(img_path) as img:
            img = img.resize((size, size), resample=Image.Resampling.LANCZOS)
            img.save(save_path, "JPEG", quality=95)
    except Exception as e:
        print(f"Error with: {img_path}: {e}")

def prepare_dataset(source_dir, target_dir, size=256):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    
    tasks = []
    for f in files:
        tasks.append((f, target_dir / f.name, size))

    print(f"Converting {len(tasks)} images in {source_dir.name}...")
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(resize_image, tasks), total=len(tasks)))

if __name__ == "__main__":
    RES = config_loader.resolution()
    
    prepare_dataset(
        "isic2018-challenge-task1-data-segmentation/ISIC2018_Task1-2_Training_Input",
        f"data_prepared/images_{RES}",
        size=RES
    )
    
    prepare_dataset(
        "isic2018-challenge-task1-data-segmentation/ISIC2018_Task1_Training_GroundTruth",
        f"data_prepared/masks_{RES}",
        size=RES
    )