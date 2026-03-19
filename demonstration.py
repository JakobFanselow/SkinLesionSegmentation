import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from unet.unet import UNet, ResUNet
from skin_lesion_dataset import SkinLesionDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = ResUNet(in_channels=3, out_channels=1) 
model.load_state_dict(torch.load("models/res_u_net.pth", map_location=device))
model.to(device)
model.eval()

def visualize_predictions(dataset, num_samples=4):
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))
    
    axes[0, 0].set_title("Original Image")
    axes[0, 1].set_title("Real Mask")
    axes[0, 2].set_title("Predicted Mask")

    for i in range(num_samples):
        image, mask = dataset[i]
        
        input_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.sigmoid(output)
            prediction = (prediction > 0.5).float() 

        img_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        pred_np = prediction.squeeze().cpu().numpy()

        axes[i, 0].imshow(img_np)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_np, cmap='gray')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('prediction_results.png')
    print("Plot wurde als 'prediction_results.png' gespeichert.")
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    dataset = SkinLesionDataset(
                BASE_DIR / "isic2018-challenge-task1-data-segmentation" / "ISIC2018_Task1-2_Training_Input",
                BASE_DIR / "isic2018-challenge-task1-data-segmentation" / "ISIC2018_Task1_Training_GroundTruth"
            )
    visualize_predictions(dataset, num_samples=4)