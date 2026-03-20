import loader
import torch
import matplotlib.pyplot as plt
import random


def show_random(wandb_path):
    test_dataset, _ = loader.get_test_dataset_wandb(wandb_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = loader.load_model(wandb_path)

    model.to(device)
    model.eval()

    random_index = random.randint(0, len(test_dataset) - 1)
    image, ground_truth = test_dataset[random_index]



    image_on_device = image.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction_on_device = model(image_on_device)
    

    prediction = prediction_on_device.squeeze().to("cpu")
    
    img_np = image.permute(1, 2, 0).cpu().numpy()
    mask_np = ground_truth.squeeze().cpu().numpy()
    pred_np = prediction.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(img_np)
    ax[0].set_title(f"Original Image (Index: {random_index})")
    ax[0].axis('off')
    
    ax[1].imshow(mask_np, cmap='gray')
    ax[1].set_title("Ground Truth")
    ax[1].axis('off')
    
    ax[2].imshow(pred_np, cmap='gray') 
    ax[2].set_title("Model Prediction")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()
