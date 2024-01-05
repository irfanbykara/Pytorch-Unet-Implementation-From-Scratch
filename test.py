from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.nn.functional import softmax
from matplotlib.colors import ListedColormap
import cv2
import os
import argparse

# Import your UNet model definition
from models import UnetReady

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Segmentation model evaluation script')
    parser.add_argument('--datapath', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--modelpath', type=str, required=True, help='Path to the trained model file (.pth)')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    return parser.parse_args()

# Get command line arguments
args = parse_arguments()

# Check if CUDA is available and requested
device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

# Folder paths
image_folder = os.path.join(args.datapath, 'val/images')
ground_truth_folder = os.path.join(args.datapath, 'val/labels')
output_folder = os.path.join(args.datapath, 'val/output_plots_final')

# Load the trained model
model = UnetReady(num_classes=3).to(device)
model.load_state_dict(torch.load(args.modelpath, map_location=device))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((572, 572)),
    transforms.ToTensor(),
])

# Get a list of image file names in the folder
image_files = os.listdir(image_folder)

# Iterate through each image
for image_file in image_files:
    # Load the input image
    image_path = os.path.join(image_folder, image_file)
    image_pil = Image.open(image_path)
    img = transform(image_pil).to(device)
    image_np = np.array(image_pil)

    # Forward pass through the model
    with torch.no_grad():
        out = model(img.unsqueeze(0))

    # Apply softmax along the first dimension
    out_softmax = softmax(out, dim=1)

    # Get the index with the highest probability (argmax) along the first dimension
    out_argmax = torch.argmax(out_softmax, dim=1)
    out_argmax_np = out_argmax.cpu().detach().numpy()

    # Load the ground truth mask
    ground_truth_path = os.path.join(ground_truth_folder, image_file)
    ground_truth_pil = Image.open(ground_truth_path)
    ground_truth_np = np.array(ground_truth_pil)

    # Create an overlay using the predicted mask
    cmap = ListedColormap(['black', 'red', 'green'])  # Assuming 3 classes including background
    overlay = cmap(out_argmax_np.squeeze())
    overlay_image = (overlay[:, :, :3] * 255).astype(np.uint8)

    # Resize overlay_image to match the dimensions of image_np
    overlay_image_resized = cv2.resize(overlay_image, (image_np.shape[1], image_np.shape[0]))

    # Combine the input image with the resized overlay
    overlayed_image = cv2.addWeighted(image_np, 0.7, overlay_image_resized, 0.3, 0)

    # Plot and save the images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(image_np)
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(out_argmax_np.squeeze(), cmap='viridis')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(ground_truth_np, cmap='viridis')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(overlayed_image)
    plt.title('Overlayed Image')
    plt.axis('off')

    # Save the plot
    output_plot_path = os.path.join(output_folder, f'{os.path.splitext(image_file)[0]}_plot.png')
    plt.savefig(output_plot_path)
    plt.close()
