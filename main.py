import torch
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from dataset import CropSegmentationDataset
from models import UNet
from losses import PixelWiseCrossEntropy
from trainer import BaselineTrainer
import os

def main():

    # Define any additional transformations if needed
    train_input_transform = transforms.Compose([

        transforms.Resize((572, 572)),
        transforms.RandomResizedCrop(120),  # Adjust the crop size as needed

        transforms.ToTensor(),
    ])
    val_input_transform = transforms.Compose([
        transforms.Resize((572, 572)),
        transforms.ToTensor(),
    ])


    train_target_transform = transforms.Compose([
        transforms.Resize((388, 388)),
        transforms.RandomResizedCrop(120),  # Adjust the crop size as needed

        transforms.ToTensor(),
    ])
    val_target_transform = transforms.Compose([
        transforms.Resize((388, 388)),
        transforms.RandomResizedCrop(120),  # Adjust the crop size as needed

        transforms.ToTensor(),
    ])



    # Instantiate TID2013VOC dataset
    train_dataset = CropSegmentationDataset(transform=train_input_transform,target_transform=train_target_transform,
                                            set_type="train")
    val_dataset = CropSegmentationDataset(transform=val_input_transform,target_transform=val_target_transform,set_type="val")


    # Create DataLoader for train and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,)

    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = UNet()
    # Check if the model state_dict file exists
    model_state_dict_file = "model_state_dict.pth"
    if os.path.exists(model_state_dict_file):
        # Load the model state_dict from file
        model.load_state_dict(torch.load(model_state_dict_file))
        print("Model state_dict loaded successfully.")

    loss = PixelWiseCrossEntropy()

    optimizer = torch.optim.Adam(model.parameters())

    trainer = BaselineTrainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        use_cuda=True
    )

    train_loss = trainer.fit(train_data_loader=train_dataloader,val_data_loader=val_dataloader,epoch=20)

    trainer.save_results(val_data_loader=val_dataloader)

    print(f"Training loss is: {train_loss},")




if __name__ == "__main__":
    main()