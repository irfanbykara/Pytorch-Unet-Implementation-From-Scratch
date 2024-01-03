from typing import Callable, List
import torch
import torch.utils.data as data
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import cv2
import matplotlib.image as mpimg
import torch.nn.functional as F


class BaselineTrainer:
    def __init__(self, model: torch.nn.Module,
                 loss: Callable,
                 optimizer: torch.optim.Optimizer,
                 validate_every=3,

                 use_cuda=True,
                 ):
        self.loss = loss
        self.validate_every = validate_every
        self.train_losses = []
        self.val_losses = []
        self.use_cuda = use_cuda
        self.optimizer = optimizer
        self.model = model

        if self.use_cuda:
            self.model = model.to(device="cuda:0")

    def fit(self, train_data_loader: data.DataLoader,
            epoch: int,
            val_data_loader: data.DataLoader,
            ):
        avg_loss = 0.
        self.model.training = True

        for e in range(epoch):
            print(f"Start epoch {e + 1}/{epoch}")
            n_batch = 0
            epoch_loss = 0.0  # Store loss for the current epoch

            for i, (x, y) in enumerate(train_data_loader):
                # Reset previous gradients
                self.optimizer.zero_grad()

                # Move data to cuda is necessary:
                if self.use_cuda:
                    x = x.to(device="cuda:0")
                    y = y.to(device="cuda:0")
                out = self.model(x)
                loss = self.loss(out, y.squeeze(1))
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()
                avg_loss += loss.item()
                epoch_loss += loss.item()
                n_batch += 1

                print(f"\r{i + 1}/{len(train_data_loader)}: loss = {loss.item()}", end='')

            # Store the average loss for the current epoch
            epoch_loss /= len(train_data_loader)
            self.train_losses.append(epoch_loss)
            # Test the model on validation dataset every validate_every epochs
            if (e + 1) % self.validate_every == 0:
                val_loss = self.evaluate(val_data_loader)
                print(f"\nValidation Loss after epoch {e + 1}: {val_loss}")
                # Assuming you have a list to store validation losses, you can append it here
                self.val_losses.append(val_loss)

            if e % 5 == 0:
                model_filename = 'model_state_dict.pth'
                torch.save(self.model.state_dict(), model_filename)

        # Plot the training loss
        # Save the model state dictionary to a file
        self.plot_training_loss(val_losses=self.val_losses)

        avg_loss /= len(train_data_loader)

        return avg_loss

    def plot_training_loss(self, val_losses=None):
        plt.plot(self.train_losses, label='Training Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.savefig('training_loss_plot.png')
        plt.show()

    def evaluate(self, val_data_loader: data.DataLoader):
        avg_loss = 0.
        self.model.eval()
        save_dir = "validation_results"
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():

            for i, (x, y) in enumerate(val_data_loader):

                if self.use_cuda:
                    x = x.to(device="cuda:0")
                    y = y.to(device="cuda:0")

                out = self.model(x)
                loss = self.loss(out, y)
                avg_loss += loss.item()



        avg_loss /= len(val_data_loader)
        return avg_loss

