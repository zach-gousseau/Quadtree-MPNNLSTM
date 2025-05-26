import numpy as np
import time
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os 
import datetime
from tqdm import tqdm
import random
import warnings

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
from torch.optim.lr_scheduler import StepLR

from model.cnn_seq2seq import CNNSeq2Seq
from model.utils import get_n_params, int_to_datetime

from abc import ABC, abstractmethod

# Import loss functions from the original mpnnlstm.py
from model.mpnnlstm import (
    SobelLoss, MSE_masked, MSE_NIIEE, NIIEE, MSE_SSIM, MSE_Sobel, 
    MSE_SIP, BCE, MSE_SIP_bin, MSE_SIP_bin_sep
)


class NextFramePredictorCNN:
    def __init__(self, 
                 experiment_name='experiment', 
                 input_features=1,
                 device=None):

        self.experiment_name = experiment_name
        self.model = None
        self.input_features = input_features 
        self.device = device


class NextFramePredictorCNNS2S(NextFramePredictorCNN):
    def __init__(self,
                 experiment_name='experiment', 
                 directory='',
                 input_features=1,
                 input_timesteps=3,
                 output_timesteps=3,
                 device=None,
                 binary=False,
                 debug=False,
                 model_kwargs={}):
        
        super().__init__(
                 experiment_name=experiment_name, 
                 input_features=input_features,
                 device=device)
        
        # Model parameters
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.input_features = input_features 
        self.binary = binary
        
        self.experiment_name = experiment_name
        self.directory = directory
        self.debug = debug
        self.device = device
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Model 
        self.model = CNNSeq2Seq(
            input_features=input_features,
            input_timesteps=input_timesteps,
            output_timesteps=output_timesteps,
            device=device,
            binary=binary,
            debug=debug,
            **model_kwargs
        ).to(device)

        # To allow calling train() multiple times
        self.training_initiated = False

    def get_n_params(self):
        return get_n_params(self.model)

    def save(self, directory=None):
        save_dir = directory if directory is not None else self.directory
        torch.save(self.model.state_dict(), os.path.join(save_dir, f'{self.experiment_name}.pth'))

    def load(self, directory):
        try:
            self.model.load_state_dict(torch.load(os.path.join(directory, f'{self.experiment_name}.pth')))
        except:
            self.model.load_state_dict(torch.load(os.path.join(directory, f'{self.experiment_name}.pth'), map_location=torch.device('cpu')))

    def initiate_training(self, lr, lr_decay, mask):
        self.loss_func = MSE_SIP() if not self.binary else torch.nn.BCELoss()
        self.loss_func_name = 'MSE_SIP' if not self.binary else 'BCE'
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=lr_decay)

        self.scaler = amp.GradScaler()
        
        self.writer = SummaryWriter('runs/' + self.experiment_name + '_' + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S"))

        self.test_loss = []
        self.train_loss = []

        self.training_initiated = True
        
        self.min_loss = np.inf
    
    def train(
        self,
        loader_train,
        loader_test,
        climatology=None,
        climatology_test=None,
        n_epochs=200,
        lr=0.01,
        lr_decay=0.95,
        mask=None,
        truncated_backprop=0,
        **kwargs  # Ignore graph_structure and other GNN-specific arguments
        ):

        image_shape = loader_train.dataset.image_shape
        
        # Convert mask to PyTorch tensor if provided
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool).to(self.device)
        
        # Initialize training only if it's the first train() call
        if not self.training_initiated:
            self.initiate_training(lr, lr_decay, mask)

        # Training loop
        st = time.time()
        batch_step = 0
        for epoch in range(n_epochs): 

            # Loop over training set
            running_loss = 0
            step = 0
            
            self.model.train()
            for x, y, launch_date in tqdm(loader_train, leave=True):

                x, y = x.squeeze(0).to(self.device), y.squeeze(0).to(self.device)
                
                if climatology is not None:
                    concat_layers = self.get_climatology_array(climatology, launch_date)
                    # Convert to CNN format: (timesteps, channels, height, width)
                    concat_layers = concat_layers.unsqueeze(1)  # Add channel dimension: (timesteps, 1, height, width)
                else:
                    concat_layers = None
                
                self.optimizer.zero_grad()
            
                with amp.autocast():
                    y_hat, _ = self.model(
                        x, 
                        y, 
                        concat_layers, 
                        teacher_forcing_ratio=0, 
                        mask=mask
                        )
                    
                    # Convert outputs to proper format
                    y_hat = torch.stack(y_hat, dim=0)  # (timesteps, batch, channels, height, width)
                    
                    # Convert to match expected format: (timesteps, batch, height, width, channels)
                    y_hat = y_hat.permute(0, 1, 3, 4, 2)
                    
                    # Squeeze batch dimension since DataLoader batch_size=1
                    y_hat = y_hat.squeeze(1)  # (timesteps, height, width, channels)
                    
                    # Apply mask if provided
                    if mask is not None:
                        # Expand mask to match y_hat dimensions
                        mask_expanded = mask.unsqueeze(0).unsqueeze(-1).expand_as(y_hat)
                        y_hat_masked = y_hat[~mask_expanded].reshape(-1, y_hat.shape[-1])
                        y_masked = y[~mask_expanded].reshape(-1, y.shape[-1])
                        loss = self.loss_func(y_hat_masked, y_masked)
                    else:
                        loss = self.loss_func(y_hat, y)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                self.writer.add_scalar("Loss/train", loss.item(), batch_step)

                step += 1
                batch_step += 1
                running_loss += loss.item()
                torch.cuda.empty_cache()

            # Loop over test set
            running_loss_test = 0
            step_test = 0
            self.model.eval()
            for x, y, launch_date in tqdm(loader_test, leave=True):

                x, y = x.squeeze(0).to(self.device), y.squeeze(0).to(self.device)

                if climatology is not None:
                    concat_layers = self.get_climatology_array(climatology_test if climatology_test is not None else climatology, launch_date)
                    concat_layers = concat_layers.unsqueeze(1)  # Add channel dimension: (timesteps, 1, height, width)
                else:
                    concat_layers = None

                with torch.no_grad():
                    with amp.autocast():
                        y_hat, _ = self.model(
                            x, 
                            y, 
                            concat_layers, 
                            teacher_forcing_ratio=0, 
                            mask=mask
                            )
                        
                        y_hat = torch.stack(y_hat, dim=0)
                        y_hat = y_hat.permute(0, 1, 3, 4, 2)
                        
                        # Squeeze batch dimension since DataLoader batch_size=1
                        y_hat = y_hat.squeeze(1)  # (timesteps, height, width, channels)
                        
                        if mask is not None:
                            mask_expanded = mask.unsqueeze(0).unsqueeze(-1).expand_as(y_hat)
                            y_hat_masked = y_hat[~mask_expanded].reshape(-1, y_hat.shape[-1])
                            y_masked = y[~mask_expanded].reshape(-1, y.shape[-1])
                            loss = self.loss_func(y_hat_masked, y_masked)
                        else:
                            loss = self.loss_func(y_hat, y)

                step_test += 1
                running_loss_test += loss

                torch.cuda.empty_cache()

            running_loss = running_loss / (step + 1)
            running_loss_test = running_loss_test / (step_test + 1)
            
            if running_loss_test < self.min_loss:
                self.save()
                self.min_loss = running_loss_test

            if np.isnan(running_loss_test.item()):
                raise ValueError('NaN loss :(')

            self.writer.add_scalar("Loss/test", running_loss_test.item(), epoch)

            self.scheduler.step()

            self.train_loss.append(running_loss)
            self.test_loss.append(running_loss_test.item())
            
            print(f"{self.experiment_name} | Epoch {epoch} train {self.loss_func_name}: {running_loss:.4f}, "+ \
                f"test {self.loss_func_name}: {running_loss_test.item():.4f}, lr: {self.scheduler.get_last_lr()[0]:.4f}, time_per_epoch: {(time.time() - st) / (epoch+1):.1f}")
        
        print(f'Finished in {(time.time() - st)/60} minutes')
        
        self.writer.flush()

        self.loss = pd.DataFrame({
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
        })

    def get_climatology_array(self, climatology, launch_date):
        """
        Get the daily climate normals for each day of the year in the output timesteps
        """
        doys = [int_to_datetime(launch_date.numpy()[0] + 8.640e13 * t).timetuple().tm_yday - 1 for t in range(1, self.output_timesteps+1)]
        out = climatology[:, :, doys]  # (height, width, timesteps)
        out = torch.moveaxis(out, -1, 0)  # (timesteps, height, width)
        return out
        
    def predict(self, loader, climatology=None, mask=None, **kwargs):
        """
        Use model in inference mode.
        """
        
        image_shape = loader.dataset.image_shape
        
        # Convert mask to PyTorch tensor if provided
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool).to(self.device)
            
        self.model.to(self.device)
        
        y_pred = []
        for x, y, launch_date in tqdm(loader, leave=False):

            x = x.squeeze(0).to(self.device)

            if climatology is not None:
                concat_layers = self.get_climatology_array(climatology, launch_date)
                concat_layers = concat_layers.unsqueeze(1)  # Add channel dimension: (timesteps, 1, height, width)
            else:
                concat_layers = None

            with torch.no_grad():
                with amp.autocast():
                    y_hat, _ = self.model(
                        x,
                        concat_layers=concat_layers, 
                        teacher_forcing_ratio=0,
                        mask=mask
                        )
                
                # Convert outputs: list of (batch, channels, height, width) -> (timesteps, batch, height, width, channels)
                y_hat = torch.stack(y_hat, dim=0).permute(0, 1, 3, 4, 2).detach().cpu()
                
                # Squeeze batch dimension since DataLoader batch_size=1
                y_hat = y_hat.squeeze(1)  # (timesteps, height, width, channels)
                
                y_hat = np.array(y_hat)
                
                y_pred.append(y_hat)

                torch.cuda.empty_cache()
            
        return np.stack(y_pred, 0)

    def score(self, x, y, rollout=None):
        pass 