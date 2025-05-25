import numpy as np
import time
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from model.utils import get_n_params, int_to_datetime


class CNNEncoder(nn.Module):
    def __init__(self, input_channels, hidden_size, kernel_size=3, dropout=0.1):
        super(CNNEncoder, self).__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class CNNDecoder(nn.Module):
    def __init__(self, input_channels, hidden_size, output_channels=1, kernel_size=3, dropout=0.1):
        super(CNNDecoder, self).__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, output_channels, kernel_size, padding=padding)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class SimpleCNNLSTM(nn.Module):
    def __init__(self, 
                 input_features, 
                 hidden_size,
                 output_features=1,
                 input_timesteps=10,
                 output_timesteps=90,
                 n_layers=1,
                 dropout=0.1,
                 kernel_size=3):
        super(SimpleCNNLSTM, self).__init__()
        
        self.encoder = CNNEncoder(input_features, hidden_size, kernel_size, dropout)
        # Add a separate encoder for the output that expects output_features channels
        self.output_encoder = CNNEncoder(output_features, hidden_size, kernel_size, dropout)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        
        # Modified decoder to accept both hidden state and persistence feature
        self.decoder = CNNDecoder(hidden_size + input_features, hidden_size, output_features, kernel_size, dropout)
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.output_features = output_features
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.n_layers = n_layers
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, input_timesteps, height, width, channels]
        batch_size, input_timesteps, height, width, channels = x.shape
        
        # Extract the last input frame for persistence
        last_input = x[:, -1]  # [batch_size, height, width, channels]
        
        # Process all timesteps at once by reshaping and using batch dimension
        x_reshaped = x.view(batch_size * input_timesteps, height, width, channels)
        # Permute to [batch_size*input_timesteps, channels, height, width] for CNN
        x_reshaped = x_reshaped.permute(0, 3, 1, 2)
        features = self.encoder(x_reshaped)  # [batch_size*input_timesteps, hidden_size, height, width]
        
        # Reshape back to separate batch and time dimensions
        features = features.view(batch_size, input_timesteps, self.hidden_size, height, width)
        
        # Reshape for LSTM: [batch*height*width, timesteps, hidden_size]
        lstm_input = features.permute(0, 3, 4, 1, 2)  # [batch, height, width, timesteps, hidden]
        lstm_input = lstm_input.reshape(batch_size * height * width, input_timesteps, self.hidden_size)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(lstm_input)  # [batch*height*width, timesteps, hidden]
        
        # Get last output and reshape to spatial form
        last_hidden = lstm_out[:, -1].view(batch_size, height, width, self.hidden_size)
        last_hidden = last_hidden.permute(0, 3, 1, 2)  # [batch, hidden, height, width]
        
        # Convert mask to tensor if it's a numpy array
        if mask is not None and isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(x.device)
        
        # Prepare mask if provided
        mask_expanded = mask.unsqueeze(0).unsqueeze(0) if mask is not None else None
        
        # Prepare persistence feature (last input)
        # Permute to [batch, channels, height, width] for CNN
        persistence = last_input.permute(0, 3, 1, 2)
        
        # Generate output sequence
        outputs = []
        current_input = None
        
        for t in range(self.output_timesteps):
            # Concatenate decoder input with persistence feature
            # For first step, use last_hidden with persistence
            if t == 0 or current_input is None:
                decoder_input = torch.cat([last_hidden, persistence], dim=1)
            else:
                # For subsequent steps, use previous output features with persistence
                decoder_input = torch.cat([current_input, persistence], dim=1)
            
            # Apply decoder
            output = self.decoder(decoder_input)
            output = torch.sigmoid(output)
            
            # Apply mask if provided
            if mask_expanded is not None:
                output = output * (~mask_expanded).float()
            
            outputs.append(output)
            
            # Prepare the next decoder input
            current_input = self.output_encoder(output)
        
        # Stack all outputs: [output_timesteps, batch_size, output_features, height, width]
        stacked_outputs = torch.stack(outputs)
        return stacked_outputs


class SimpleCNNLSTMPredictor:
    def __init__(self,
                 experiment_name='experiment', 
                 input_features=1,
                 hidden_size=32,
                 input_timesteps=10,
                 output_timesteps=90,
                 n_layers=1,
                 dropout=0.1,
                 kernel_size=3,
                 binary=False,
                 device=None):
        
        self.experiment_name = experiment_name
        self.input_features = input_features
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.binary = binary
        self.device = device
        
        # Model 
        self.model = SimpleCNNLSTM(
            input_features=input_features,
            hidden_size=hidden_size,
            output_features=1,
            input_timesteps=input_timesteps,
            output_timesteps=output_timesteps,
            n_layers=n_layers,
            dropout=dropout,
            kernel_size=kernel_size
        ).to(device)
        
        # To allow calling train() multiple times
        self.training_initiated = False

    def get_n_params(self):
        return get_n_params(self.model)

    def save(self, directory):
        torch.save(self.model.state_dict(), os.path.join(directory, f'{self.experiment_name}.pth'))

    def load(self, directory):
        try:
            self.model.load_state_dict(torch.load(os.path.join(directory, f'{self.experiment_name}.pth')))
        except:
            self.model.load_state_dict(torch.load(os.path.join(directory, f'{self.experiment_name}.pth'), map_location=torch.device('cpu')))

    def initiate_training(self, lr, lr_decay):
        self.loss_func = nn.MSELoss() if not self.binary else nn.BCELoss()
        self.loss_func_name = 'MSE' if not self.binary else 'BCE'  # For printing
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=lr_decay)
        
        self.train_loss = []
        self.test_loss = []
        
        self.training_initiated = True
    
    def train(
        self,
        loader_train,
        loader_test,
        n_epochs=200,
        lr=0.01,
        lr_decay=0.95,
        mask=None
        ):
        
        # Initialize training only if it's the first train() call
        if not self.training_initiated:
            self.initiate_training(lr, lr_decay)

        # Convert mask to tensor if it's a numpy array
        if mask is not None and isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(self.device)
            
        # Training loop
        st = time.time()
        for epoch in range(n_epochs): 

            # Loop over training set
            running_loss = 0
            step = 0
            self.model.train()
            for x, y, launch_date in tqdm(loader_train, leave=True):
                
                # Move to device
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                y_hat = self.model(x, mask=mask)
                
                # Permute y_hat to match y dimensions: [timesteps, batch, channels, h, w] -> [batch, timesteps, h, w, channels]
                y_hat = y_hat.permute(1, 0, 3, 4, 2)
                
                # Calculate loss with masking if needed
                if mask is not None:
                    mask_expanded = mask.unsqueeze(0).unsqueeze(0)
                    y_hat_masked = y_hat.masked_fill(mask_expanded, 0)
                    y_masked = y.masked_fill(mask_expanded, 0)
                    loss = self.loss_func(y_hat_masked, y_masked)
                else:
                    loss = self.loss_func(y_hat, y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                step += 1
                running_loss += loss.item()
                torch.cuda.empty_cache()

            # Evaluation on test set
            running_loss_test = 0
            step_test = 0
            self.model.eval()
            for x, y, launch_date in tqdm(loader_test, leave=True):
                
                x, y = x.to(self.device), y.to(self.device)
                
                with torch.no_grad():
                    y_hat = self.model(x, mask=mask)
                    
                    # Permute y_hat to match y dimensions
                    y_hat = y_hat.permute(1, 0, 3, 4, 2)
                    
                    # Calculate loss with same masking logic
                    if mask is not None:
                        mask_expanded = mask.unsqueeze(0).unsqueeze(0)
                        y_hat_masked = y_hat.masked_fill(mask_expanded, 0)
                        y_masked = y.masked_fill(mask_expanded, 0)
                        loss = self.loss_func(y_hat_masked, y_masked)
                    else:
                        loss = self.loss_func(y_hat, y)

                step_test += 1
                running_loss_test += loss.item()
                torch.cuda.empty_cache()

            running_loss = running_loss / (step + 1)
            running_loss_test = running_loss_test / (step_test + 1)

            if np.isnan(running_loss_test):
                raise ValueError('NaN loss :(')

            self.scheduler.step()

            self.train_loss.append(running_loss)
            self.test_loss.append(running_loss_test)
            
            print(f"{self.experiment_name} | Epoch {epoch} train {self.loss_func_name}: {running_loss:.4f}, "+ \
                f"test {self.loss_func_name}: {running_loss_test:.4f}, lr: {self.scheduler.get_last_lr()[0]:.4f}, time_per_epoch: {(time.time() - st) / (epoch+1):.1f}")
        
        print(f'Finished in {(time.time() - st)/60} minutes')
        
        self.loss = pd.DataFrame({
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
        })
        
    def predict(self, loader, mask=None):
        """
        Use model in inference mode
        """
        self.model.to(self.device)
        self.model.eval()
        
        # Convert mask to tensor if it's a numpy array
        if mask is not None and isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(self.device)
            
        y_pred = []
        for x, y, launch_date in tqdm(loader, leave=False):
            x = x.to(self.device)

            with torch.no_grad():
                y_hat = self.model(x, mask=mask)
                
                # Permute y_hat to match expected dimensions: [timesteps, batch, channels, h, w] -> [batch, timesteps, h, w, channels]
                y_hat = y_hat.permute(1, 0, 3, 4, 2)
                
                y_hat = y_hat.cpu().numpy()
                y_pred.append(y_hat)
                
                torch.cuda.empty_cache()
            
        return np.stack(y_pred, 0) 
