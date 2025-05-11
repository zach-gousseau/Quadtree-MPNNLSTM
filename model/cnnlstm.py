import numpy as np
import time
import pandas as pd
import os
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from model.utils import get_n_params, int_to_datetime


class CNNEncoder(nn.Module):
    def __init__(self, input_channels, hidden_size, kernel_size=3, dropout=0.1):
        super(CNNEncoder, self).__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        return x


class CNNDecoder(nn.Module):
    def __init__(self, input_channels, hidden_size, output_channels=1, kernel_size=3, dropout=0.1):
        super(CNNDecoder, self).__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.conv3 = nn.Conv2d(hidden_size, output_channels, kernel_size, padding=padding)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.conv3(x)
        return x


class CNNLSTM(nn.Module):
    def __init__(self, 
                 input_features, 
                 hidden_size,
                 output_features=1,
                 n_layers=2,
                 dropout=0.1,
                 kernel_size=3):
        super(CNNLSTM, self).__init__()
        
        self.encoder = CNNEncoder(input_features, hidden_size, kernel_size, dropout)
        
        # LSTM layer takes flattened CNN features
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        
        self.decoder = CNNDecoder(hidden_size, hidden_size, output_features, kernel_size, dropout)
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.output_features = output_features
        self.n_layers = n_layers
        
    def forward(self, x, hidden=None):
        # x shape: [input_timesteps, batch_size, channels, height, width]
        input_timesteps, batch_size, channels, height, width = x.shape
        
        # Process each timestep with the CNN encoder
        cnn_features = []
        for t in range(input_timesteps):
            # CNN encoder
            features = self.encoder(x[t])  # [batch_size, hidden_size, height, width]
            
            # Reshape for LSTM: keep spatial dimensions but make hidden_size the channel dim
            # This preserves spatial information while processing temporal data
            cnn_features.append(features)
        
        # Stack timesteps for LSTM
        cnn_features = torch.stack(cnn_features)  # [input_timesteps, batch_size, hidden_size, height, width]
        
        # For LSTM processing, we need to reshape to [batch_size, timesteps, features]
        # where features is hidden_size*height*width to preserve spatial information
        lstm_input = cnn_features.permute(1, 0, 2, 3, 4)  # [batch_size, timesteps, hidden_size, height, width]
        lstm_input = lstm_input.reshape(batch_size, input_timesteps, self.hidden_size, -1)  # [batch_size, timesteps, hidden_size, height*width]
        
        # Process each spatial position with the same LSTM
        # Reshape to treat each spatial position as a separate sequence
        spatial_dim = lstm_input.shape[-1]
        lstm_input = lstm_input.reshape(batch_size, input_timesteps, self.hidden_size * spatial_dim)
        
        # Apply LSTM
        if hidden is None:
            lstm_out, (hidden, cell) = self.lstm(lstm_input)
        else:
            lstm_out, (hidden, cell) = self.lstm(lstm_input, hidden)
            
        # Take only the output from the final timestep
        lstm_out = lstm_out[:, -1]  # [batch_size, hidden_size * (height*width)]
        
        # Reshape back to spatial form for decoder
        lstm_out = lstm_out.reshape(batch_size, self.hidden_size, height, width)
        
        # Apply decoder
        output = self.decoder(lstm_out)  # [batch_size, output_features, height, width]
        
        # Apply sigmoid for ice concentration output (0-1 range)
        output = torch.sigmoid(output)
        
        return output, (hidden, cell)


class CNNLSTMSeq2Seq(nn.Module):
    def __init__(self, 
                 input_features, 
                 hidden_size,
                 output_features=1,
                 input_timesteps=10,
                 output_timesteps=90,
                 n_layers=2,
                 dropout=0.1,
                 kernel_size=3,
                 binary=False,
                 device=None):
        super(CNNLSTMSeq2Seq, self).__init__()
        
        self.encoder = CNNEncoder(input_features, hidden_size, kernel_size, dropout)
        
        # LSTM layer for encoder
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        
        # LSTM layer for decoder
        self.decoder_lstm = nn.LSTM(1 + hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        
        # CNN decoder
        self.decoder = CNNDecoder(hidden_size, hidden_size, output_features, kernel_size, dropout)
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.output_features = output_features
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.n_layers = n_layers
        self.binary = binary
        self.device = device
        
    def forward(self, x, y=None, concat_layers=None, teacher_forcing_ratio=0.5, mask=None):
        # x shape: [batch_size, input_timesteps, channels, height, width]
        batch_size, input_timesteps, channels, height, width = x.shape
        
        # Encoder part
        # -------------
        # Process each timestep with the CNN encoder
        cnn_features = []
        for t in range(input_timesteps):
            # CNN encoder
            features = self.encoder(x[:, t])  # [batch_size, hidden_size, height, width]
            cnn_features.append(features)
        
        # Stack timesteps for LSTM
        cnn_features = torch.stack(cnn_features, dim=1)  # [batch_size, input_timesteps, hidden_size, height, width]
        
        # Reshape for LSTM: [batch_size, timesteps, features]
        lstm_input = cnn_features.permute(0, 1, 2, 3, 4)  # [batch_size, timesteps, hidden_size, height, width]
        
        # Flatten spatial dimensions
        lstm_input = lstm_input.reshape(batch_size, input_timesteps, self.hidden_size, -1)
        spatial_dim = lstm_input.shape[-1]
        lstm_input = lstm_input.reshape(batch_size, input_timesteps, self.hidden_size * spatial_dim)
        
        # Apply encoder LSTM
        _, (hidden, cell) = self.encoder_lstm(lstm_input)
        
        # Decoder part
        # -------------
        outputs = []
        
        # Initialize decoder input with last channel from input (usually ice concentration)
        decoder_input = x[:, -1, 0:1]  # [batch_size, 1, height, width]
        
        # For each output timestep
        for t in range(self.output_timesteps):
            # Use ground truth with teacher forcing or previous output
            use_teacher_forcing = True if y is not None and torch.rand(1).item() < teacher_forcing_ratio else False
            
            if use_teacher_forcing and t > 0:
                decoder_input = y[:, t-1, 0:1]  # Use ground truth from previous timestep
            
            # Encode the input with CNN
            decoder_features = self.encoder(decoder_input)  # [batch_size, hidden_size, height, width]
            
            # Reshape for LSTM
            decoder_features = decoder_features.reshape(batch_size, self.hidden_size, -1)
            decoder_features = decoder_features.reshape(batch_size, 1, self.hidden_size * spatial_dim)
            
            # Add climatology if provided
            if concat_layers is not None:
                clim_layer = concat_layers[t].unsqueeze(1)  # [batch_size, 1, height, width]
                clim_flat = clim_layer.reshape(batch_size, 1, -1)
                decoder_features = torch.cat([decoder_features, clim_flat], dim=2)
            
            # Apply decoder LSTM
            lstm_out, (hidden, cell) = self.decoder_lstm(decoder_features, (hidden, cell))
            
            # Reshape LSTM output back to spatial form
            lstm_out = lstm_out.reshape(batch_size, self.hidden_size, height, width)
            
            # Apply CNN decoder
            output = self.decoder(lstm_out)  # [batch_size, output_features, height, width]
            
            # Apply sigmoid for normalized output
            output = torch.sigmoid(output)
            
            # Store output
            outputs.append(output)
            
            # Use output as next input
            decoder_input = output
        
        # Stack all outputs: [output_timesteps, batch_size, output_features, height, width]
        outputs = torch.stack(outputs)
        
        # Perform masking if needed
        if mask is not None:
            mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand_as(outputs)
            outputs = outputs.masked_fill(mask_expanded, 0)
            
        return outputs


class NextFramePredictorCNNLSTM:
    def __init__(self,
                 experiment_name='experiment', 
                 input_features=4,
                 hidden_size=32,
                 input_timesteps=10,
                 output_timesteps=90,
                 n_layers=2,
                 dropout=0.1,
                 kernel_size=3,
                 binary=False,
                 debug=False,
                 device=None):
        
        self.experiment_name = experiment_name
        self.input_features = input_features
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.binary = binary
        self.debug = debug
        self.device = device
        
        # Model 
        self.model = CNNLSTMSeq2Seq(
            input_features=input_features,
            hidden_size=hidden_size,
            output_features=1,
            input_timesteps=input_timesteps,
            output_timesteps=output_timesteps,
            n_layers=n_layers,
            dropout=dropout,
            kernel_size=kernel_size,
            binary=binary,
            device=device
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
        
        self.writer = SummaryWriter('runs/' + self.experiment_name + '_' + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S"))

        self.test_loss = []
        self.train_loss = []

        self.training_initiated = True
    
    def train(
        self,
        loader_train,
        loader_test,
        climatology=None,
        n_epochs=200,
        lr=0.01,
        lr_decay=0.95,
        mask=None,
        high_interest_region=None,
        truncated_backprop=0
        ):
        
        # Initialize training only if it's the first train() call
        if not self.training_initiated:
            self.initiate_training(lr, lr_decay)

        # Training loop
        st = time.time()
        batch_step = 0
        for epoch in range(n_epochs): 

            # Loop over training set
            running_loss = 0
            step = 0
            for x, y, launch_date in tqdm(loader_train, leave=True):
                
                # Move to device and reshape to [batch_size, timesteps, channels, height, width]
                x, y = x.to(self.device), y.to(self.device)
                
                if climatology is not None:
                    concat_layers = self.get_climatology_array(climatology, launch_date)
                else:
                    concat_layers = None
                
                self.optimizer.zero_grad()
                
                # Forward pass
                y_hat = self.model(x, y, concat_layers, teacher_forcing_ratio=0.5, mask=mask)
                
                # Calculate loss
                # If mask is provided, only calculate loss on non-masked areas
                if mask is not None:
                    # Expand mask to match y_hat dimensions
                    mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand_as(y_hat)
                    loss = self.loss_func(y_hat.masked_select(~mask_expanded), y.masked_select(~mask_expanded))
                else:
                    loss = self.loss_func(y_hat, y)
                
                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()
                
                if self.debug:
                    # Log gradient norms
                    grad_norms = torch.norm(torch.stack([torch.norm(param.grad.detach()) 
                                                         for param in self.model.parameters() 
                                                         if param.grad is not None]))
                    self.writer.add_scalar("Grad/norm", grad_norms, batch_step)
                
                self.writer.add_scalar("Loss/train", loss.item(), batch_step)

                step += 1
                batch_step += 1
                running_loss += loss.item()
                torch.cuda.empty_cache()

            # Loop over test set
            running_loss_test = 0
            step_test = 0
            for x, y, launch_date in tqdm(loader_test, leave=True):
                
                x, y = x.to(self.device), y.to(self.device)

                if climatology is not None:
                    concat_layers = self.get_climatology_array(climatology, launch_date)
                else:
                    concat_layers = None

                with torch.no_grad():
                    y_hat = self.model(x, y=None, concat_layers=concat_layers, teacher_forcing_ratio=0, mask=mask)
                    
                    # Calculate loss
                    if mask is not None:
                        mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand_as(y_hat)
                        loss = self.loss_func(y_hat.masked_select(~mask_expanded), y.masked_select(~mask_expanded))
                    else:
                        loss = self.loss_func(y_hat, y)

                step_test += 1
                running_loss_test += loss.item()
                torch.cuda.empty_cache()

            running_loss = running_loss / (step + 1)
            running_loss_test = running_loss_test / (step_test + 1)

            if np.isnan(running_loss_test):
                raise ValueError('NaN loss :(')

            self.writer.add_scalar("Loss/test", running_loss_test, epoch)

            self.scheduler.step()

            self.train_loss.append(running_loss)
            self.test_loss.append(running_loss_test)
            
            print(f"{self.experiment_name} | Epoch {epoch} train {self.loss_func_name}: {running_loss:.4f}, "+ \
                f"test {self.loss_func_name}: {running_loss_test:.4f}, lr: {self.scheduler.get_last_lr()[0]:.4f}, time_per_epoch: {(time.time() - st) / (epoch+1):.1f}")
        
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
        doys = [int_to_datetime(launch_date.numpy()[0] + 8.640e13 * t).timetuple().tm_yday - 1 
                for t in range(0, self.output_timesteps)]

        out = climatology[:, doys]
        out = torch.moveaxis(out, 0, -1)
        return out
        
    def predict(self, loader, climatology=None, mask=None, high_interest_region=None, graph_structure=None):
        """
        Use model in inference mode.
        """
        self.model.to(self.device)
        self.model.eval()
        
        y_pred = []
        for x, y, launch_date in tqdm(loader, leave=False):
            x = x.to(self.device)

            if climatology is not None:
                concat_layers = self.get_climatology_array(climatology, launch_date)
            else:
                concat_layers = None

            with torch.no_grad():
                y_hat = self.model(
                    x,
                    y=None,
                    concat_layers=concat_layers, 
                    teacher_forcing_ratio=0,
                    mask=mask
                )
                
                y_hat = y_hat.cpu().numpy()
                y_pred.append(y_hat)
                
                torch.cuda.empty_cache()
            
        return np.stack(y_pred, 0) 