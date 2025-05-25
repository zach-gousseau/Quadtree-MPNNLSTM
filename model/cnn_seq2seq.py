import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from model.cnn_model import CConvLSTM, CConvGRU, SplitCConvLSTM, NoCConvLSTM, CNNConv

import gc 
import random
import numpy as np
import psutil
import os


class CNNEncoder(torch.nn.Module):
    def __init__(self, input_features, hidden_size, dropout, n_layers=1, rnn_type='LSTM', n_conv_layers=3, dummy=False, kernel_size=3, padding=1):
        super().__init__()

        assert rnn_type in ['GRU', 'LSTM', 'SplitLSTM', 'NoCConvLSTM']

        if rnn_type == 'LSTM' or rnn_type == 'NoCConvLSTM':
            rnn = CConvLSTM
        elif rnn_type == 'GRU':
            rnn = CConvGRU
        elif rnn_type == 'SplitLSTM':
            rnn = SplitCConvLSTM
        else:
            raise ValueError
        
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dummy = dummy

        if not self.dummy:
            if rnn_type == 'NoCConvLSTM':
                self.rnns = nn.ModuleList([NoCConvLSTM(input_features, hidden_size, 1)] +  [NoCConvLSTM(hidden_size, hidden_size, 1) for _ in range(n_layers-1)])
            else:
                self.rnns = nn.ModuleList(
                    [rnn(input_features, hidden_size, n_conv_layers=n_conv_layers, kernel_size=kernel_size, padding=padding, name='encoder')] + \
                    [rnn(hidden_size, hidden_size, n_conv_layers=n_conv_layers, kernel_size=kernel_size, padding=padding, name='encoder') for _ in range(n_layers-1)]
                )
        
        self.dropout = nn.Dropout(dropout)

        # For CNN, we use BatchNorm instead of LayerNorm
        self.norm_h = nn.BatchNorm2d(hidden_size)
        self.norm_c = nn.BatchNorm2d(hidden_size)
        
    def forward(self, X, H=None, C=None):
        if self.dummy:
            return H, C

        # X shape: (timesteps, batch, channels, height, width)
        # Take the first timestep
        X = X[0]  # (batch, channels, height, width)

        _, hidden_layer, cell_layer = self.rnns[0](X, H=H, C=C)

        # First layer
        hidden_layer = self.norm_h(hidden_layer)

        # Subsequent layers
        hidden, cell = [hidden_layer], [cell_layer]
        for i in range(1, self.n_layers):
            _, hidden_layer, cell_layer = self.rnns[i](hidden[-1], H=None, C=None)
            hidden.append(hidden_layer)
            cell.append(cell_layer)

        hidden = torch.stack(hidden)
        cell = torch.stack(cell) if self.rnn_type != 'GRU' else [None]*len(hidden)
        return hidden, cell


class CNNDecoder(torch.nn.Module):
    def __init__(self, input_features, hidden_size, dropout, n_layers=1, rnn_type='LSTM', n_conv_layers=3, binary=False, dummy=False, multitask=True, kernel_size=3, padding=1):
        super().__init__()

        assert rnn_type in ['GRU', 'LSTM', 'SplitLSTM', 'NoCConvLSTM']

        if rnn_type == 'LSTM':
            rnn = CConvLSTM
        elif rnn_type == 'GRU':
            rnn = CConvGRU
        elif rnn_type == 'SplitLSTM':
            rnn = SplitCConvLSTM
        elif rnn_type == 'NoCConvLSTM':
            rnn = NoCConvLSTM
        else:
            raise ValueError
        
        self.rnn_type = rnn_type
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.binary = binary
        self.dummy = dummy
        self.multitask = multitask

        n_conv_layers = 1  # Hard-coded single convolutional layer in the decoder

        if not self.dummy:
            if rnn_type == 'NoCConvLSTM':
                self.rnns = nn.ModuleList([NoCConvLSTM(input_features, hidden_size, 1)] +  [NoCConvLSTM(hidden_size, hidden_size, 1) for _ in range(n_layers-1)])
            else:
                self.rnns = nn.ModuleList(
                    [rnn(input_features, hidden_size, n_conv_layers=n_conv_layers, kernel_size=kernel_size, padding=padding, name='decoder')] + \
                    [rnn(hidden_size, hidden_size, n_conv_layers=n_conv_layers, kernel_size=kernel_size, padding=padding, name='decoder') for _ in range(n_layers-1)]
                    )

        # Output layers - CNN version
        in_channels = hidden_size
        outdim = 2 if self.multitask else 1
        
        self.cnn_out1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=kernel_size, padding=padding)
        self.cnn_out2 = nn.Conv2d(in_channels=hidden_size, out_channels=outdim, kernel_size=kernel_size, padding=padding)
        
        # For concat layers processing
        self.cnn1 = nn.Conv2d(in_channels=2, out_channels=hidden_size, kernel_size=kernel_size, padding=padding)  # concat_layers_dim=2

        self.norm_o = nn.BatchNorm2d(hidden_size)
        self.norm_h = nn.BatchNorm2d(hidden_size)
        self.norm_c = nn.BatchNorm2d(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, concat_layers, y_initial, H, C):
        if self.dummy:
            if concat_layers is not None:
                X = torch.cat([X, concat_layers], dim=1)  # Concatenate along channel dimension
            X = self.cnn_out1(X)
            return X, H, C
        
        # First layer
        output, hidden_layer, cell_layer = self.rnns[0](X, H=H[0], C=C[0])

        hidden_layer = self.norm_h(hidden_layer)

        # Subsequent layers
        hidden, cell = [hidden_layer], [cell_layer]
        for i in range(1, self.n_layers):
            output, hidden_layer, cell_layer = self.rnns[i](hidden[-1], H=H[i], C=C[i])
            hidden_layer = self.norm_h(hidden_layer)
            hidden.append(hidden_layer)
            cell.append(cell_layer)

        hidden = torch.stack(hidden)
        cell = torch.stack(cell) if self.rnn_type != 'GRU' else [None]*len(hidden)

        # Use top layer's output
        output = F.leaky_relu(output)

        # Concatenate with the concat layers
        if concat_layers is not None:
            cnn_output = self.cnn(concat_layers)
            output = output + cnn_output

        # Pass output through the final CNN to reduce to desired dimensionality
        output = self.mlp_out(output)

        # Squeeze everything to (-1, 1)
        if not self.multitask:
            output = torch.tanh(output)
        
        # Add to previous step's SIC map
        output = output + y_initial
        
        # MULTITASK
        if self.multitask:
            sic = torch.sigmoid(output[:, [0]])  # First channel
            sip = output[:, [1]]  # Second channel
            output = torch.cat((sic, sip), dim=1)

        if self.binary:
            output = torch.sigmoid(output)

        return output, hidden, cell

    def mlp_out(self, x):
        x = self.cnn_out1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.cnn_out2(x)
        x = self.dropout(x)
        return x
    
    def cnn(self, x):
        x = self.cnn1(x)
        x = F.leaky_relu(x)
        return x        


class CNNSeq2Seq(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 dropout,
                 input_timesteps=3,
                 input_features=4,
                 output_timesteps=5,
                 n_layers=4,
                 n_conv_layers=2,
                 rnn_type='LSTM',
                 binary=False,
                 image_shape=None,
                 dummy=False,
                 device=None,
                 debug=False,
                 multitask=True,
                 kernel_size=3,
                 padding=1):
        super().__init__()
        
        self.encoder = CNNEncoder(
            input_features,
            hidden_size,
            dropout,
            n_layers=n_layers,
            rnn_type=rnn_type,
            n_conv_layers=n_conv_layers,
            dummy=dummy,
            kernel_size=kernel_size,
            padding=padding
            )
        
        self.decoder_1 = CNNDecoder(
            1+3 if not multitask else 2+3,  # 1 output variable + 3 (positional encoding and node_size) -> for CNN we'll use fewer features
            hidden_size,
            dropout,
            n_layers=n_layers,
            rnn_type=rnn_type,
            n_conv_layers=n_conv_layers,
            binary=binary,
            dummy=dummy,
            multitask=multitask,
            kernel_size=kernel_size,
            padding=padding
            )

        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.n_layers = n_layers
        self.debug = debug
        self.image_shape = image_shape
        self.multitask = multitask

        self.dropout = nn.Dropout(dropout)
        self.device = device

    def process_inputs(self, x, mask=None):
        """
        Process input images for CNN-LSTM
        x shape: (timesteps, batch, height, width, channels)
        """
        # Convert to CNN format: (timesteps, batch, channels, height, width)
        x = x.permute(0, 1, 4, 2, 3)
        
        # Store for later use
        self.hidden, self.cell = None, None
        
        # Encoder
        for t in range(self.input_timesteps):
            hidden, cell = self.encoder(
                X=x[[t]], 
                H=self.hidden[-1] if self.hidden is not None else None, 
                C=self.cell[-1] if self.cell is not None else None
                )
            
            self.hidden = hidden
            self.cell = cell

        # Persistence (last input frame)
        self.persistence = x[-1, :, [0]]  # Take first channel of last timestep
        
        # First input to the decoder is the last input to the encoder 
        # For CNN, we'll use fewer channels than the GNN version
        if self.multitask:
            # [SIC, SIP, x_pos, y_pos] - simplified positional encoding
            batch, _, height, width = x[-1].shape
            x_pos = torch.linspace(-1, 1, width).view(1, 1, 1, width).expand(batch, 1, height, width).to(x.device)
            y_pos = torch.linspace(-1, 1, height).view(1, 1, height, 1).expand(batch, 1, height, width).to(x.device)
            
            sic = x[-1, :, [0]]  # SIC channel
            sip = (sic > 0.15).float()  # SIP derived from SIC
            
            self.decoder_input = torch.cat([sic, sip, x_pos, y_pos], dim=1)
        else:
            # [SIC, x_pos, y_pos, constant]
            batch, _, height, width = x[-1].shape
            x_pos = torch.linspace(-1, 1, width).view(1, 1, 1, width).expand(batch, 1, height, width).to(x.device)
            y_pos = torch.linspace(-1, 1, height).view(1, 1, height, 1).expand(batch, 1, height, width).to(x.device)
            constant = torch.ones(batch, 1, height, width).to(x.device)
            
            self.decoder_input = torch.cat([x[-1, :, [0]], x_pos, y_pos, constant], dim=1)

    def unroll_output(self, unroll_steps, y, concat_layers=None, teacher_forcing_ratio=0.5, mask=None):
        """
        Unroll the decoder for multiple timesteps
        """
        outputs = []

        for t in unroll_steps:
            if self.debug:
                if self.device.type == 'cuda':
                    print(
                        f'Decoder step {t} \n' + \
                        f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0)/1024/1024/1024}GB\n" + \
                        f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(0)/1024/1024/1024}GB\n" + \
                        f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(0)/1024/1024/1024}GB",
                    )
                else:
                    pid = os.getpid()
                    python_process = psutil.Process(pid)
                    memoryUse = python_process.memory_info()[0]/2.**30
                    print('CPU memory usage:', memoryUse, 'GB', end='\r')
            
            # Process concat layers if provided
            if concat_layers is not None:
                # concat_layers should be in format (batch, channels, height, width)
                concat_layers_t = torch.cat([concat_layers[t], (torch.ones_like(concat_layers[t][:, [0]]) * t)/self.output_timesteps], dim=1)
                self.concat_layers = concat_layers_t
            else:
                concat_layers_t = None

            # Perform decoding step
            decoder = self.decoder_1
                
            output, hidden, cell = decoder(
                X=self.decoder_input,
                concat_layers=self.concat_layers if hasattr(self, 'concat_layers') else None,
                y_initial=self.persistence if not self.multitask else self.decoder_input[:, :2],
                H=self.hidden, 
                C=self.cell
                )

            outputs.append(output)

            # Decide whether to use prediction or ground truth for next step
            teacher_force = random.random() < teacher_forcing_ratio
            teacher_input = y[t] if teacher_force else None

            self.update_without_remesh(output, hidden, cell, teacher_force=teacher_force, teacher_input=teacher_input)

        return outputs

    def forward(self, x, y=None, concat_layers=None, teacher_forcing_ratio=0.5, mask=None, **kwargs):
        """
        Forward pass for CNN Seq2Seq
        x shape: (timesteps, batch, height, width, channels)
        """
        # Encoder
        self.process_inputs(x, mask=mask)
        
        # Decoder
        outputs = self.unroll_output(
            range(self.output_timesteps),
            y,
            concat_layers=concat_layers,
            teacher_forcing_ratio=teacher_forcing_ratio,
            mask=mask
            )

        return outputs, None  # Return None for mappings since we don't use graph structure

    def update_without_remesh(self, data, hidden, cell, teacher_force=False, teacher_input=None):
        """Update decoder input for next timestep"""
        if teacher_force and teacher_input is not None:
            # Use ground truth
            # Convert teacher_input from (batch, height, width, channels) to (batch, channels, height, width)
            teacher_input = teacher_input.permute(0, 3, 1, 2)
            
            # Keep positional encoding from current decoder input
            pos_encoding = self.decoder_input[:, 1:] if not self.multitask else self.decoder_input[:, 2:]
            self.decoder_input = torch.cat([teacher_input, pos_encoding], dim=1)
        else:
            # Use prediction
            pos_encoding = self.decoder_input[:, 1:] if not self.multitask else self.decoder_input[:, 2:]
            self.decoder_input = torch.cat([data, pos_encoding], dim=1)

        self.hidden = hidden
        self.cell = cell 