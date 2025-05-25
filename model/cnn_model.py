import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros

class CNNConv(nn.Module):
    """CNN equivalent of GraphConv - stacks multiple 2D convolutions"""
    def __init__(self, in_channels, out_channels, n_layers, kernel_size=3, padding=1):
        super(CNNConv, self).__init__()
        
        self.n_layers = n_layers
        
        if n_layers > 0:
            self.convolutions = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)] + \
                [nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding) for _ in range(n_layers - 1)]
            )
        else:
            self.n_layers = 0
    
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        for i in range(self.n_layers):
            x = self.convolutions[i](x)
        return x


class CConvLSTM(nn.Module):
    """CNN version of GConvLSTM - uses 2D convolutions instead of graph convolutions"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_conv_layers: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
        name='CConvLSTM'
    ):
        super(CConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.padding = padding
        self.name = name

        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):
        self.conv_x_i = CNNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        self.conv_h_i = CNNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels, 1, 1))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels, 1, 1))

    def _create_forget_gate_parameters_and_layers(self):
        self.conv_x_f = CNNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        self.conv_h_f = CNNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels, 1, 1))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels, 1, 1))

    def _create_cell_state_parameters_and_layers(self):
        self.conv_x_c = CNNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        self.conv_h_c = CNNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        self.b_c = Parameter(torch.Tensor(1, self.out_channels, 1, 1))

    def _create_output_gate_parameters_and_layers(self):
        self.conv_x_o = CNNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        self.conv_h_o = CNNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels, 1, 1))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels, 1, 1))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels, X.shape[2], X.shape[3]).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels, X.shape[2], X.shape[3]).to(X.device)
        return C

    def _calculate_input_gate(self, X, H, C):
        I = self.conv_x_i(X)
        I = I + self.conv_h_i(H)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, H, C):
        F = self.conv_x_f(X)
        F = F + self.conv_h_f(H)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, H, C, I, F):
        T = self.conv_x_c(X)
        T = T + self.conv_h_c(H)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, H, C):
        O = self.conv_x_o(X)
        O = O + self.conv_h_o(H)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Forward pass for CNN-LSTM
        
        Args:
            X: Input tensor of shape (batch, channels, height, width)
            H: Hidden state tensor of shape (batch, out_channels, height, width)
            C: Cell state tensor of shape (batch, out_channels, height, width)
            
        Returns:
            output, hidden_state, cell_state
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, H, C)
        F = self._calculate_forget_gate(X, H, C)
        C = self._calculate_cell_state(X, H, C, I, F)
        O = self._calculate_output_gate(X, H, C)
        H = self._calculate_hidden_state(O, C)
        return O, H, C


class CConvGRU(nn.Module):
    """CNN version of GConvGRU"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_conv_layers: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
        name='CConvGRU'
    ):
        super(CConvGRU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.padding = padding
        self.name = name
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = CNNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        self.conv_h_z = CNNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = CNNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        self.conv_h_r = CNNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = CNNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        self.conv_h_h = CNNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels, X.shape[2], X.shape[3]).to(X.device)
        return H

    def _calculate_update_gate(self, X, H):
        Z = self.conv_x_z(X)
        Z = Z + self.conv_h_z(H)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, H):
        R = self.conv_x_r(X)
        R = R + self.conv_h_r(H)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, H, R):
        H_tilde = self.conv_x_h(X)
        H_tilde = H_tilde + self.conv_h_h(H * R)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,  # Compatibility with LSTM 
    ) -> torch.FloatTensor:
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, H)
        R = self._calculate_reset_gate(X, H)
        H_tilde = self._calculate_candidate_state(X, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H, H, None


class SplitCConvLSTM(nn.Module):
    """CNN version of SplitGConvLSTM"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_conv_layers: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
        name='SplitCConvLSTM'
    ):
        super(SplitCConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_conv_layers = n_conv_layers
        self.name = name

        # First apply CNN, then LSTM
        self.conv = CNNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_layers=self.n_conv_layers,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # We'll need to flatten spatial dimensions for LSTM
        # The LSTM will be applied per-pixel
        self.rnn = nn.LSTM(out_channels, out_channels, 1)

    def forward(
        self,
        X: torch.FloatTensor,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        # Apply convolution
        X = self.conv(X)  # (batch, channels, height, width)
        
        # Reshape for LSTM: (batch, channels, height, width) -> (height*width, batch, channels)
        batch, channels, height, width = X.shape
        X_reshaped = X.permute(2, 3, 0, 1).reshape(height * width, batch, channels)
        
        # Prepare hidden states
        if H is not None:
            H_reshaped = H.permute(2, 3, 0, 1).reshape(height * width, batch, channels)
            C_reshaped = C.permute(2, 3, 0, 1).reshape(height * width, batch, channels)
            # LSTM expects (num_layers, batch, hidden_size)
            H_lstm = H_reshaped.permute(1, 0, 2).unsqueeze(0)  # (1, batch, height*width*channels)
            C_lstm = C_reshaped.permute(1, 0, 2).unsqueeze(0)
            outputs, (hidden, cell) = self.rnn(X_reshaped, (H_lstm, C_lstm))
        else:
            outputs, (hidden, cell) = self.rnn(X_reshaped)
        
        # Reshape back: (height*width, batch, channels) -> (batch, channels, height, width)
        outputs = outputs.reshape(height, width, batch, channels).permute(2, 3, 0, 1)
        hidden = hidden.squeeze(0).permute(1, 2, 0).reshape(batch, channels, height, width)
        cell = cell.squeeze(0).permute(1, 2, 0).reshape(batch, channels, height, width)
        
        return outputs, hidden, cell


class NoCConvLSTM(nn.LSTM):
    """CNN version of NoGConvLSTM - just a regular LSTM that ignores spatial structure"""
    def __init__(self, in_channels, out_channels, num_layers=1):
        # For CNN version, we'll treat each pixel independently
        super(NoCConvLSTM, self).__init__(in_channels, out_channels, num_layers)

    def forward(self, input, H=None, C=None):
        # input shape: (batch, channels, height, width)
        # Reshape to (batch*height*width, channels) for LSTM
        batch, channels, height, width = input.shape
        input_reshaped = input.permute(0, 2, 3, 1).reshape(batch * height * width, channels)
        
        if H is not None:
            H_reshaped = H.permute(0, 2, 3, 1).reshape(batch * height * width, self.hidden_size)
            C_reshaped = C.permute(0, 2, 3, 1).reshape(batch * height * width, self.hidden_size)
            # LSTM expects (seq_len, batch, input_size)
            o, (h, c) = super(NoCConvLSTM, self).forward(
                input_reshaped.unsqueeze(0), 
                (H_reshaped.unsqueeze(0), C_reshaped.unsqueeze(0))
            )
        else:
            o, (h, c) = super(NoCConvLSTM, self).forward(input_reshaped.unsqueeze(0))
        
        # Reshape back
        o = o.squeeze(0).reshape(batch, height, width, self.hidden_size).permute(0, 3, 1, 2)
        h = h.squeeze(0).reshape(batch, height, width, self.hidden_size).permute(0, 3, 1, 2)
        c = c.squeeze(0).reshape(batch, height, width, self.hidden_size).permute(0, 3, 1, 2)
        
        return o, h, c 