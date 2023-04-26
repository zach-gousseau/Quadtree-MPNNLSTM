import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Optional, Tuple, Union
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv, ChebConv, TransformerConv, GATConv, GATv2Conv
from torch_geometric.nn.inits import glorot, zeros

import copy

class DummyLSTM(MessagePassing):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None,
    H: OptTensor = None, C: OptTensor = None) -> Tensor:
        return x, H, C

class MHTransformerConv(TransformerConv):
    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, heads: int = 1, concat: bool = True, beta: bool = False, dropout: float = 0, edge_dim: Optional[int] = None, bias: bool = True, root_weight: bool = True, **kwargs):
        super().__init__(in_channels, out_channels, heads, concat, beta, dropout, edge_dim, bias, root_weight, **kwargs)

        self.lin = Linear(out_channels*heads, out_channels)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):

        out = super().forward(x, edge_index, edge_attr, return_attention_weights)
        out = self.lin(out)
        return out

CONVOLUTIONS = {
    'GCNConv': GCNConv,
    'TransformerConv': TransformerConv,
    'MHTransformerConv': MHTransformerConv,
    'ChebConv': ChebConv,
    'GATConv': GATConv,
    'GATv2Conv': GATv2Conv,
    'Dummy': None
}

CONVOLUTION_KWARGS = {
    'GCNConv': dict(add_self_loops=False),
    'TransformerConv': dict(heads=1, edge_dim=2, dropout=0.1, concat=False),
    'MHTransformerConv': dict(heads=3, edge_dim=2, dropout=0.1),
    'ChebConv': dict(K=3, normalization='sym', bias=True),
    'GATConv': dict(heads=1, edge_dim=2),
    'GATv2Conv': dict(heads=1, edge_dim=2),
    'Dummy': dict(),
}

class GraphConv(nn.Module):
    def __init__(self, convolution_type, in_channels, out_channels, n_layers):

        super(GraphConv, self).__init__()

        self.convolution_type = convolution_type
        self.n_layers = n_layers

        conv_func = CONVOLUTIONS[convolution_type]
        conv_kwargs = CONVOLUTION_KWARGS[convolution_type]

        if convolution_type is not 'Dummy':
            self.convolutions = nn.ModuleList(
                [conv_func(in_channels, out_channels, **conv_kwargs)] + \
                [conv_func(out_channels, out_channels, **conv_kwargs) for  _ in range(n_layers - 1)]
                )
        else:
            self.n_layers = 0

    
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor = None):
        for i in range(self.n_layers):
            x = self.convolutions[i](x, edge_index, edge_attr)
        return x

    


class GConvLSTM(nn.Module):
    r"""
    
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_conv_layers: int = 1, 
        convolution_type='GCNConv'
    ):
        super(GConvLSTM, self).__init__()

        assert convolution_type in CONVOLUTIONS

        self.convolution_type = convolution_type
        self.n_conv_layers = n_conv_layers

        self.in_channels = in_channels
        self.out_channels = out_channels

        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):

        self.conv_x_i = GraphConv(
            convolution_type=self.convolution_type,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_layers = self.n_conv_layers
        )

        self.conv_h_i = GraphConv(
            convolution_type=self.convolution_type,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            n_layers = self.n_conv_layers
        )

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_x_f = GraphConv(
            convolution_type=self.convolution_type,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_layers = self.n_conv_layers
        )

        self.conv_h_f = GraphConv(
            convolution_type=self.convolution_type,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            n_layers = self.n_conv_layers
        )

        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):

        self.conv_x_c = GraphConv(
            convolution_type=self.convolution_type,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_layers = self.n_conv_layers
        )

        self.conv_h_c = GraphConv(
            convolution_type=self.convolution_type,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            n_layers = self.n_conv_layers
        )

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):

        self.conv_x_o = GraphConv(
            convolution_type=self.convolution_type,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_layers = self.n_conv_layers
        )

        self.conv_h_o = GraphConv(
            convolution_type=self.convolution_type,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            n_layers = self.n_conv_layers
        )

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

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
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C):
        I = self.conv_x_i(X, edge_index, edge_weight)
        I = I + self.conv_h_i(H, edge_index, edge_weight)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C):
        F = self.conv_x_f(X, edge_index, edge_weight)
        F = F + self.conv_h_f(H, edge_index, edge_weight)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F):
        T = self.conv_x_c(X, edge_index, edge_weight)
        T = T + self.conv_h_c(H, edge_index, edge_weight)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C):
        O = self.conv_x_o(X, edge_index, edge_weight)
        O = O + self.conv_h_o(H, edge_index, edge_weight)
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
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C)
        H = self._calculate_hidden_state(O, C)
        return O, H, C


class MPNNLSTM(nn.Module):

    def __init__(self, hidden_size, dropout, input_timesteps=3, input_features=4, output_features=1):
        super(MPNNLSTM, self).__init__()
        self.dropout = dropout
        self.input_timesteps = input_timesteps

        self.convolution1 = GCNConv(input_features, hidden_size)
        self.convolution2 = GCNConv(hidden_size, hidden_size)
        self.convolution3 = GCNConv(hidden_size, hidden_size)
        # self.convolution4 = GCNConv(hidden_size, hidden_size)
        
        # self.bn1 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
        # self.bn2 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
        # self.bn3 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
        # self.bn4 = nn.BatchNorm1d(hidden_size, track_running_stats=False)

        self.bn1 = nn.LayerNorm(hidden_size)
        self.bn2 = nn.LayerNorm(hidden_size)
        self.bn3 = nn.LayerNorm(hidden_size)

        self.recurrents = nn.LSTM(hidden_size, hidden_size, 4)

        self.lin1 = nn.Linear(hidden_size+input_timesteps, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_features)

        # self.lin1 = nn.Linear(hidden_size+input_timesteps, 1)


    def forward(self, X, edge_index, edge_weight=None):

        # Convolutions, BN
        C = []
        for i in range(X.shape[0]):
            H = F.relu(self.convolution1(X[i], edge_index, edge_weight=edge_weight))
            H = self.bn1(H)
            H = F.dropout(H, p=self.dropout, training=self.training)

            H = F.relu(self.convolution2(H, edge_index, edge_weight=edge_weight))
            H = self.bn2(H)
            H = F.dropout(H, p=self.dropout, training=self.training)

            H = F.relu(self.convolution3(H, edge_index, edge_weight=edge_weight))
            H = self.bn3(H)
            H = F.dropout(H, p=self.dropout, training=self.training)

            # H = F.relu(self.convolution4(H, edge_index, edge_weight=edge_weight))
            # H = self.bn4(H)
            # H = F.dropout(H, p=self.dropout, training=self.training)
            C.append(H)

        C = torch.stack(C)

        _, (H, _) = self.recurrents(C)
        
        H = F.relu(H[-1])  # Keep only last hidden state of last layer

        S = X[:, :, 0].T  # Skip connection thang

        H = torch.cat([H, S], dim=-1)

        # FC for output
        H = self.lin1(H)
        H = F.relu(H)
        H = self.lin2(H)
        # H = F.relu(H)
        
        # Dropout and sigmoid out
        H = F.dropout(H, p=self.dropout, training=self.training)
        H = torch.sigmoid(H)
        # H = F.relu(H)
        return H

class MPNNLSTMI(nn.Module):
    def __init__(self, hidden_size, dropout, input_timesteps=3, input_features=4, n_layers=2, output_features=1):
        super(MPNNLSTMI, self).__init__()
        # self.recurrent_1 = GConvLSTM(input_features, hidden_size)
        # self.recurrent_2 = GConvLSTM(hidden_size, hidden_size)

        self.recurrents = nn.ModuleList([GConvLSTM(input_features, hidden_size)] + [GConvLSTM(hidden_size, hidden_size) for _ in range(n_layers-1)])

        self.bn1 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
        # self.bn2 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
        # self.bn3 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
        # self.bn4 = nn.BatchNorm1d(hidden_size, track_running_stats=False)

        self.lin1 = nn.Linear(hidden_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_features)

        # self.lin1 = nn.Linear(hidden_size, 1)

        self.dropout = dropout

        self.input_timesteps = input_timesteps

        self.n_layers = n_layers

    def forward(self,  X, edge_index, edge_weight=None):

        # Process the sequence of graphs with our 2 GConvLSTM layers
        # Initialize hidden and cell states to None so they are properly
        # initialized automatically in the GConvLSTM layers.
        # h1, c1, h2, c2 = None, None, None, None
        hs = [None] * self.n_layers
        cs = [None] * self.n_layers
        for x in X:
            _, h, c = self.recurrents[0](x, edge_index, edge_weight, H=hs[0], C=hs[1])
            hs[0], cs[0] = h, c
            for i in range(1, self.n_layers):
                _, h, c = self.recurrents[i](hs[i-1], edge_index, edge_weight, H=hs[i], C=cs[i])
                hs[i], cs[i] = h, c

        # Use the final hidden state output of 2nd recurrent layer for input to classifier
        x = F.relu(hs[-1])
        x = self.bn1(x)
        x = self.lin1(x)
        x = F.relu(x)
        # x = self.bn1(x)
        x = self.lin2(x)
        # x = torch.sigmoid(x)
        # x = F.relu(x)
        # x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sigmoid(x)



        # for x in X:
        #     _, h1, c1 = self.recurrent_1(x, edge_index, edge_weight, H=h1, C=c1)
        #     # h1 = self.bn1(h1)
        #     # c1 = self.bn2(c1)
        #     # Feed hidden state output of first layer to the 2nd layer
        #     _, h2, c2 = self.recurrent_2(h1, edge_index, edge_weight, H=h2, C=c2)
        #     # h2 = self.bn3(h2)
        #     # c2 = self.bn4(c2)

        # # Use the final hidden state output of 2nd recurrent layer for input to classifier
        # x = F.relu(h2)
        # x = self.bn1(x)
        # x = self.lin1(x)
        # x = F.relu(x)
        # # x = self.bn1(x)
        # x = self.lin2(x)
        # # x = torch.sigmoid(x)
        # # x = F.relu(x)
        # # x = self.lin2(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = torch.sigmoid(x)
        return x