import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.nn.inits import glorot, zeros

from graph_functions import image_to_graph, flatten, create_graph_structure, unflatten
from utils import add_positional_encoding

import gc 
import random
import numpy as np

from model import GConvLSTM, GConvLSTM_Cheb

# class Encoder(torch.nn.Module):
#     def __init__(self, input_features, hidden_size, dropout, n_layers=1):
#         super().__init__()
        
#         self.hidden_size = hidden_size
        
#         self.n_layers = n_layers

#         self.rnn = GConvLSTM(input_features, hidden_size)
#         self.bn_h = nn.LayerNorm(hidden_size)
#         self.bn_c = nn.LayerNorm(hidden_size)
        
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, X, edge_index, edge_weight, H=None, C=None):

#         _, hidden, cell = self.rnn(X, edge_index, edge_weight, H=H, C=C)

#         hidden = self.bn_h(hidden)
#         cell = self.bn_c(cell)

#         # hidden = torch.unsqueeze(hidden, 0)
#         # cell = torch.unsqueeze(cell, 0)
        
#         return hidden, cell


# class Decoder(torch.nn.Module):
#     def __init__(self, input_features, hidden_size, dropout, n_layers=1):
#         super().__init__()
        
#         self.input_features = input_features
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
        
#         self.rnn = GConvLSTM(input_features, hidden_size)
        
#         self.fc_out = torch.nn.Linear(hidden_size, 1)  # Assuming output has 1 dimension
#         self.bn_o = nn.LayerNorm(hidden_size)

#         self.bn_c = nn.LayerNorm(hidden_size)
#         self.bn_h = nn.LayerNorm(hidden_size)
        
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, X, edge_index, edge_weight, skip, H, C):


#         output, hidden, cell = self.rnn(X, edge_index, edge_weight, H=H, C=C)
        
#         output = output.squeeze(0)  # Use top layer's output

#         output = F.relu(output)
#         output = self.bn_o(output)

#         hidden = self.bn_h(hidden)
#         cell = self.bn_c(cell)

#         # skip connection
#         # output = torch.cat([output, skip], dim=-1)
#         prediction = self.fc_out(output)
#         prediction = torch.sigmoid(prediction)

#         return prediction, hidden, cell

class Encoder(torch.nn.Module):
    def __init__(self, input_features, hidden_size, dropout, n_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.n_layers = n_layers

        self.rnns = nn.ModuleList([GConvLSTM(input_features, hidden_size)] + [GConvLSTM(hidden_size, hidden_size) for _ in range(n_layers-1)])
        
        self.dropout = nn.Dropout(dropout)

        # self.bn1 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
        self.norm_h = nn.LayerNorm(hidden_size)
        self.norm_c = nn.LayerNorm(hidden_size)
        
    def forward(self, X, edge_index, edge_weight, H=None, C=None):
        X = X.squeeze(0)

        _, hidden_layer, cell_layer = self.rnns[0](X, edge_index, edge_weight, H=H, C=C)
        hidden_layer, cell_layer = hidden_layer.squeeze(0), cell_layer.squeeze(0)
        hidden_layer = self.norm_h(hidden_layer)
        cell_layer = self.norm_c(cell_layer)

        hidden, cell = [hidden_layer], [cell_layer]
        for i in range(1, self.n_layers):
            _, hidden_layer, cell_layer = self.rnns[i](hidden[-1], edge_index, edge_weight, H=hidden_layer, C=cell_layer)

            # hidden_layer = self.bn1(hidden_layer)
            hidden_layer = self.norm_h(hidden_layer)
            cell_layer = self.norm_c(cell_layer)

            hidden.append(hidden_layer)
            cell.append(cell_layer)

        hidden = torch.stack(hidden)
        cell = torch.stack(cell)

        if hidden.isnan().any():
            raise ValueError
        
        return hidden, cell

class Decoder(torch.nn.Module):
    def __init__(self, input_features, hidden_size, dropout, n_layers=1, add_skip=True):
        super().__init__()
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.rnns = nn.ModuleList([GConvLSTM(input_features, hidden_size)] + [GConvLSTM(hidden_size, hidden_size) for _ in range(n_layers-1)])

        hidden_size_in = hidden_size + 1 if add_skip else hidden_size
        
        self.fc_out1 = torch.nn.Linear(hidden_size_in, hidden_size)
        self.fc_out2 = torch.nn.Linear(hidden_size, 1)
        self.norm_o = nn.LayerNorm(hidden_size)
        self.norm_h = nn.LayerNorm(hidden_size)
        self.norm_c = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, edge_index, edge_weight, skip, H, C):
        output, hidden_layer, cell_layer = self.rnns[0](X, edge_index, edge_weight, H=H[[0]], C=C[[0]])

        hidden_layer = self.norm_h(hidden_layer)
        cell_layer = self.norm_c(cell_layer)
        hidden, cell = [hidden_layer], [cell_layer]

        for i in range(1, self.n_layers):
            output, hidden_layer, cell_layer = self.rnns[i](hidden[-1], edge_index, edge_weight, H=H[[i]], C=C[[i]])

            hidden_layer = self.norm_h(hidden_layer)
            cell_layer = self.norm_c(cell_layer)

            hidden.append(hidden_layer)
            cell.append(cell_layer)

        hidden = torch.cat(hidden)
        cell = torch.cat(cell)
        
        output = output.squeeze(0)  # Use top layer's output
        output = self.norm_o(output)
        output = F.relu(output)
        if skip is not None:
            output = torch.cat([output, skip], dim=-1)
        output = self.fc_out1(output)
        output = F.relu(output)
        output = self.fc_out2(output)
        output = torch.sigmoid(output)
        return output, hidden, cell
        # return output, hidden, cell
        

class Seq2Seq(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 dropout,
                 thresh,
                 input_timesteps=3,
                 input_features=4,
                 output_timesteps=5,
                 n_layers=4,
                 transform_func=None,
                 condition='max_larger_than',
                 remesh_input=False,
                 device=None):
        super().__init__()
        
        self.encoder = Encoder(input_features, hidden_size, dropout, n_layers=n_layers)
        self.decoder = Decoder(1+3, hidden_size, dropout, n_layers=n_layers)  # 1 output variable + 3 (positional encoding and node size)

        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.n_layers = n_layers
        self.condition = condition
        self.remesh_input = remesh_input

        self.thresh = thresh
        self.transform_func = transform_func

        self.graph = None

        self.device = device
        
    def forward(self, x, y=None, skip=None, teacher_forcing_ratio=0.5, mask=None):
        num_samples, w, h, c = x.shape
        image_shape = (w, h)
        
        if self.remesh_input:
            x0 = add_positional_encoding(x[[0]])
            graph_structure = image_to_graph(x0, thresh=self.thresh, mask=mask, transform_func=self.transform_func, condition=self.condition)
        else:
            x = add_positional_encoding(x)
            graph_structure = image_to_graph(x, thresh=self.thresh, mask=mask, transform_func=self.transform_func, condition=self.condition)

        self.graph = create_graph_structure(graph_structure['graph_nodes'], graph_structure['distances'])

        self.graph.x = graph_structure['data']

        self.graph.graph_structure = graph_structure
        self.graph.image_shape = image_shape
        self.graph.to(self.device)
        
        # Lists to store decoder outputs
        outputs = []
        outputs_graph_structures = []
        
        # Encoder ------------------------------------------------------------------------------------------------------------
        self.graph.hidden, self.graph.cell = None, None
        for t in range(self.input_timesteps):

            # Perform encoding step
            hidden, cell = self.encoder(
                X=self.graph.x if self.remesh_input else self.graph.x[[t]], 
                edge_index=self.graph.edge_index, 
                edge_weight=self.graph.edge_weight, 
                H=self.graph.hidden[-1] if hasattr(self.graph, 'hidden') else None, 
                C=self.graph.cell[-1] if hasattr(self.graph, 'cell') else None
                )

            if t < self.input_timesteps:
                if self.thresh != -np.inf:
                    if self.remesh_input:
                        self.do_remesh_input(x[[t+1]], hidden, cell, mask)
                    else:
                        if hidden.isnan().any():
                            raise ValueError
                        self.graph.hidden = hidden
                        self.graph.cell = cell

        
        # Decoder ------------------------------------------------------------------------------------------------------------
        # First input to the decoder is the last input to the encoder 
        self.graph.x = self.graph.x[[-1]][..., [0, -1, -2, -3]]

        # Skip connection is also the last input to the encoder
        # self.graph.skip = self.graph.x[[-1]][..., [0, -1, -2, -3]]

        for t in range(self.output_timesteps):
            
            if skip is not None:
                self.graph.skip = flatten(skip[[t]].unsqueeze(-1), self.graph.graph_structure['mapping'], self.graph.graph_structure['n_pixels_per_node']).squeeze(0)

            self.graph.to(self.device)

            # Perform decoding step
            output, hidden, cell = self.decoder(
                X=self.graph.x,
                edge_index=self.graph.edge_index,
                edge_weight=self.graph.edge_weight, 
                skip=self.graph.skip if hasattr(self.graph, 'skip') else None,
                H=self.graph.hidden, 
                C=self.graph.cell
                )

            # This is the prediction we are outputting. 
            outputs.append(output)
            outputs_graph_structures.append(self.graph.graph_structure)

            output = output.unsqueeze(0)#.to(self.device)

            # Ddecide whether to use the prediction or ground truth for the input to the next rollout step
            teacher_force = random.random() < teacher_forcing_ratio
            teacher_input = y[[t]] if teacher_force else None

            if self.thresh != -np.inf:
                self.do_remesh(output, hidden, cell, mask, teacher_force=teacher_force, teacher_input=teacher_input)
            else:
                self.update_without_remesh(output, teacher_force=teacher_force, teacher_input=teacher_input)
            
        return outputs, outputs_graph_structures

    def update_without_remesh(self, output, teacher_force=False, teacher_input=None):
        # curr_graph_structure does not change but the input (curr_graph.x) does change
        if teacher_force:
            teacher_input = add_positional_encoding(teacher_input)
            input_x = flatten(teacher_input, self.graph.graph_structure['mapping'], self.graph.graph_structure['n_pixels_per_node'])

            self.graph.x = torch.cat((self.graph.x[..., 1:], torch.from_numpy(input_x[..., [0]])), -1)
            # self.graph.skip = torch.from_numpy(input_x[0, :, :1])
            
        else:
            self.graph.x = torch.cat((self.graph.x[..., 1:], output), -1)
            # self.graph.skip = output


    def do_remesh(self, data, hidden, cell, mask=None, teacher_force=False, teacher_input=None):
        image_shape = self.graph.image_shape

        # Output is a prediction on the original graph structure
        # First convert it back to its grid representation
        # We also convert the hidden and cell state in the same way 
        data_img = unflatten(data, self.graph.graph_structure['mapping'], image_shape)
        hidden_img = unflatten(hidden, self.graph.graph_structure['mapping'], image_shape)
        cell_img = unflatten(cell, self.graph.graph_structure['mapping'], image_shape)

        # Then we convert it back to a graph representation where the graph is determined by
        # its own values (rather than the one created by the input images / previous step)
        if teacher_force:
            teacher_input = add_positional_encoding(teacher_input)
            graph_structure = image_to_graph(teacher_input, thresh=self.thresh, mask=mask, transform_func=self.transform_func, condition=self.condition)
        else:
            data_img = add_positional_encoding(data_img)  # Add pos. embedding
            graph_structure = image_to_graph(data_img, thresh=self.thresh, mask=mask, transform_func=self.transform_func, condition=self.condition)

        # skip = graph_structure['data'][:, :, [0]]

        # TODO: Why do I need to do this..?
        # hidden_img, cell_img = hidden_img.to(self.device), cell_img.to(self.device)
        # graph_structure['mapping'] = graph_structure['mapping'].to(self.device)
        # graph_structure['n_pixels_per_node'] = graph_structure['n_pixels_per_node'].to(self.device)

        # Use the graph structure to convert the hidden and cell states to their graph representations
        hidden_img, cell_img = torch.swapaxes(hidden_img, 0, -1), torch.swapaxes(cell_img, 0, -1)
        hidden = flatten(hidden_img, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
        cell = flatten(cell_img, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
        hidden, cell = torch.swapaxes(hidden, 0, -1), torch.swapaxes(cell, 0, -1)

        # Create a graph object for input into next rollout
        self.graph = create_graph_structure(graph_structure['graph_nodes'], graph_structure['distances'])
        self.graph.x = graph_structure['data']
        # self.graph.skip = skip
        self.graph.graph_structure = graph_structure

        self.graph.hidden = hidden
        self.graph.cell = cell

        self.graph.image_shape = image_shape

    def do_remesh_input(self, data_img, hidden, cell, mask=None):
        image_shape = self.graph.image_shape

        # Convert H and C to their image represenation using the old graph
        hidden_img = unflatten(hidden, self.graph.graph_structure['mapping'], image_shape)
        cell_img = unflatten(cell, self.graph.graph_structure['mapping'], image_shape)

        # Create graph using the current input
        data_img = add_positional_encoding(data_img)  # Add pos. embedding
        graph_structure = image_to_graph(data_img, thresh=self.thresh, mask=mask, transform_func=self.transform_func, condition=self.condition)

        # Use the graph structure to convert the hidden and cell states to their graph representations
        hidden_img, cell_img = torch.swapaxes(hidden_img, 0, -1), torch.swapaxes(cell_img, 0, -1)
        hidden = flatten(hidden_img, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
        cell = flatten(cell_img, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
        hidden, cell = torch.swapaxes(hidden, 0, -1), torch.swapaxes(cell, 0, -1)

        # Create a graph object for input into next rollout
        self.graph = create_graph_structure(graph_structure['graph_nodes'], graph_structure['distances'])
        self.graph.x = graph_structure['data']
        self.graph.graph_structure = graph_structure

        self.graph.hidden = hidden
        self.graph.cell = cell

        self.graph.image_shape = image_shape
        
