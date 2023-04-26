import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv, ChebConv, TransformerConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros

from graph_functions import image_to_graph, flatten, create_graph_structure, unflatten
from utils import add_positional_encoding

import gc 
import random
import numpy as np
import psutil
import os

from model import GConvLSTM, DummyLSTM, CONVOLUTION_KWARGS, CONVOLUTIONS


class Encoder(torch.nn.Module):
    def __init__(self, input_features, hidden_size, dropout, n_layers=1, convolution_type='GCNConv', n_conv_layers=3, dummy=False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dummy = dummy

        if not self.dummy:
            self.rnns = nn.ModuleList(
                [GConvLSTM(input_features, hidden_size, convolution_type=convolution_type, n_conv_layers=n_conv_layers)] + \
                [GConvLSTM(hidden_size, hidden_size, convolution_type=convolution_type, n_conv_layers=n_conv_layers) for _ in range(n_layers-1)]
            )
        
        self.dropout = nn.Dropout(dropout)

        self.norm_h = nn.LayerNorm(hidden_size)
        self.norm_c = nn.LayerNorm(hidden_size)
        
    def forward(self, X, edge_index, edge_weight, H=None, C=None):

        if self.dummy:
            return H, C

        X = X.squeeze(0)

        _, hidden_layer, cell_layer = self.rnns[0](X, edge_index, edge_weight, H=H, C=C)

        hidden_layer, cell_layer = hidden_layer.squeeze(0), cell_layer.squeeze(0)
        hidden_layer = self.norm_h(hidden_layer)
        cell_layer = self.norm_c(cell_layer)

        hidden, cell = [hidden_layer], [cell_layer]
        for i in range(1, self.n_layers):
            _, hidden_layer, cell_layer = self.rnns[i](hidden[-1], edge_index, edge_weight, H=None, C=None)

            hidden_layer = self.norm_h(hidden_layer)
            cell_layer = self.norm_c(cell_layer)

            hidden.append(hidden_layer)
            cell.append(cell_layer)

        hidden = torch.stack(hidden)
        cell = torch.stack(cell)
        return hidden, cell

class Decoder(torch.nn.Module):
    def __init__(self, input_features, hidden_size, dropout, n_layers=1, skip_dim=2, convolution_type='GCNConv', n_conv_layers=3, binary=False, dummy=False):
        super().__init__()
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.binary = binary
        self.dummy = dummy

        n_conv_layers = 1

        if not self.dummy:
            self.rnns = nn.ModuleList(
                [GConvLSTM(input_features, hidden_size, convolution_type=convolution_type, n_conv_layers=n_conv_layers)] + \
                [GConvLSTM(hidden_size, hidden_size, convolution_type=convolution_type, n_conv_layers=n_conv_layers) for _ in range(n_layers-1)]
                )

        conv_func = CONVOLUTIONS[convolution_type]
        conv_func_kwargs = CONVOLUTION_KWARGS[convolution_type]

        in_channels = hidden_size + skip_dim if not dummy else 3 + skip_dim

        self.fc_out1 = conv_func(in_channels=in_channels, out_channels=hidden_size, **conv_func_kwargs)
        # self.fc_out2 = conv_func(in_channels=hidden_size, out_channels=hidden_size, **conv_func_kwargs)
        # self.fc_out3 = conv_func(in_channels=hidden_size, out_channels=hidden_size, **conv_func_kwargs)
        self.fc_out2 = conv_func(in_channels=hidden_size, out_channels=1, **conv_func_kwargs)
        # self.fc_out1 = Linear(in_channels=in_channels, out_channels=hidden_size)
        # self.fc_out2 = Linear(in_channels=hidden_size, out_channels=hidden_size)
        # self.fc_out3 = Linear(in_channels=hidden_size, out_channels=hidden_size)
        # self.fc_out4 = Linear(in_channels=hidden_size, out_channels=1)

        self.norm_o = nn.LayerNorm(hidden_size)
        self.norm_h = nn.LayerNorm(hidden_size)
        self.norm_c = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, edge_index, edge_weight, skip, H, C):

        if self.dummy:
            if skip is not None:
                X = torch.cat([X, skip], dim=-1)
            X = self.gnn_out(X, edge_index, edge_weight)
            return X, H, C

        output, hidden_layer, cell_layer = self.rnns[0](X, edge_index, edge_weight, H=H[0], C=C[0])

        hidden_layer = self.norm_h(hidden_layer)
        cell_layer = self.norm_c(cell_layer)
        hidden, cell = [hidden_layer], [cell_layer]

        for i in range(1, self.n_layers):
            output, hidden_layer, cell_layer = self.rnns[i](hidden[-1], edge_index, edge_weight, H=H[i], C=C[i])

            hidden_layer = self.norm_h(hidden_layer)
            cell_layer = self.norm_c(cell_layer)

            hidden.append(hidden_layer)
            cell.append(cell_layer)

        hidden = torch.stack(hidden)
        cell = torch.stack(cell)
        
        output = output.squeeze(0)  # Use top layer's output
        output = self.norm_o(output)
        output = F.relu(output)

        if skip is not None:
            output = torch.cat([output, skip], dim=-1)

        output = self.gnn_out(output, edge_index, edge_weight)

        output += X[:, [0]]

        return output, hidden, cell

    def gnn_out(self, x, edge_index, edge_weight):
        x = self.fc_out1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.fc_out2(x, edge_index, edge_weight)
        # x = F.relu(x)
        # x = self.fc_out3(x, edge_index, edge_weight)
        # x = F.relu(x)
        # x = self.fc_out4(x, edge_index, edge_weight)
        # x = torch.sigmoid(x)#F.relu(x)
        # x = self.fc_out3(x, edge_index, edge_weight)
        # x = torch.sigmoid(x)#F.relu(x)
        # x = self.fc_out4(x, edge_index, edge_weight)

        if self.binary:
            x = torch.sigmoid(x)
        return x
        

class Seq2Seq(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 dropout,
                 thresh,
                 input_timesteps=3,
                 input_features=3, #4 node_size
                 output_timesteps=5,
                 n_layers=4,
                 n_conv_layers=2,
                 transform_func=None,
                 condition='max_larger_than',
                 remesh_input=False,
                 convolution_type='ChebConv',
                 binary=False,
                 dummy=False,
                 device=None,
                 debug=False):
        super().__init__()
        
        self.encoder = Encoder(
            input_features,
            hidden_size,
            dropout,
            n_layers=n_layers,
            convolution_type=convolution_type,
            n_conv_layers=n_conv_layers,
            dummy=dummy,
            )
        self.decoder = Decoder(
            1+2,  # 1 output variable + 3 (positional encoding and node_size)
            hidden_size,
            dropout,
            n_layers=n_layers,
            convolution_type=convolution_type,
            n_conv_layers=n_conv_layers,
            binary=binary,
            dummy=dummy,
            )

        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.n_layers = n_layers
        self.condition = condition
        self.remesh_input = remesh_input
        self.debug = debug

        self.convolution_type = convolution_type

        if convolution_type in ['MHTransformerConv', 'TransformerConv', 'GATConv']:
            self.use_edge_attrs = True
        else:
            self.use_edge_attrs = False

        self.thresh = thresh
        self.transform_func = transform_func

        self.graph = None

        self.device = device

    def process_inputs(self, x, mask=None):

        num_samples, w, h, c = x.shape
        image_shape = (w, h)

        self.mask = mask
        
        if self.remesh_input:
            x0 = add_positional_encoding(x[[0]])
            graph_structure = image_to_graph(x0, thresh=self.thresh, mask=mask, transform_func=self.transform_func, condition=self.condition, use_edge_attrs=self.use_edge_attrs)
        else:
            x = add_positional_encoding(x)
            graph_structure = image_to_graph(x, thresh=self.thresh, mask=mask, transform_func=self.transform_func, condition=self.condition, use_edge_attrs=self.use_edge_attrs)

        self.graph = create_graph_structure(graph_structure['edge_index'], graph_structure['edge_attrs'])

        self.graph.pyg.x = graph_structure['data']

        self.graph.mapping = graph_structure['mapping']
        self.graph.n_pixels_per_node = graph_structure['n_pixels_per_node']
        self.graph.image_shape = image_shape
        self.graph.pyg.to(self.device)
        
        # Encoder ------------------------------------------------------------------------------------------------------------
        self.graph.hidden, self.graph.cell = None, None
        for t in range(self.input_timesteps):

            # Perform encoding step
            hidden, cell = self.encoder(
                X=self.graph.pyg.x if self.remesh_input else self.graph.pyg.x[[t]], 
                edge_index=self.graph.pyg.edge_index, 
                edge_weight=self.graph.pyg.edge_attr, 
                H=self.graph.hidden[-1] if self.graph.hidden is not None else None, 
                C=self.graph.cell[-1] if self.graph.hidden is not None else None
                )

            if t < self.input_timesteps:
                if self.thresh != -np.inf:
                    if self.remesh_input:
                        self.do_remesh_input(x[[t+1]], hidden, cell, mask)
                    else:
                        self.graph.hidden = hidden
                        self.graph.cell = cell
                else:
                    self.graph.hidden = hidden
                    self.graph.cell = cell

        # Persistance
        self.graph.persistence = x[[-1]][:, :, :, [0]]
        
        # First input to the decoder is the last input to the encoder 
        # self.graph.pyg.x = self.graph.pyg.x[-1, :, [0, -3, -2, -1]].unsqueeze(0)
        self.graph.pyg.x = self.graph.pyg.x[-1, :, [0, -2, -1]]  # node_size


    def unroll_output(self, unroll_steps, y, skip=None, teacher_forcing_ratio=0.5, mask=None, remesh_every=1):

        # Lists to store decoder outputs
        outputs = []
        output_mappings = []

        for t in unroll_steps:

            if self.debug:
                if self.device.type == 'cuda':
                    print(
                        f'Decoder step {t} \n' + \
                        f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0)/1024/1024/1024}GB\n" + \
                        f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(0)/1024/1024/1024}GB\n" + \
                        f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(0)/1024/1024/1024}GB",
                        end='\033[A\033[A\033[A'
                    )
                else:
                    pid = os.getpid()
                    python_process = psutil.Process(pid)
                    memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
                    print('CPU memory usage:', memoryUse, 'GB', end='\r')
            
            if skip is not None:
                skip_t = torch.cat([skip[t].unsqueeze(0), self.graph.persistence], dim=-1)
            else:
                skip_t = self.graph.persistence

            skip_t = flatten(skip_t, self.graph.mapping, self.graph.n_pixels_per_node, self.mask).squeeze(0)

            self.graph.skip = skip_t
            self.graph.pyg.to(self.device)

            # Perform decoding step
            output, hidden, cell = self.decoder(
                X=self.graph.pyg.x,
                edge_index=self.graph.pyg.edge_index,
                edge_weight=self.graph.pyg.edge_attr, 
                skip=self.graph.skip if hasattr(self.graph, 'skip') else None,
                H=self.graph.hidden, 
                C=self.graph.cell
                )

            # This is the prediction we are outputting. 
            outputs.append(output)
            output_mappings.append(self.graph.mapping)

            output = output#.to(self.device)

            # Decide whether to use the prediction or ground truth for the input to the next rollout step
            teacher_force = random.random() < teacher_forcing_ratio
            teacher_input = y[[t]] if teacher_force else None

            if (self.thresh != -np.inf) and ((t+1) % remesh_every == 0):
                self.do_remesh(output, hidden, cell, mask, teacher_force=teacher_force, teacher_input=teacher_input)
            else:
                self.update_without_remesh(output, hidden, cell, teacher_force=teacher_force, teacher_input=teacher_input)

        return outputs, output_mappings
        

        
    def forward(self, x, y=None, skip=None, teacher_forcing_ratio=0.5, mask=None, remesh_every=1):

        # Encoder
        self.process_inputs(x, mask=mask)

        # Decoder
        outputs, output_mappings = self.unroll_output(
            range(self.output_timesteps),
            y,
            skip=skip,
            teacher_forcing_ratio=teacher_forcing_ratio,
            mask=mask,
            remesh_every=remesh_every
            )
            
        return outputs, output_mappings

    def update_without_remesh(self, data, hidden, cell, teacher_force=False, teacher_input=None):
        if teacher_force:
            teacher_input = add_positional_encoding(teacher_input)  # Add positional encoding
            self.graph.pyg.x = flatten(teacher_input, self.graph.mapping, self.graph.n_pixels_per_node, self.mask).squeeze(0)
            # self.graph.pyg.x = torch.cat([self.graph.pyg.x, self.graph.n_pixels_per_node.unsqueeze(0).unsqueeze(-1)], dim=-1)  # Add node sizes
        else:
            # Add positional encoding
            pos_encoding = self.graph.pyg.x[..., 1:]
            self.graph.pyg.x = torch.cat([data, pos_encoding], dim=-1)

        self.graph.hidden = hidden
        self.graph.cell = cell


    def do_remesh(self, data, hidden, cell, mask=None, teacher_force=False, teacher_input=None):
        image_shape = self.graph.image_shape

        # Output is a prediction on the original graph structure
        # First convert it back to its grid representation
        # We also convert the hidden and cell state in the same way 
        data_img = unflatten(data, self.graph.mapping, image_shape)
        hidden_img = unflatten(hidden, self.graph.mapping, image_shape)
        cell_img = unflatten(cell, self.graph.mapping, image_shape)

        del self.graph.mapping

        # Then we convert it back to a graph representation where the graph is determined by
        # its own values (rather than the one created by the input images / previous step)
        if teacher_force:
            teacher_input = add_positional_encoding(teacher_input)
            graph_structure = image_to_graph(teacher_input, thresh=self.thresh, mask=mask, transform_func=self.transform_func, condition=self.condition, use_edge_attrs=self.use_edge_attrs)
        else:
            data_img = add_positional_encoding(data_img)  # Add pos. embedding
            graph_structure = image_to_graph(data_img, thresh=self.thresh, mask=mask, transform_func=self.transform_func, condition=self.condition, use_edge_attrs=self.use_edge_attrs)

        # skip = graph_structure['data'][:, :, [0]]

        # Use the graph structure to convert the hidden and cell states to their graph representations
        hidden_img, cell_img = torch.swapaxes(hidden_img, 0, -1), torch.swapaxes(cell_img, 0, -1)
        hidden = flatten(hidden_img, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
        cell = flatten(cell_img, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
        hidden, cell = torch.swapaxes(hidden, 0, -1), torch.swapaxes(cell, 0, -1)

        # Create a graph object for input into next rollout
        self.graph = create_graph_structure(graph_structure['edge_index'], graph_structure['edge_attrs'])
        self.graph.pyg.x = graph_structure['data']
        # self.graph.skip = skip
        self.graph.mapping = graph_structure['mapping']
        self.graph.n_pixels_per_node = graph_structure['n_pixels_per_node']

        self.graph.hidden = hidden
        self.graph.cell = cell

        self.graph.image_shape = image_shape

    def do_remesh_input(self, data_img, hidden, cell, mask=None):
        image_shape = self.graph.image_shape

        # Convert H and C to their image represenation using the old graph
        hidden_img = unflatten(hidden, self.graph.mapping, image_shape)
        cell_img = unflatten(cell, self.graph.mapping, image_shape)

        # Create graph using the current input
        data_img = add_positional_encoding(data_img)  # Add pos. embedding
        graph_structure = image_to_graph(data_img, thresh=self.thresh, mask=mask, transform_func=self.transform_func, condition=self.condition, use_edge_attrs=self.use_edge_attrs)

        # Use the graph structure to convert the hidden and cell states to their graph representations
        hidden_img, cell_img = torch.swapaxes(hidden_img, 0, -1), torch.swapaxes(cell_img, 0, -1)
        hidden = flatten(hidden_img, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
        cell = flatten(cell_img, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
        hidden, cell = torch.swapaxes(hidden, 0, -1), torch.swapaxes(cell, 0, -1)

        # Create a graph object for input into next rollout
        self.graph = create_graph_structure(graph_structure['edge_index'], graph_structure['edge_attrs'])
        self.graph.pyg.x = graph_structure['data']
        self.graph.mapping = graph_structure['mapping']
        self.graph.n_pixels_per_node = graph_structure['n_pixels_per_node']

        self.graph.hidden = hidden
        self.graph.cell = cell

        self.graph.image_shape = image_shape
    