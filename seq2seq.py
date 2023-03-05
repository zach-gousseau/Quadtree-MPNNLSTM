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
        
        _, hidden_layer, cell_layer = self.rnns[0](X, edge_index, edge_weight, H=H, C=C)
        hidden_layer, cell_layer = hidden_layer.squeeze(0), cell_layer.squeeze(0)
        hidden_layer = self.norm_h(hidden_layer)
        cell_layer = self.norm_c(cell_layer)

        # _ = _.detach().numpy()
        # _ = np.expand_dims(_, 0)

        # d = image_to_graph(np.zeros((1, 32, 32, 8), dtype=float), thresh=-np.inf, max_grid_size=8, mask=None, transform_func=self.transform_func)
        # import matplotlib.pyplot as plt
        # img = unflatten(_, d['graph_nodes'], d['mappings'], image_shape=(32, 32), nan_value=np.nan)
        # fig, axs = plt.subplots(1, 8, figsize=(20, 4))
        # for i in range(8):
        #     axs[i].imshow(img[0, ..., i])


        hidden, cell = [hidden_layer], [cell_layer]
        for i in range(1, self.n_layers):
            _, hidden_layer, cell_layer = self.rnns[i](hidden[-1], edge_index, edge_weight, H=hidden_layer, C=cell_layer)

            # hidden_layer = self.bn1(hidden_layer)
            hidden_layer = self.norm_h(hidden_layer)
            cell_layer = self.norm_c(cell_layer)

            # hidden_layer, cell_layer = hidden_layer.squeeze(0), cell_layer.squeeze(0)

            hidden.append(hidden_layer)
            cell.append(cell_layer)

        hidden = torch.stack(hidden)
        cell = torch.stack(cell)
        
        return hidden, cell

class Decoder(torch.nn.Module):
    def __init__(self, input_features, hidden_size, dropout, n_layers=1):
        super().__init__()
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.rnns = nn.ModuleList([GConvLSTM(input_features, hidden_size)] + [GConvLSTM(hidden_size, hidden_size) for _ in range(n_layers-1)])
        
        self.fc_out1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_out2 = torch.nn.Linear(hidden_size, 1)  # Assuming output has 1 dimension
        # self.bn1 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
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

        output = F.relu(output)
        output = self.norm_o(output)

        # skip connection
        # output = torch.cat([output, skip], dim=-1)
        prediction = self.fc_out1(output)
        output = F.relu(output)
        prediction = self.fc_out2(output)
        prediction = torch.sigmoid(prediction)

        return prediction, hidden, cell
        

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
                 device=None):
        super().__init__()
        
        self.encoder = Encoder(input_features, hidden_size, dropout, n_layers=n_layers)
        self.decoder = Decoder(1+3, hidden_size, dropout, n_layers=n_layers)  # 1 output variable + 3 (positional encoding and node size)

        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.n_layers = n_layers

        self.thresh = thresh
        self.transform_func = transform_func

        self.device = device
        
    def forward(self, graph, image_shape, teacher_forcing_ratio=0.5, mask=None):
        
        #tensor to store decoder outputs
        outputs = []
        outputs_graph_structures = []

        # Input graph structure
        # curr_graph = create_graph_structure(input_graph_structure['graph_nodes'], input_graph_structure['distances']).to(device)
        curr_graph = graph
        curr_graph_structure = graph.input_graph_structure
        image_shape = graph.image_shape
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = [None], [None]
        for t in range(self.input_timesteps):
            hidden, cell = self.encoder(graph.x[t], curr_graph.edge_index, curr_graph.edge_weight, H=hidden[-1], C=hidden[-1])
        
        #first input to the decoder is the <sos> tokens
        # curr_graph.x = torch.zeros_like(X[:1])
        curr_graph.x = graph.x[[-1]][..., [0, -1, -2, -3]]
        curr_graph.skip = graph.x[[-1]][..., [0, -1, -2, -3]]  # TODO: Check that this is the right skip connection !!

        # hidden = hidden.squeeze(0)
        # cell = cell.squeeze(0)
        
        for t in range(self.output_timesteps):
            
            curr_graph.to(self.device[0])

            # Perform decoding step
            output, hidden, cell = self.decoder(curr_graph.x, curr_graph.edge_index, curr_graph.edge_weight, curr_graph.skip, hidden, cell)

            # This is the prediction we are outputting. 
            outputs.append(output)
            outputs_graph_structures.append(curr_graph_structure)

            if self.thresh != -np.inf:


                # Output is a prediction on the original graph structure
                # First convert it back to its grid representation
                output_detached = output#.cpu().detach().numpy()
                output_detached = output_detached.unsqueeze(0)  #np.expand_dims(output_detached, 0)
                # print(t)
                y_hat_img = unflatten(output_detached, curr_graph_structure['mapping'], image_shape)
                
                # Then we convert it back to a graph representation where the graph is determined by
                # its own values (rather than the one created by the input images / previous step)
                y_hat_img = add_positional_encoding(y_hat_img)  # Add pos. embedding
                graph_structure = image_to_graph(y_hat_img, thresh=self.thresh, mask=mask, transform_func=self.transform_func)  # Generate new graph using the new X

                # hidden, cell = hidden.cpu().detach().numpy(), cell.cpu().detach().numpy()

                hidden_img = unflatten(hidden, curr_graph_structure['mapping'], image_shape=image_shape)
                cell_img = unflatten(cell, curr_graph_structure['mapping'], image_shape=image_shape)


                # Now we decide whether to use the prediction or ground truth for the input to the next rollout step
                teacher_force = random.random() < teacher_forcing_ratio
                if teacher_force:
                    input_img = graph.y[[t]]

                    # try:
                    #     input_img = input_img.cpu()
                    # except AttributeError:
                    #     pass
                    
                    input_img = add_positional_encoding(input_img)
                    curr_graph_structure = image_to_graph(input_img, thresh=self.thresh, mask=mask, transform_func=self.transform_func)

                    skip = curr_graph_structure['data'][0, :, :1]

                else:
                    curr_graph_structure = graph_structure  # Use the prediction graph structure (which includes the predicted data)
                    
                    skip = curr_graph_structure['data'][0, :, :1]

                hidden_img = torch.swapaxes(hidden_img, 0, -1)

                hidden = flatten(hidden_img, curr_graph_structure['mapping'], curr_graph_structure['n_pixels_per_node'])

                cell_img = torch.swapaxes(cell_img, 0, -1)
                cell = flatten(cell_img, curr_graph_structure['mapping'], curr_graph_structure['n_pixels_per_node'])

                hidden, cell = torch.swapaxes(hidden, 0, -1), torch.swapaxes(cell, 0, -1)

                # hidden = torch.Tensor(hidden).to(self.device[0])
                # cell = torch.Tensor(cell).to(self.device[0])

                del curr_graph

                # Create a PyG graph object for input into next rollout
                curr_graph = create_graph_structure(curr_graph_structure['graph_nodes'], curr_graph_structure['distances'])
                curr_graph.x = torch.Tensor(curr_graph_structure['data'])
                # curr_graph.skip = torch.from_numpy(skip).type(torch.float32)
                curr_graph.skip = skip

            else:
                # curr_graph_structure does not change
                # but the input (curr_graph.x) does change
                teacher_force = random.random() < teacher_forcing_ratio
                teacher_force = False
                if teacher_force:
                    input_img = graph.y[[t]]
                    input_img = add_positional_encoding(input_img)
                    input_x = flatten(input_img, curr_graph_structure['mapping'], curr_graph_structure['n_pixels_per_node'])

                    curr_graph.x = torch.cat((curr_graph.x[..., 1:], torch.from_numpy(input_x[..., [0]])), -1)#.float()

                    curr_graph.skip = torch.from_numpy(input_x[0, :, :1])#.float()
                    
                else:
                    # TODO the input and skip are the same thing... that's silly
                    output_detached = torch.from_numpy(np.expand_dims(output.detach().numpy(), 0))
                    # output_detached = output.detach()
                    curr_graph.x = torch.cat((curr_graph.x[..., 1:], output_detached), -1)#.float()
                    curr_graph.skip = output
            
        return outputs, outputs_graph_structures