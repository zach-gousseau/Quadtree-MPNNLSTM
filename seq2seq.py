import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.nn.inits import glorot, zeros

from graph_functions import image_to_graph, flatten, create_graph_structure, unflatten
from utils import add_positional_encoding

import random
import numpy as np

from model import GConvLSTM

# class Encoder(torch.nn.Module):
#     def __init__(self, input_features, hidden_size, dropout, n_layers=1):
#         super().__init__()
        
#         self.hidden_size = hidden_size
        
#         self.n_layers = n_layers

#         self.rnn = GConvLSTM(input_features, hidden_size)
        
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, X, edge_index, edge_weight):

#         _, hidden, cell = self.rnn(X, edge_index, edge_weight)

#         hidden = torch.unsqueeze(hidden, 0)
#         cell = torch.unsqueeze(cell, 0)
        
#         return hidden, cell


# class Decoder(torch.nn.Module):
#     def __init__(self, input_features, hidden_size, dropout, n_layers=1):
#         super().__init__()
        
#         self.input_features = input_features
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
        
#         self.rnn = GConvLSTM(input_features, hidden_size)
        
#         self.fc_out = torch.nn.Linear(hidden_size+1, 1)  # Assuming output has 1 dimension
#         self.bn1 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
        
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, X, edge_index, edge_weight, skip, H, C):


#         output, hidden, cell = self.rnn(X, edge_index, edge_weight, H=H, C=C)
        
#         output = output.squeeze(0)  # Use top layer's output

#         output = F.relu(output)
#         output = self.bn1(output)

#         # skip connection
#         output = torch.cat([output, skip], dim=-1)
#         prediction = self.fc_out(output)
#         prediction = torch.sigmoid(prediction)

#         return prediction, hidden, cell

class Encoder(torch.nn.Module):
    def __init__(self, input_features, hidden_size, dropout, n_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.n_layers = n_layers

        self.rnns = [GConvLSTM(input_features, hidden_size)] + [GConvLSTM(hidden_size, hidden_size) for _ in range(n_layers-1)]
        
        self.dropout = nn.Dropout(dropout)

        # self.bn1 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
        self.bn1 = nn.LayerNorm(hidden_size)
        
    def forward(self, X, edge_index, edge_weight):
        
        _, hidden_layer, cell_layer = self.rnns[0](X, edge_index, edge_weight)

        hidden, cell = [hidden_layer], [cell_layer]
        for i in range(1, self.n_layers):
            _, hidden_layer, cell_layer = self.rnns[i](hidden[-1], edge_index, edge_weight)

            # hidden_layer = self.bn1(hidden_layer)
            hidden_layer = self.bn1(hidden_layer)
            cell_layer = self.bn1(cell_layer)

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
        
        # self.rnn1 = GConvLSTM(input_features, hidden_size)
        # self.rnn2 = GConvLSTM(hidden_size, hidden_size)

        self.rnns = [GConvLSTM(input_features, hidden_size)] + [GConvLSTM(hidden_size, hidden_size) for _ in range(n_layers-1)]
        
        self.fc_out = torch.nn.Linear(hidden_size, 1)  # Assuming output has 1 dimension
        # self.bn1 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
        self.bn1 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, edge_index, edge_weight, skip, H, C):


        # output1, hidden1, cell1 = self.rnn1(X, edge_index, edge_weight, H=hidden[[0]], C=cell[[0]])
        # output2, hidden2, cell2 = self.rnn2(hidden1, edge_index, edge_weight, H=hidden[[1]], C=cell[[1]])
        # hidden = [hidden1, hidden2]
        # cell = [cell1, cell2]

        output, hidden_layer, cell_layer = self.rnns[0](X, edge_index, edge_weight, H=H[[0]], C=C[[0]])

        hidden, cell = [hidden_layer], [cell_layer]
        for i in range(1, self.n_layers):
            output, hidden_layer, cell_layer = self.rnns[i](hidden[-1], edge_index, edge_weight, H=H[[i]], C=C[[i]])
            hidden.append(hidden_layer)
            cell.append(cell_layer)
        
        output = output.squeeze(0)  # Use top layer's output

        output = F.relu(output)
        output = self.bn1(output)

        # skip connection
        # output = torch.cat([output, skip], dim=-1)
        prediction = self.fc_out(output)
        prediction = torch.sigmoid(prediction)

        hidden = torch.cat(hidden)
        cell = torch.cat(cell)

        return prediction, hidden, cell
        

class Seq2Seq(torch.nn.Module):
    def __init__(self, hidden_size, dropout, input_timesteps=3, input_features=4, output_timesteps=5, n_layers=1):
        super().__init__()
        
        self.encoder = Encoder(input_features, hidden_size, dropout, n_layers=n_layers)
        self.decoder = Decoder(input_features, hidden_size, dropout, n_layers=n_layers)

        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.n_layers = n_layers
        
    def forward(self, X, y, input_graph_structure, input_skip_connection, image_shape, thresh, teacher_forcing_ratio=0.5):
        
        #tensor to store decoder outputs
        outputs = []
        outputs_graph_structures = []

        # Input graph structure
        curr_graph = create_graph_structure(input_graph_structure['graph_nodes'], input_graph_structure['distances'])
        curr_graph_structure = input_graph_structure

        # Skip connection
        skip = input_skip_connection
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        for t in range(self.input_timesteps):
            hidden, cell = self.encoder(X[t], curr_graph.edge_index, curr_graph.edge_weight)
        
        #first input to the decoder is the <sos> tokens
        curr_graph.x = torch.zeros_like(X[:1])
        
        for t in range(self.output_timesteps):

            # Perform decoding step
            output, hidden, cell = self.decoder(curr_graph.x, curr_graph.edge_index, curr_graph.edge_weight, skip, hidden, cell)

            # This is the prediction we are outputting. 
            # outputs.append(y_hat_img[0])
            outputs.append(output)
            outputs_graph_structures.append(curr_graph_structure)

            if thresh != -np.inf:

                # Output is a prediction on the original graph structure
                # First convert it back to its grid representation
                output_detached = output.detach().numpy()
                output_detached = np.expand_dims(output_detached, 0)

                y_hat_img = unflatten(output_detached, curr_graph_structure['graph_nodes'], curr_graph_structure['mappings'], image_shape=image_shape)
                
                # y_hat_img = np.expand_dims(y_hat_img, (0, -1))
                
                # Then we convert it back to a graph representation where the graph is determined by
                # its own values (rather than the one created by the input images / previous step)
                y_hat_img = np.expand_dims(y_hat_img, (0))
                y_hat_img = add_positional_encoding(y_hat_img)  # Add pos. embedding
                graph_structure = image_to_graph(y_hat_img[0], thresh=thresh)  # Generate new graph using the new X

                device = hidden.device

                hidden, cell = hidden.detach().numpy(), cell.detach().numpy()

                hidden_img = unflatten(hidden, curr_graph_structure['graph_nodes'], curr_graph_structure['mappings'], image_shape=image_shape)
                cell_img = unflatten(cell, curr_graph_structure['graph_nodes'], curr_graph_structure['mappings'], image_shape=image_shape)


                # Now we decide whether to use the prediction or ground truth for the input to the next rollout step
                teacher_force = random.random() < teacher_forcing_ratio
                if teacher_force:
                    input_img = y[t]
                    input_img = np.expand_dims(input_img, (0, 1))
                    input_img = add_positional_encoding(input_img)
                    curr_graph_structure = image_to_graph(input_img[0], num_features=3, thresh=thresh)

                    skip = torch.from_numpy(curr_graph_structure['data'][0, :, :1]).float()
                else:
                    curr_graph_structure = graph_structure  # Use the prediction graph structure (which includes the predicted data)
                    
                    skip = torch.from_numpy(curr_graph_structure['data'][0, :, :1]).float()

                hidden_img = np.swapaxes(hidden_img, 0, -1)

                hidden, _ = flatten(hidden_img, curr_graph_structure['labels'])

                cell_img = np.swapaxes(cell_img, 0, -1)
                cell, _ = flatten(cell_img, curr_graph_structure['labels'])

                hidden, cell = np.swapaxes(hidden, 0, -1), np.swapaxes(cell, 0, -1)

                hidden = torch.Tensor(hidden)
                hidden.to(device)
                cell = torch.Tensor(cell)
                cell.to(device)

                # Create a PyG graph object for input into next rollout
                curr_graph = create_graph_structure(curr_graph_structure['graph_nodes'], curr_graph_structure['distances'])
                curr_graph.x = torch.Tensor(curr_graph_structure['data'])

            else:
                # curr_graph_structure does not change
                # but the input (curr_graph.x) does change
                teacher_force = random.random() < teacher_forcing_ratio
                if teacher_force:
                    input_img = y[t]
                    input_img = np.expand_dims(input_img, (0, 1))
                    input_img = add_positional_encoding(input_img).squeeze(0)
                    input_x, _ = flatten(input_img, curr_graph_structure['labels'])

                    curr_graph.x = torch.cat((curr_graph.x[..., 1:], torch.from_numpy(input_x[..., [0]])), -1).float()

                    skip = torch.from_numpy(input_x[0, :, :1]).float()
                else:
                    # TODO the input and skip are the same thing... that's silly
                    output_detached = torch.from_numpy(np.expand_dims(output.detach().numpy(), 0))
                    curr_graph.x = torch.cat((curr_graph.x[..., 1:], output_detached), -1).float()
                    skip = output
            
        return outputs, outputs_graph_structures