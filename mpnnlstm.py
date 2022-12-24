import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os 

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, TransformerConv

from torch.optim.lr_scheduler import StepLR

from graph_functions import image_to_graph, flatten, create_graph_structure, unflatten

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MPNNLSTM(nn.Module):

    def __init__(self, hidden_size, dropout, input_timesteps=3, input_features=4):
        super(MPNNLSTM, self).__init__()
        self.dropout = dropout
        self.input_timesteps = input_timesteps

        self.convolution1 = GCNConv(input_features, hidden_size)
        self.convolution2 = GCNConv(hidden_size, hidden_size)
        self.convolution3 = GCNConv(hidden_size, hidden_size)
        self.convolution4 = GCNConv(hidden_size, hidden_size)
        
        self.bn1 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
        self.bn4 = nn.BatchNorm1d(hidden_size, track_running_stats=False)

        self.recurrents = nn.LSTM(hidden_size, hidden_size, 2)

        self.lin1 = torch.nn.Linear(hidden_size+input_timesteps, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, 1)


    def forward(self, X, edge_index, edge_weight=None):

        # Convolutions, BN
        C = []
        for i in range(X.shape[0]):
            H = F.relu(self.convolution1(X[i], edge_index, edge_weight=edge_weight))
            H = self.bn1(H)

            H = F.relu(self.convolution2(H, edge_index, edge_weight=edge_weight))
            H = self.bn2(H)

            H = F.relu(self.convolution3(H, edge_index, edge_weight=edge_weight))
            H = self.bn3(H)

            H = F.relu(self.convolution4(H, edge_index, edge_weight=edge_weight))
            H = self.bn4(H)
            C.append(H)

        C = torch.stack(C)

        _, (H, _) = self.recurrents(C)
        
        H = F.relu(H[-1])  # Keep only last hidden state of last layer

        S = X[:, :, 0].T  # Skip connection thang

        H = torch.cat([H, S], dim=-1)

        # FC for output
        H = self.lin1(H)
        H = self.lin2(H)
        
        # Dropout and sigmoid out
        H = F.dropout(H, p=self.dropout, training=self.training)
        # H = torch.sigmoid(H)
        # H = F.relu(H)
        return H


class NextFramePredictor():
    def __init__(self, experiment_name='reduced', decompose=True, input_features=1, **model_kwargs):
        self.experiment_name = experiment_name

        # Set the threshold to negative infinity if we want to keep the full basis (i.e. split all the way down)
        self.thresh = None if decompose else -np.inf
        self.decompose = decompose

        # Add 3 to the number of input features since we add positional encoding (x, y) and node size (s)
        self.model = MPNNLSTM(input_features=input_features+3, **model_kwargs).float()

        self.input_features = input_features  # Number of user features 

    def add_positional_encoding(self, x):
        image_shape = x[0].shape[1:]

        # Position encoding
        ii = np.tile(np.array(range(image_shape[0])), (image_shape[1], 1))
        jj = np.tile(np.array(range(image_shape[0])), (image_shape[1], 1)).T

        pos_encoding = np.moveaxis(np.array([[ii, jj]]*self.model.input_timesteps), 1, -1)

        x = np.concatenate((x, np.array([pos_encoding]*len(x))), axis=-1)
        return x

    def test_threshold(self, x, thresh, frame_index=0):
        image_shape = x[0].shape[1:]

        x_with_pos_encoding = self.add_positional_encoding(x)
        frames = x_with_pos_encoding[frame_index]

        graph = image_to_graph(frames, num_features=3, thresh=thresh)
        img_reconstructed = unflatten(graph['data'][..., 0], graph['graph_nodes'], graph['mappings'], image_shape=image_shape)

        fig, axs = plt.subplots(1, self.model.input_timesteps, figsize=(3*self.model.input_timesteps, 3))

        for i in range(self.model.input_timesteps):
            axs[i].imshow(img_reconstructed[i, ..., 0])
        plt.suptitle(f'Threshold: {thresh}')
        return fig, axs

    def set_thresh(self, thresh):
        self.thresh = thresh

    def train(
        self,
        x,
        y,
        x_test,
        y_test,
        n_epochs=200,
        lr=0.01,
        lr_decay=0.95,
        ):

        if self.thresh is None:
            raise ValueError('Please set the threshold using set_thresh(thresh)!')

        x = self.add_positional_encoding(x)
        x_test = self.add_positional_encoding(x_test)

        # Add 2 to the number of features since we add positional encoding (x, y)
        x_graph = [image_to_graph(img, num_features=self.input_features+2, thresh=self.thresh) for img in x]
        x_test_graph = [image_to_graph(img, num_features=self.input_features+2, thresh=self.thresh) for img in x_test]
            
        self.model.to(device)
        self.model.train()

        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=10, gamma=lr_decay)

        test_loss = []
        train_loss = []

        st = time.time()
        for epoch in range(n_epochs): 
            running_loss = 0
            step = 0
            for i in range(len(x_graph)):

                graph_nodes, distances = x_graph[i]['graph_nodes'], x_graph[i]['distances']

                graph = create_graph_structure(graph_nodes, distances)

                x_batch = x_graph[i]['data']
                
                # Turn target frame into graph using the graph structure from the input frames
                y_batch, _, _ = flatten(y[i], x_graph[i]['labels'], num_features=1)

                graph.x = torch.from_numpy(x_batch).float()
                graph.y = torch.from_numpy(y_batch).float()

                graph.to(device)

                optimizer.zero_grad()
                
                y_hat = self.model(graph.x, graph.edge_index, graph.edge_attr)
                loss = loss_func(y_hat, torch.tensor(graph.y[0]))
                loss = loss_func(y_hat, graph.y[0])

                loss.backward()
                optimizer.step()

                step += 1
                running_loss += loss

            running_loss_test = 0
            step_test = 0
            for i in range(len(x_test_graph)):

                graph_nodes, distances = x_test_graph[i]['graph_nodes'], x_test_graph[i]['distances']

                graph = create_graph_structure(graph_nodes, distances)

                x_batch = x_test_graph[i]['data']

                # Turn target frame into graph using the graph structure from the input frames
                y_batch, _, _ = flatten(y_test[i], x_test_graph[i]['labels'], num_features=1)

                graph.x = torch.from_numpy(x_batch).float()
                graph.y = torch.from_numpy(y_batch).float()

                graph.to(device)
                y_hat = self.model(graph.x, graph.edge_index, graph.edge_attr)
                loss = loss_func(y_hat, torch.tensor(graph.y[0, ..., :1]))

                step_test += 1
                running_loss_test += loss

            running_loss = running_loss / (step + 1)
            running_loss_test = running_loss_test / (step_test + 1)

            scheduler.step()

            train_loss.append(running_loss.item())
            test_loss.append(running_loss_test.item())
            
            print(f"Epoch {epoch} train MSE: {running_loss.item():.4f}, "+ \
                f"test MSE: {running_loss_test.item():.4f}, lr: {scheduler.get_last_lr()[0]:.4f}, time_per_epoch: {(time.time() - st) / (epoch+1):.1f}")
        
        print(f'Finished in {(time.time() - st)/60} minutes')

        self.loss = pd.DataFrame({
            'train_loss': train_loss,
            'test_loss': test_loss,

        })

    def save(self, directory):
        torch.save(self.model.state_dict(), os.path.join(directory, f'{self.experiment_name}.pth'))

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, x):
        self.model.eval()
        
        image_shape = x[0].shape[1:-1]

        # Position encoding
        ii = np.tile(np.array(range(image_shape[0])), (image_shape[1], 1))
        jj = np.tile(np.array(range(image_shape[0])), (image_shape[1], 1)).T

        pos_encoding = np.moveaxis(np.array([[ii, jj]]*self.model.input_timesteps), 1, -1)

        x = np.concatenate((x, np.array([pos_encoding]*len(x))), axis=-1)

        x_graph = [image_to_graph(img, num_features=self.input_features+2, thresh=self.thresh) for img in x]

        y_pred = []
        for i in range(len(x_graph)):

            graph_nodes, distances, mappings = x_graph[i]['graph_nodes'], x_graph[i]['distances'], x_graph[i]['mappings']

            graph = create_graph_structure(graph_nodes, distances)

            x_batch = x_graph[i]['data']

            graph.x = torch.from_numpy(x_batch).float()
            
            y_hat = self.model(graph.x, graph.edge_index, graph.edge_attr)
            y_hat = unflatten(np.expand_dims(y_hat[..., 0].cpu().detach(), 0), graph_nodes, mappings, image_shape=image_shape)

            y_pred.append(y_hat)

        return np.array(y_pred)


    def score(self, x, y):
        self.model.eval()
        metric = torch.nn.MSELoss()

        y_hat = self.predict(x)

        score = metric(torch.Tensor(y_hat), torch.Tensor(y[..., 0]))
        return score
