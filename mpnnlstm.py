import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, TransformerConv

from torch.optim.lr_scheduler import StepLR

from graph_functions import image_to_graph, flatten, create_graph_structure, unflatten
from model import MPNNLSTMI, MPNNLSTM
from utils import get_n_params, add_positional_encoding


class NextFramePredictor():
    def __init__(self, 
                 experiment_name='reduced', 
                 decompose=True, 
                 input_features=1,
                 multi_step_loss=1, 
                 integrated_space_time=True,
                 device=None,
                 **model_kwargs):

        self.experiment_name = experiment_name
        self.multi_step_loss = multi_step_loss

        # Set the threshold to negative infinity if we want to keep the full basis (i.e. split all the way down)
        self.thresh = None if decompose else -np.inf
        self.decompose = decompose

        # Add 3 to the number of input features since we add positional encoding (x, y) and node size (s)
        if integrated_space_time:
            self.model = MPNNLSTMI(input_features=input_features+3, **model_kwargs).float()
        else:
            self.model = MPNNLSTM(input_features=input_features+3, **model_kwargs).float()

        self.input_features = input_features  # Number of user features 

        self.device = device

    def test_threshold(self, x, thresh, frame_index=0):
        image_shape = x[0].shape[1:]

        x_with_pos_encoding = add_positional_encoding(x, self.model.input_timesteps)
        frames = x_with_pos_encoding[frame_index]

        graph = image_to_graph(frames, num_features=3, thresh=thresh)
        img_reconstructed = unflatten(graph['data'][..., :1], graph['graph_nodes'], graph['mappings'], image_shape=image_shape)

        num_nodes = len(np.unique(graph['labels']))
        fig, axs = plt.subplots(1, self.model.input_timesteps, figsize=(4*self.model.input_timesteps, 3))

        for i in range(self.model.input_timesteps):
            axs[i].imshow(img_reconstructed[i, ..., 0])

        plt.suptitle(f'Threshold: {thresh} | Num. nodes: {num_nodes}')
        return fig, axs

    def set_thresh(self, thresh):
        if self.decompose:
            self.thresh = thresh

    def get_n_params(self):
        return get_n_params(self.model)

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

        image_shape = x[0].shape[1:-1]

        # x = add_positional_encoding(x, self.model.input_timesteps)
        # x_test = add_positional_encoding(x_test, self.model.input_timesteps)

        # Add 2 to the number of features since we add positional encoding (x, y)
        # x_graph = [image_to_graph(img, num_features=self.input_features+2, thresh=self.thresh) for img in x]
        # x_test_graph = [image_to_graph(img, num_features=self.input_features+2, thresh=self.thresh) for img in x_test]
            
        self.model.to(self.device)
        self.model.train()

        # loss_func = torch.nn.MSELoss()
        loss_func = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=10, gamma=lr_decay)

        test_loss = []
        train_loss = []

        st = time.time()
        for epoch in range(n_epochs): 
            running_loss = 0
            step = 0
            for i in tqdm(range(len(x)), leave=False):

                x_batch_img = x[[i]]  # 2D images (num_timesteps, x, y)
                x_batch_img = add_positional_encoding(x_batch_img, self.model.input_timesteps).squeeze(0)


                x_graph = image_to_graph(x_batch_img, num_features=self.input_features+2, thresh=self.thresh)

                # Create a PyG graph object
                graph = create_graph_structure(x_graph['graph_nodes'], x_graph['distances'])

                x_batch = x_graph['data']  # Image in graph format
                
                # Turn target frame into graph using the graph structure from the input frames
                y_batch, _, _ = flatten(y[i], x_graph['labels'], num_features=1)

                for j in range(self.multi_step_loss):
                    graph.x = torch.from_numpy(x_batch).float()
                    graph.y = torch.from_numpy(y_batch).float()

                    graph.to(self.device)

                    optimizer.zero_grad()
                    
                    y_hat = self.model(graph.x, graph.edge_index, graph.edge_attr)
                    loss = loss_func(y_hat, graph.y[j])

                    loss.backward()
                    optimizer.step()

                    step += 1
                    running_loss += loss

                    if self.multi_step_loss > 1:

                        # Add predicted frame to the X
                        y_hat = np.expand_dims(y_hat.detach().numpy(), 0)
                        y_hat_img = unflatten(y_hat, x_graph['graph_nodes'], x_graph['mappings'], image_shape=image_shape)
                        y_hat_img = np.expand_dims(y_hat_img, (0))
                        y_hat_img = add_positional_encoding(y_hat_img, num_timesteps=1)
                        x_batch_img = np.concatenate([x_batch_img[1:], y_hat_img[0]], 0)

                        # Generate new graph using the new X
                        x_graph = image_to_graph(x_batch_img, num_features=self.input_features+2, thresh=self.thresh)

                        # Create a PyG graph object
                        graph = create_graph_structure(x_graph['graph_nodes'], x_graph['distances'])

                        x_batch = x_graph['data']  # Image in graph format
                        
                        # Turn target frame into graph using the graph structure from the input frames
                        y_batch, _, _ = flatten(y[i], x_graph['labels'], num_features=1)


            running_loss_test = 0
            step_test = 0
            for i in range(len(x_test)):

                x_batch_test_img = x_test[[i]]  # 2D images (num_timesteps, x, y)
                x_batch_test_img = add_positional_encoding(x_batch_test_img, self.model.input_timesteps).squeeze(0)

                x_test_graph = image_to_graph(x_batch_test_img, num_features=self.input_features+2, thresh=self.thresh)

                graph = create_graph_structure(x_test_graph['graph_nodes'], x_test_graph['distances'])

                x_batch = x_test_graph['data']

                # Turn target frame into graph using the graph structure from the input frames
                y_batch, _, _ = flatten(y_test[i], x_test_graph['labels'], num_features=1)

                for j in range(self.multi_step_loss):
                    graph.x = torch.from_numpy(x_batch).float()
                    graph.y = torch.from_numpy(y_batch).float()

                    graph.to(self.device)
                    y_hat = self.model(graph.x, graph.edge_index, graph.edge_attr)
                    loss = loss_func(y_hat, graph.y[j])

                    step_test += 1
                    running_loss_test += loss

                    if self.multi_step_loss > 1:

                        # Add predicted frame to the X
                        y_hat = np.expand_dims(y_hat.detach().numpy(), 0)
                        y_hat_img = unflatten(y_hat, x_test_graph['graph_nodes'], x_test_graph['mappings'], image_shape=image_shape)
                        y_hat_img = np.expand_dims(y_hat_img, (0))
                        y_hat_img = add_positional_encoding(y_hat_img, num_timesteps=1)
                        x_batch_img = np.concatenate([x_batch_img[1:], y_hat_img[0]], 0)
                        # x_batch_img = np.concatenate([x_batch_img[1:], y_hat_img], 0)

                        # Generate new graph using the new X
                        x_test_graph = image_to_graph(x_batch_img, num_features=self.input_features+2, thresh=self.thresh)

                        # Create a PyG graph object
                        graph = create_graph_structure(x_test_graph['graph_nodes'], x_test_graph['distances'])

                        x_batch = x_test_graph['data']  # Image in graph format
                        
                        # Turn target frame into graph using the graph structure from the input frames
                        y_batch, _, _ = flatten(y[i], x_test_graph['labels'], num_features=1)

            running_loss = running_loss / (step + 1)
            running_loss_test = running_loss_test / (step_test + 1)

            scheduler.step()

            train_loss.append(running_loss.item())
            test_loss.append(running_loss_test.item())
            
            print(f"Epoch {epoch} train loss: {running_loss.item():.4f}, "+ \
                f"test loss: {running_loss_test.item():.4f}, lr: {scheduler.get_last_lr()[0]:.4f}, time_per_epoch: {(time.time() - st) / (epoch+1):.1f}")
        
        print(f'Finished in {(time.time() - st)/60} minutes')

        self.loss = pd.DataFrame({
            'train_loss': train_loss,
            'test_loss': test_loss,

        })

    def save(self, directory):
        torch.save(self.model.state_dict(), os.path.join(directory, f'{self.experiment_name}.pth'))

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, x, autoregressive_steps=1):
        self.model.eval()
        
        image_shape = x[0].shape[1:-1]

        x = add_positional_encoding(x, self.model.input_timesteps)

        y_pred = []
        for i in range(len(x)):

            x_batch_img = x[i]  # 2D images (num_timesteps, x, y)

            x_graph = image_to_graph(x_batch_img, num_features=self.input_features+2, thresh=self.thresh)

            # Create a PyG graph object
            graph = create_graph_structure(x_graph['graph_nodes'], x_graph['distances'])

            x_batch = x_graph['data']  # Image in graph format
            
            y_hat_batch = []
            for j in tqdm(range(autoregressive_steps)):
                graph.x = torch.from_numpy(x_batch).float()
                # graph.y = torch.from_numpy(y_batch).float()

                graph.to(self.device)
                y_hat = self.model(graph.x, graph.edge_index, graph.edge_attr)

                # if self.multi_step_loss > 1:

                # Add predicted frame to the X
                y_hat = np.expand_dims(y_hat.detach().numpy(), 0)
                y_hat_img = unflatten(y_hat, x_graph['graph_nodes'], x_graph['mappings'], image_shape=image_shape)
                y_hat_img = np.expand_dims(y_hat_img, (0))
                
                y_hat_img = add_positional_encoding(y_hat_img, num_timesteps=1)

                x_batch_img = np.concatenate([x_batch_img[1:], y_hat_img[0]], 0)

                # Generate new graph using the new X
                x_graph = image_to_graph(x_batch_img, num_features=self.input_features+2, thresh=self.thresh)

                # Create a PyG graph object
                graph = create_graph_structure(x_graph['graph_nodes'], x_graph['distances'])

                x_batch = x_graph['data']  # Image in graph format
                
                # Turn target frame into graph using the graph structure from the input frames
                y_hat_batch.append(y_hat_img[0, 0, ..., 0])
            y_pred.append(y_hat_batch)

        return np.array(y_pred)


    def score(self, x, y, autoregressive_steps=1):

        # metric = torch.nn.MSELoss()
        metric = torch.nn.BCELoss()

        y_hat = self.predict(x, autoregressive_steps=autoregressive_steps)

        score = metric(torch.Tensor(y_hat), torch.Tensor(y[..., 0]))  # These indices are weird. Look into it. 
        return score
