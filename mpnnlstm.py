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

from graph_functions import image_to_graph, flatten, create_graph_structure, unflatten, plot_contours
from model import MPNNLSTMI, MPNNLSTM
from utils import get_n_params, add_positional_encoding

from abc import ABC, abstractmethod


class NextFramePredictor(ABC):
    def __init__(self, 
                 model,
                 thresh,
                 experiment_name='experiment', 
                 decompose=True, 
                 input_features=1,
                 transform_func=None,
                 device=None):

        self.experiment_name = experiment_name

        # Set the threshold to negative infinity if we want to keep the full basis (i.e. split all the way down)
        self.thresh = None if decompose else -np.inf
        self.decompose = decompose

        self.model = model.to(device[0])
        # self.model = model
        
        self.thresh = thresh
        self.transform_func = transform_func
        self.input_features = input_features 
        
        if device == 'cpu' or device is None:
            device = [device]

        n_devices = len(device)
        
        self.device = self.model.device = device

    def test_threshold(self, x, thresh, mask=None):
        n_sample, w, h, c = x.shape
        image_shape = (w, h)

        x_with_pos_encoding = add_positional_encoding(x)

        graph = image_to_graph(x_with_pos_encoding, thresh=thresh, mask=mask, transform_func=self.transform_func)
        img_reconstructed = unflatten(graph['data'][..., [0]], graph['mapping'], image_shape)

        num_nodes = len(np.unique(graph['labels']))
        fig, axs = plt.subplots(1, n_sample, figsize=(5*n_sample, 4))

        for i in range(n_sample):
            axs[i].imshow(img_reconstructed[i, ..., 0])
            plot_contours(axs[i], graph['labels'])

        plt.suptitle(f'Threshold: {thresh} | Num. nodes: {num_nodes}')
        return fig, axs

    def get_n_params(self):
        return get_n_params(self.model)

    def save(self, directory):
        torch.save(self.model.state_dict(), os.path.join(directory, f'{self.experiment_name}.pth'))

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    @abstractmethod
    def train(
        self,
        loader_train,
        loader_test,
        n_epochs=200,
        lr=0.01,
        lr_decay=0.95,
        mask=None
        ):
        pass

    @abstractmethod
    def predict(self, x, mask=None, rollout=None):
        pass

    @abstractmethod
    def score(self, x, y, rollout=None):
        pass


class NextFramePredictorAR(NextFramePredictor):
    def __init__(self, 
                 model,
                 thresh,
                 experiment_name='experiment', 
                 decompose=True, 
                 input_features=1,
                 multi_step_loss=1, 
                 device=None):
        super().__init__(
                 model,
                 thresh=thresh,
                 experiment_name=experiment_name, 
                 decompose=decompose, 
                 input_features=input_features,
                 device=device)

        self.multi_step_loss = multi_step_loss


    def train(
        self,
        loader_train,
        loader_test,
        n_epochs=200,
        lr=0.01,
        lr_decay=0.95,
        mask=None
        ):

        image_shape = x[0].shape[1:-1]

        if mask is not None:
            assert mask.shape == image_shape
            
        self.model.to(self.device)
        self.model.train()

        # loss_func = torch.nn.MSELoss()
        loss_func = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=10, gamma=lr_decay)

        scaler = torch.cuda.amp.GradScaler()

        test_loss = []
        train_loss = []

        st = time.time()
        for epoch in range(n_epochs): 
            running_loss = 0
            step = 0
            for i in tqdm(range(len()), leave=False):

                x_batch_img = x[[i]]  # 2D images (num_timesteps, x, y)
                x_batch_img = add_positional_encoding(x_batch_img).squeeze(0)


                x_graph = image_to_graph(x_batch_img, thresh=self.thresh, mask=mask)

                # Create a PyG graph object
                graph = create_graph_structure(x_graph['graph_nodes'], x_graph['distances'])

                x_batch = x_graph['data']  # Image in graph format

                # Turn target frame into graph using the graph structure from the input frames
                y_batch, _ = flatten(y[i], x_graph['labels'])

                for j in range(self.multi_step_loss):
                    graph.x = torch.from_numpy(x_batch).float()
                    graph.y = torch.from_numpy(y_batch).float()

                    graph.to(self.device)

                    optimizer.zero_grad()
                    
                    y_hat = self.model(graph.x, graph.edge_index, graph.edge_attr)
                    loss = loss_func(y_hat, graph.y[j])

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # loss.backward()
                    # optimizer.step()

                    step += 1
                    running_loss += loss

                    if self.multi_step_loss > 1:

                        # Add predicted frame to the X
                        y_hat = np.expand_dims(y_hat.detach().numpy(), 0)
                        y_hat_img = unflatten(y_hat, x_graph['graph_nodes'], x_graph['mappings'], image_shape=image_shape)
                        y_hat_img = np.expand_dims(y_hat_img, (0))
                        y_hat_img = add_positional_encoding(y_hat_img)
                        x_batch_img = np.concatenate([x_batch_img[1:], y_hat_img[0]], 0)

                        # Generate new graph using the new X
                        x_graph = image_to_graph(x_batch_img, thresh=self.thresh, mask=mask)

                        # Create a PyG graph object
                        graph = create_graph_structure(x_graph['graph_nodes'], x_graph['distances'])

                        x_batch = x_graph['data']  # Image in graph format
                        
                        # Turn target frame into graph using the graph structure from the input frames
                        y_batch, _ = flatten(y[i], x_graph['labels'])


            running_loss_test = 0
            step_test = 0
            for i in range(len(x_test)):

                x_batch_test_img = x_test[[i]]  # 2D images (num_timesteps, x, y)
                x_batch_test_img = add_positional_encoding(x_batch_test_img).squeeze(0)

                x_test_graph = image_to_graph(x_batch_test_img, thresh=self.thresh, mask=mask)

                graph = create_graph_structure(x_test_graph['graph_nodes'], x_test_graph['distances'])

                x_batch = x_test_graph['data']

                # Turn target frame into graph using the graph structure from the input frames
                y_batch, _ = flatten(y_test[i], x_test_graph['labels'])

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
                        y_hat_img = add_positional_encoding(y_hat_img)
                        x_batch_img = np.concatenate([x_batch_img[1:], y_hat_img], 0)
                        # x_batch_img = np.concatenate([x_batch_img[1:], y_hat_img], 0)

                        # Generate new graph using the new X
                        x_test_graph = image_to_graph(x_batch_img, thresh=self.thresh, mask=mask)

                        # Create a PyG graph object
                        graph = create_graph_structure(x_test_graph['graph_nodes'], x_test_graph['distances'])

                        x_batch = x_test_graph['data']  # Image in graph format
                        
                        # Turn target frame into graph using the graph structure from the input frames
                        y_batch, _ = flatten(y[i], x_test_graph['labels'])

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

    def predict(self, x, rollout=1, mask=None):
        self.model.eval()
        
        image_shape = x[0].shape[1:-1]

        x = add_positional_encoding(x)

        y_pred = []
        for i in range(len(x)):

            x_batch_img = x[i]  # 2D images (num_timesteps, x, y)

            x_graph = image_to_graph(x_batch_img, thresh=self.thresh, mask=mask)

            # Create a PyG graph object
            graph = create_graph_structure(x_graph['graph_nodes'], x_graph['distances'])

            x_batch = x_graph['data']  # Image in graph format
            
            y_hat_batch = []
            for j in range(rollout):
                graph.x = torch.from_numpy(x_batch).float()
                # graph.y = torch.from_numpy(y_batch).float()

                graph.to(self.device)
                y_hat = self.model(graph.x, graph.edge_index, graph.edge_attr)

                # if self.multi_step_loss > 1:

                # Add predicted frame to the X
                y_hat = np.expand_dims(y_hat.cpu().detach().numpy(), 0)
                y_hat_img = unflatten(y_hat, x_graph['graph_nodes'], x_graph['mappings'], image_shape=image_shape)
                y_hat_img = np.expand_dims(y_hat_img, (0))
                
                y_hat_img = add_positional_encoding(y_hat_img)

                x_batch_img = np.concatenate([x_batch_img[1:], y_hat_img[0]], 0)

                # Generate new graph using the new X
                x_graph = image_to_graph(x_batch_img, thresh=self.thresh, mask=mask)

                # Create a PyG graph object
                graph = create_graph_structure(x_graph['graph_nodes'], x_graph['distances'])

                x_batch = x_graph['data']  # Image in graph format
            
                y_hat_batch.append(y_hat_img[0, 0, ..., :self.input_features])
            y_pred.append(y_hat_batch)

        return np.array(y_pred)


    def score(self, x, y, rollout=1):

        # metric = torch.nn.MSELoss()
        metric = torch.nn.BCELoss()

        y_hat = self.predict(x, rollout=rollout)

        score = metric(torch.Tensor(y_hat), torch.Tensor(y))
        return score


class NextFramePredictorS2S(NextFramePredictor):
    def __init__(self,
                 model,
                 thresh,
                 experiment_name='experiment', 
                 decompose=True, 
                 input_features=1,
                 output_timesteps=3,
                 device=None,
                 transform_func=None,
                 **model_kwargs):
        
        super().__init__(
                 model=model,
                 thresh=thresh,
                 experiment_name=experiment_name, 
                 decompose=decompose, 
                 input_features=input_features,
                 device=device,
                 transform_func=transform_func,
                 **model_kwargs)
        
        self.output_timesteps = output_timesteps

    def train(
        self,
        loader_train,
        loader_test,
        n_epochs=200,
        lr=0.01,
        lr_decay=0.95,
        mask=None
        ):

        image_shape = loader_train.dataset.image_shape

        if mask is not None:
            assert mask.shape == image_shape, f'Mask and image shapes do not match. Got {mask.shape} and {image_shape}'
            
        # model = nn.DataParallel(self.model)

        loss_func = torch.nn.BCELoss()  # torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=3, gamma=lr_decay)

        test_loss = []
        train_loss = []

        st = time.time()
        for epoch in range(n_epochs): 
            running_loss = 0
            step = 0
            
            for x, y in tqdm(loader_train, leave=False):

                x, y = x.squeeze(0), y.squeeze(0)
                    
                # for j in range(n_devices):
                x = add_positional_encoding(x)

                x_graph = image_to_graph(x, thresh=self.thresh, mask=mask, transform_func=self.transform_func)

                # Create a PyG graph object
                graph = create_graph_structure(x_graph['graph_nodes'], x_graph['distances'])

                x_batch = x_graph['data']  # Image in graph format

                # graph.x = torch.from_numpy(x_batch).float()
                graph.x = x_batch
                graph.y = y

                graph.input_graph_structure = x_graph
                graph.image_shape = image_shape
                graph.to(self.device[0])

                optimizer.zero_grad()

                skip = graph.x[-1, :, [0]]  # 0th index variable is the variable of interest
                
                y_hat, y_hat_graph = self.model(graph, image_shape=image_shape, teacher_forcing_ratio=0.5, mask=mask)

                # Transform 
                y_true = [torch.Tensor(flatten(graph.y[[i]], y_hat_graph[i]['mapping'], y_hat_graph[i]['n_pixels_per_node']))[0] for i in range(self.output_timesteps)]

                y_hat = torch.cat(y_hat, dim=0)
                y_true = torch.cat(y_true, dim=0)

                y_true = y_true.to(self.device[0])  # TODO: Somehow y_true has to be distributed to all GPUs....
                
                loss = loss_func(y_hat, y_true)

                loss.backward()
                optimizer.step()

                step += 1
                running_loss += loss

            running_loss_test = 0
            step_test = 0
            for x, y in loader_test:

                x = add_positional_encoding(x)

                x_test_graph = image_to_graph(x, thresh=self.thresh, mask=mask, transform_func=self.transform_func)

                graph = create_graph_structure(x_test_graph['graph_nodes'], x_test_graph['distances'])

                x_batch = x_test_graph['data']
                
                
                graph.x = torch.from_numpy(x_batch).float()
                graph.y = y

                graph.input_graph_structure = x_test_graph
                graph.image_shape = image_shape

                skip = graph.x[-1, :, [0]]  # 0th index variable is the variable of interest
                graph.skip = skip

                graph.to(self.device[0])
                
                y_hat, y_hat_graph = self.model(graph, image_shape=image_shape, teacher_forcing_ratio=0.5, mask=mask)
                
                y_true = [torch.Tensor(flatten(graph.y[[i]], y_hat_graph[i]['mapping'],  y_hat_graph[i]['n_pixels_per_node']))[0] for i in range(self.output_timesteps)]
                
                y_hat = torch.cat(y_hat, dim=0).to(self.device[0])
                y_true = torch.cat(y_true, dim=0).to(self.device[0])
                
                loss = loss_func(y_hat, y_true)

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
        
    def predict(self, loader, mask=None):
        
        image_shape = loader.dataset.image_shape
            
        self.model.to(self.device[0])
        
        y_pred = []
        for x, y in tqdm(loader, leave=False):

            # for j in range(n_devices):
            x = add_positional_encoding(x)

            x_graph = image_to_graph(x, thresh=self.thresh, mask=mask, transform_func=self.transform_func)

            # Create a PyG graph object
            graph = create_graph_structure(x_graph['graph_nodes'], x_graph['distances'])

            x_batch = x_graph['data']  # Image in graph format

            graph.x = torch.from_numpy(x_batch).float()

            graph.input_graph_structure = x_graph
            graph.image_shape = image_shape
            graph.to(self.device[0])

            skip = graph.x[-1, :, [0]]  # 0th index variable is the variable of interest

            y_hat, y_hat_graph = self.model(graph, image_shape=image_shape, teacher_forcing_ratio=0, mask=mask)
            
            y_hat = [unflatten(np.expand_dims(y_hat[i].detach(), 0), y_hat_graph[i]['mapping'], image_shape) for i in range(self.output_timesteps)]
            
            y_hat = np.stack(y_hat).squeeze(1)
            
            y_pred.append(y_hat)
            
        return np.stack(y_pred, 0)

    def score(self, x, y, rollout=None):
        pass
