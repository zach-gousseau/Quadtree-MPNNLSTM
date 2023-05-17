import numpy as np
import time
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os 
import datetime
from tqdm import tqdm
import warnings

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, TransformerConv
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
from torchviz import make_dot

import torch.autograd.profiler as profiler

from torch.optim.lr_scheduler import StepLR

from graph_functions import image_to_graph, flatten, create_graph_structure, unflatten, plot_contours
from model import MPNNLSTMI, MPNNLSTM
from seq2seq import Seq2Seq
from utils import get_n_params, add_positional_encoding, int_to_datetime

from abc import ABC, abstractmethod


class NextFramePredictor(ABC):
    def __init__(self, 
                 thresh,
                 experiment_name='experiment', 
                 decompose=True, 
                 input_features=1,
                 transform_func=None,
                 condition='max_larger_than',
                 device=None):

        self.experiment_name = experiment_name

        # Set the threshold to negative infinity if we want to keep the full basis (i.e. split all the way down)
        self.thresh = None if decompose else -np.inf
        self.decompose = decompose
        
        self.model = None
        
        self.thresh = thresh
        self.transform_func = transform_func
        self.condition = condition
        self.input_features = input_features 
        
        self.device = device

    def test_threshold(self, x, thresh, mask=None, contours=True):
        n_sample, w, h, c = x.shape
        image_shape = (w, h)

        x_with_pos_encoding = add_positional_encoding(x)

        graph = image_to_graph(x_with_pos_encoding, thresh=thresh, mask=mask, transform_func=self.transform_func)
        img_reconstructed = unflatten(graph['data'][..., [0]], graph['mapping'], image_shape)

        num_nodes = len(np.unique(graph['labels']))
        fig, axs = plt.subplots(1, n_sample, figsize=(5*n_sample, 4))

        for i in range(n_sample):
            axs[i].imshow(img_reconstructed[i, ..., 0])
            if contours:
                plot_contours(axs[i], graph['labels'])

        plt.suptitle(f'Threshold: {thresh} | Num. nodes: {num_nodes}')
        return fig, axs

    def get_n_params(self):
        return get_n_params(self.model)

    def save(self, directory):
        torch.save(self.model.state_dict(), os.path.join(directory, f'{self.experiment_name}.pth'))

    def load(self, directory):
        try:
            self.model.load_state_dict(torch.load(os.path.join(directory, f'{self.experiment_name}.pth')))
        except:
            self.model.load_state_dict(torch.load(os.path.join(directory, f'{self.experiment_name}.pth'), map_location=torch.device('cpu')))

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

class NextFramePredictorS2S(NextFramePredictor):
    def __init__(self,
                 thresh,
                 experiment_name='experiment', 
                 decompose=True, 
                 input_features=1,
                 input_timesteps=3,
                 output_timesteps=3,
                 device=None,
                 transform_func=None,
                 condition='max_larger_than',
                 remesh_input=False,
                 binary=False,
                 debug=False,
                 model_kwargs={}):
        
        super().__init__(
                 thresh=thresh,
                 experiment_name=experiment_name, 
                 decompose=decompose, 
                 input_features=input_features,
                 device=device,
                 transform_func=transform_func,
                 condition=condition)
        
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.binary = binary
        self.debug = debug
        
        self.model = Seq2Seq(
            input_features=input_features + 3,  # 3 (node_size)
            input_timesteps=input_timesteps,
            output_timesteps=output_timesteps,
            thresh=thresh,
            device=device,
            remesh_input=remesh_input,
            binary=binary,
            debug = debug,
            **model_kwargs
        ).to(device)

        self.training_initiated = False


    def initiate_training(self, lr, lr_decay):
        self.loss_func = torch.nn.MSELoss() if not self.binary else torch.nn.BCELoss()
        self.loss_func_name = 'MSE' if not self.binary else 'BCE'
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=lr_decay)

        # scaler = amp.GradScaler()
        
        self.writer = SummaryWriter('runs/' + self.experiment_name + '_' + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S"))

        self.test_loss = []
        self.train_loss = []

        self.training_initiated = True
    
    # @profile
    def train(
        self,
        loader_train,
        loader_test,
        climatology=None,
        n_epochs=200,
        lr=0.01,
        lr_decay=0.95,
        mask=None,
        truncated_backprop=45,
        ):

        image_shape = loader_train.dataset.image_shape

        if not self.training_initiated:
            self.initiate_training(lr, lr_decay)

        if mask is not None:
            assert mask.shape == image_shape, f'Mask and image shapes do not match. Got {mask.shape} and {image_shape}'

        st = time.time()
        batch_step = 0
        batch_step_test = 0
        for epoch in range(n_epochs): 
            running_loss = 0
            step = 0
            
            for x, y, launch_date in tqdm(loader_train, leave=True):

                x, y = x.squeeze(0).to(self.device), y.squeeze(0).to(self.device)
                
                if climatology is not None:
                    skip = self.get_climatology_array(climatology, launch_date)
                else:
                    skip = None
                
                if truncated_backprop == 0:
                    self.optimizer.zero_grad()
                
                    # with amp.autocast():
                    y_hat, y_hat_mappings = self.model(x, y, skip, teacher_forcing_ratio=0, mask=mask)
                    
                    y_hat = [unflatten(y_hat[i], y_hat_mappings[i], image_shape, mask) for i in range(self.output_timesteps)]
                    y_hat = torch.stack(y_hat, dim=0)
                    
                    loss = self.loss_func(y_hat[:, ~mask], y[:, ~mask])  

                    # loss = 0 
                    # for t in range(len(y)):
                    #     loss_step = self.loss_func(y_hat[t, ~mask], y[t, ~mask]) 

                    #     loss_step.backward(retain_graph=True)

                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

                    # loss += loss_step



                    # scaler.scale(loss).backward()
                    # scaler.step(self.optimizer)
                    # scaler.update()

                    # with profiler.profile(enabled=True, use_cuda=True) as prof:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)

                    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
                    # prof.export_chrome_trace("profiling_results.json")
                    # quit()

                    self.optimizer.step()

                    if self.debug:
                        # decoder_params = [p for p in self.model.decoder.parameters()]
                        # decoder_grads = [p.grad.mean().cpu() for p in decoder_params if p.grad is not None]
                        # decoder_param_means = [p.mean().abs().detach().cpu() for p in decoder_params]
                        # self.writer.add_scalar("Grad/decoder/mean", np.mean(np.abs(decoder_grads)), batch_step)
                        # self.writer.add_scalar("Param/decoder/mean", np.mean(decoder_param_means), batch_step)

                        # encoder_params = [p for p in self.model.encoder.parameters()]
                        # encoder_grads = [p.grad.mean().cpu() for p in encoder_params if p.grad is not None]
                        # encoder_param_means = [p.mean().abs().detach().cpu() for p in encoder_params]
                        # self.writer.add_scalar("Grad/encoder/mean", np.mean(np.abs(encoder_grads)), batch_step)
                        # self.writer.add_scalar("Param/encoder/mean", np.mean(encoder_param_means), batch_step)

                        # print("Grad/decoder/mean", np.mean(np.abs(decoder_grads)), batch_step)
                        # print("Param/decoder/mean", np.mean(decoder_param_means), batch_step)
                        # print("Grad/encoder/mean", np.mean(np.abs(encoder_grads)), batch_step)
                        # print("Param/encoder/mean", np.mean(encoder_param_means), batch_step)

                        en_grad_norms = torch.norm(torch.stack([torch.norm(param.grad.detach()) for param in self.model.encoder.parameters() if param.grad is not None]))
                        de_grad_norms = torch.norm(torch.stack([torch.norm(param.grad.detach()) for param in self.model.decoder.parameters() if param.grad is not None]))

                        self.writer.add_scalar("Grad/encoder/grad_norms", en_grad_norms, batch_step)
                        self.writer.add_scalar("Grad/decoder/grad_norms", de_grad_norms, batch_step)

                        # print("Grad/encoder/grad_norms", en_grad_norms, batch_step)
                        # print("Grad/decoder/grad_norms", de_grad_norms, batch_step)

                    del y_hat
                    del y_hat_mappings

                else:
                    output_timestep = 0
                    while output_timestep < self.output_timesteps:

                        output_timestep = min(output_timestep + truncated_backprop, self.output_timesteps+1)

                        unroll_steps = range(output_timestep-truncated_backprop, output_timestep)

                        self.optimizer.zero_grad()

                        # Encoder
                        self.model.process_inputs(x, mask=mask)

                        # Decoder
                        y_hat, y_hat_mappings = self.model.unroll_output(
                            unroll_steps,
                            y,
                            skip=skip,
                            teacher_forcing_ratio=0,
                            mask=mask,
                            remesh_every=1
                            )
                    
                        y_hat = [unflatten(y_hat[i], y_hat_mappings[i], image_shape, mask) for i in range(truncated_backprop)]
                        y_hat = torch.stack(y_hat, dim=0)
                        
                        loss = self.loss_func(y_hat[:, ~mask], y[output_timestep-truncated_backprop:output_timestep, ~mask])  
                        loss.backward(retain_graph=True)

                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=.5)

                        del y_hat, y_hat_mappings

                    self.optimizer.step()

                self.writer.add_scalar("Loss/train", loss.item(), batch_step)
                # print("Loss/train", loss.item(), batch_step)

                step += 1
                batch_step += 1
                running_loss += loss.item()
                torch.cuda.empty_cache()

            running_loss_test = 0
            step_test = 0
            for x, y, launch_date in tqdm(loader_test, leave=True):

                x, y = x.squeeze(0).to(self.device), y.squeeze(0).to(self.device)

                if climatology is not None:
                    skip = self.get_climatology_array(climatology, launch_date)
                else:
                    skip = None

                with torch.no_grad():
                    y_hat, y_hat_mappings = self.model(x, y, skip, teacher_forcing_ratio=0, mask=mask)

                    y_hat = [unflatten(y_hat[i], y_hat_mappings[i], image_shape, mask) for i in range(self.output_timesteps)]
                    y_hat = torch.stack(y_hat, dim=0)
                
                    loss = self.loss_func(y_hat[:, ~mask], y[:, ~mask])  

                step_test += 1
                running_loss_test += loss

                del y_hat
                del y_hat_mappings
                torch.cuda.empty_cache()


            running_loss = running_loss / (step + 1)
            running_loss_test = running_loss_test / (step_test + 1)

            if np.isnan(running_loss_test.item()):
                raise ValueError('NaN loss :(')

            if running_loss_test.item() > 4:
                raise ValueError('Diverged :(')

            self.writer.add_scalar("Loss/test", running_loss_test.item(), epoch)

            self.scheduler.step()

            self.train_loss.append(running_loss)
            self.test_loss.append(running_loss_test.item())
            
            print(f"{self.experiment_name} | Epoch {epoch} train {self.loss_func_name}: {running_loss:.4f}, "+ \
                f"test {self.loss_func_name}: {running_loss_test.item():.4f}, lr: {self.scheduler.get_last_lr()[0]:.4f}, time_per_epoch: {(time.time() - st) / (epoch+1):.1f}")
        
        print(f'Finished in {(time.time() - st)/60} minutes')
        
        self.writer.flush()

        self.loss = pd.DataFrame({
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,

        })

    def get_climatology_array(self, climatology, launch_date):
        doys = [int_to_datetime(launch_date.numpy()[0] + 8.640e13 * t).timetuple().tm_yday - 1 for t in range(0, self.output_timesteps)]

        skip = climatology[:, doys]
        skip = torch.moveaxis(skip, 0, -1)
        return skip
        
    def predict(self, loader, climatology=None, mask=None):
        
        image_shape = loader.dataset.image_shape
            
        self.model.to(self.device)
        
        y_pred = []
        for x, y, launch_date in tqdm(loader, leave=False):

            x = x.squeeze(0).to(self.device)

            if climatology is not None:
                skip = self.get_climatology_array(climatology, launch_date)
            else:
                skip = None

            with torch.no_grad():
                y_hat, y_hat_mappings = self.model(x, skip=skip, teacher_forcing_ratio=0, mask=mask)
                
                y_hat = [unflatten(y_hat[i], y_hat_mappings[i], image_shape, mask).detach().cpu() for i in range(self.output_timesteps)]
                
                y_hat = np.stack(y_hat)#.squeeze(1)
                
                y_pred.append(y_hat)

                del y_hat_mappings
                torch.cuda.empty_cache()
            
        return np.stack(y_pred, 0)

    def score(self, x, y, rollout=None):
        pass
