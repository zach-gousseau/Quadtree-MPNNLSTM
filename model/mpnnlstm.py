import numpy as np
import time
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os 
import datetime
from tqdm import tqdm
import random
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

from model.graph_functions import image_to_graph, flatten, Graph, unflatten, plot_contours
from model.model import MPNNLSTMI, MPNNLSTM
from model.seq2seq import Seq2Seq
from model.utils import get_n_params, add_positional_encoding, int_to_datetime

from abc import ABC, abstractmethod

from pytorch_msssim import ssim

class SobelLoss(nn.Module):
    def __init__(self):
        super(SobelLoss, self).__init__()

        # Define Sobel filters
        self.sobel_x = nn.Parameter(torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False)

    def forward(self, pred, true, mask=None):
        pred_sobel_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_sobel_y = F.conv2d(pred, self.sobel_y, padding=1)
        
        true_sobel_x = F.conv2d(true, self.sobel_x, padding=1)
        true_sobel_y = F.conv2d(true, self.sobel_y, padding=1)
        
        if mask is not None:
            pred_sobel_x, true_sobel_x = pred_sobel_x[:, :, ~mask], true_sobel_x[:, :, ~mask]
            pred_sobel_y, true_sobel_y = pred_sobel_y[:, :, ~mask], true_sobel_y[:, :, ~mask]

        loss_x = F.mse_loss(pred_sobel_x, true_sobel_x)
        loss_y = F.mse_loss(pred_sobel_y, true_sobel_y)

        return loss_x + loss_y
    
class MSE_masked(nn.Module):
    def __init__(self):
        super(MSE_masked, self).__init__()

    def forward(self, pred, true, mask=None):
        
        if mask is not None:
            pred, true = pred[:, ~mask], true[:, ~mask]

        return F.mse_loss(pred, true)

class MSE_NIIEE(nn.Module):
    def __init__(self):
        super(MSE_NIIEE, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.niiee = NIIEE()
        
    def forward(self, output, target):
        loss = 0.1*self.niiee(output, target) + self.mse(output, target)
        return loss
        
class NIIEE(nn.Module):
    def __init__(self):
        super(NIIEE, self).__init__()

    def forward(self, output, target):
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target) - intersection
        loss = 1 - intersection / union
        return loss
    
class MSE_SSIM(nn.Module):
    def __init__(self):
        super(MSE_SSIM, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.ssim = ssim
        
    def forward(self, output, target):
        loss = 0.1*self.ssim(output.unsqueeze(1), target.unsqueeze(1), data_range=1) + self.mse(output, target)
        return loss
    
class MSE_Sobel(nn.Module):
    def __init__(self):
        super(MSE_Sobel, self).__init__()
        self.mse = MSE_masked()
        self.sobel = SobelLoss()
        
    def forward(self, output, target, mask=None):
        loss = 0.01*self.sobel(output.moveaxis(-1, 1), target.moveaxis(-1, 1), mask) + self.mse(output, target, mask)
        return loss
    
# class MSE_SIP(nn.Module):
#     def __init__(self):
#         super(MSE_SIP, self).__init__()
#         self.mse = torch.nn.MSELoss()
        
#     def sip_loss(self, pred, true):
#         return 1
        
#     def forward(self, output, target):
#         loss = self.sip_loss(output, target) + self.mse(output, target)
#         return loss
    
class MSE_SIP(nn.Module):
    def __init__(self):
        super(MSE_SIP, self).__init__()
        
    def weight_func(self, x):
        return 4*(x-0.5)**2 + 1
        
    def forward(self, output, target, mask=None, weights=None):
        if mask is not None:
            output, target = output[:, ~mask], target[:, ~mask]
        # alpha_tensor = ((target < 0.15) | (target > 0.85)) * (alpha-1) + 1
        alpha_tensor = self.weight_func(target)
        # output = torch.sigmoid(output)
        loss = (output - target) ** 2
        # loss = (loss * alpha_tensor)
        # if weights is not None:
            # loss = loss * weights[None, :, None]
        return loss.mean()
    
class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()
        # self.bce = torch.nn.BCELoss(reduction='none')
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, output, target, mask=None, weights=None):
        if mask is not None:
            output, target = output[:, ~mask], target[:, ~mask]
        
        # output, target  = output > 0.15, target > 0.15
        # output = torch.sigmoid(output)
        target  = (target > 0.15).float()
        loss = self.bce(output, target)
        
        if weights is not None:
            loss = loss * weights[None, :, None]
        return loss.mean()

class MSE_SIP_bin(nn.Module):
    def __init__(self):
        super(MSE_SIP_bin, self).__init__()
        self.mse = MSE_SIP()
        self.bce = BCE()
        
    def forward(self, output, target, mask=None, weights=None, alpha=1):
        mse_loss = self.mse(output, target, mask, weights)
        bce_loss = self.bce(output, target, mask, weights) * 0.1
        loss = alpha * mse_loss + (1-alpha) * bce_loss
        return loss
    
class MSE_SIP_bin_sep(nn.Module):
    def __init__(self):
        super(MSE_SIP_bin_sep, self).__init__()
        self.mse = MSE_SIP()
        self.bce = BCE()
        
    def forward(self, output, target, mask=None, weights=None):
        mse_loss = self.mse(output[..., [0]], target, mask, weights)
        bce_loss = self.bce(output[..., [1]], target, mask, weights) * 0.1
        loss = mse_loss + bce_loss
        return loss


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
                 directory='',
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
        
        # Model parameters
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.input_features = input_features 
        self.binary = binary
        
        # Quadtree decomposition parameters
        self.thresh = thresh if decompose else -np.inf
        self.decompose = decompose
        self.transform_func = transform_func
        self.condition = condition

        self.experiment_name = experiment_name
        self.directory = directory
        self.debug = debug
        self.device = device
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Model 
        self.model = Seq2Seq(
            input_features=input_features + 3,  # Add 3 for the positional encoding (x, y) and node size features
            input_timesteps=input_timesteps,
            output_timesteps=output_timesteps,
            thresh=thresh,
            device=device,
            remesh_input=remesh_input,
            binary=binary,
            debug = debug,
            **model_kwargs
        ).to(device)

        # To allow calling train() multiple times
        self.training_initiated = False

    def test_threshold(self, x, thresh, mask=None, high_interest_region=None, contours=True):
        n_sample, w, h, c = x.shape
        image_shape = (w, h)

        x_with_pos_encoding = add_positional_encoding(x)

        graph = image_to_graph(x_with_pos_encoding, thresh=thresh, mask=mask, high_interest_region=high_interest_region, transform_func=self.transform_func)
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

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.directory, f'{self.experiment_name}.pth'))

    def load(self, directory):
        try:
            self.model.load_state_dict(torch.load(os.path.join(directory, f'{self.experiment_name}.pth')))
        except:
            self.model.load_state_dict(torch.load(os.path.join(directory, f'{self.experiment_name}.pth'), map_location=torch.device('cpu')))

    def initiate_training(self, lr, lr_decay, mask):
        # self.loss_func = MSE_NIIEE() if not self.binary else torch.nn.BCELoss()
        # self.loss_func_name = 'MSE+0.1NIEE' if not self.binary else 'BCE'  # For printing
        
        # self.loss_func = torch.nn.MSELoss() if not self.binary else torch.nn.BCELoss()
        # self.loss_func_name = 'MSE' if not self.binary else 'BCE'  # For printing
        
        # self.loss_func = MSE_weighted() if not self.binary else torch.nn.BCELoss()
        # self.loss_func_name = 'MSE' if not self.binary else 'BCE'  # For printing
        
        self.loss_func = MSE_SIP_bin_sep() if not self.binary else torch.nn.BCELoss()
        self.loss_func_name = 'MSE_SSIM' if not self.binary else 'BCE'  # For printing
        
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)#, weight_decay=0.001)
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=lr_decay)

        self.scaler = amp.GradScaler()  # Not used 
        
        self.writer = SummaryWriter('runs/' + self.experiment_name + '_' + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S"))

        self.test_loss = []
        self.train_loss = []

        self.training_initiated = True
        
        self.min_loss = np.inf
    
    # @profile
    def train(
        self,
        loader_train,
        loader_test,
        climatology=None,
        climatology_test=None,
        n_epochs=200,
        lr=0.01,
        lr_decay=0.95,
        mask=None,
        high_interest_region=None,
        truncated_backprop=45,
        graph_structure=None,
        graph_structure_test=None,
        ):

        image_shape = loader_train.dataset.image_shape
        
        # Initialize training only if it's the first train() call
        if not self.training_initiated:
            self.initiate_training(lr, lr_decay, mask)

        # if mask is not None:
            # assert mask.shape == image_shape, f'Mask and image shapes do not match. Got {mask.shape} and {image_shape}'

        # Training loop
        st = time.time()
        batch_step = 0
        batch_step_test = 0
        for epoch in range(n_epochs): 

            # Loop over training set
            running_loss = 0
            step = 0
            
            self.model.train()
            self.model.epoch = epoch
            for x, y, launch_date in tqdm(loader_train, leave=True):

                x, y = x.squeeze(0).to(self.device), y.squeeze(0).to(self.device)
                
                if climatology is not None:
                    concat_layers = self.get_climatology_array(climatology, launch_date)
                else:
                    concat_layers = None
                
                # Single-step forward/backward pass if no truncated backpropogation, otherwise split into truncated_backprop chunks
                if truncated_backprop == 0:
                    self.optimizer.zero_grad()
                
                    with amp.autocast():
                        y_hat, y_hat_mappings = self.model(
                            x, 
                            y, 
                            concat_layers, 
                            teacher_forcing_ratio=0, 
                            mask=mask, 
                            high_interest_region=high_interest_region,
                            graph_structure=graph_structure
                            )
                        # print(y_hat[0].dtype, y_hat_mappings[0].dtype)
                        # with amp.autocast(enabled=False):
                        #     if graph_structure is not None:
                        #         y_hat = unflatten(torch.stack(y_hat), graph_structure['mapping'], image_shape, mask)
                        #     else:
                        #         y_hat = [unflatten(y_hat[i], y_hat_mappings[i], image_shape, mask) for i in range(self.output_timesteps)]
                        #         y_hat = torch.stack(y_hat, dim=0)
                        
                        # loss = self.loss_func(y_hat[:, ~mask], y[:, ~mask])  
                        y_hat = torch.stack(y_hat, dim=0)
                        loss = self.loss_func(y_hat, y, weights=graph_structure['n_pixels_per_node'])  
                        
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        # loss = self.loss_func(y_hat, y)  

                        # with profiler.profile(enabled=True, use_cuda=True) as prof:
                        # loss.backward()

                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)

                        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
                        # prof.export_chrome_trace("profiling_results.json")
                        # quit()

                        # self.optimizer.step()

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

                        en_grad_norms = torch.norm(torch.stack([torch.norm(param.grad.detach()) for param in self.model.encoder.parameters() if param.grad is not None]))
                        de_grad_norms = torch.norm(torch.stack([torch.norm(param.grad.detach()) for param in self.model.decoder.parameters() if param.grad is not None]))
                        self.writer.add_scalar("Grad/encoder/grad_norms", en_grad_norms, batch_step)
                        self.writer.add_scalar("Grad/decoder/grad_norms", de_grad_norms, batch_step)

                    del y_hat
                    del y_hat_mappings

                else:
                    output_timestep = 0
                    while output_timestep < self.output_timesteps:

                        output_timestep = min(output_timestep + truncated_backprop, self.output_timesteps+1)

                        unroll_steps = range(output_timestep-truncated_backprop, output_timestep)

                        self.optimizer.zero_grad()

                        # Encoder
                        self.model.process_inputs(x, mask=mask, high_interest_region=high_interest_region, graph_structure=graph_structure)

                        # Decoder
                        y_hat, y_hat_mappings = self.model.unroll_output(
                            unroll_steps,
                            y,
                            concat_layers=concat_layers,
                            teacher_forcing_ratio=0,
                            mask=mask,
                            high_interest_region=high_interest_region,
                            remesh_every=1
                            )
                    
                        y_hat = [unflatten(y_hat[i], y_hat_mappings[i], image_shape, mask) for i in range(len(y_hat))]
                        y_hat = torch.stack(y_hat, dim=0)
                        
                        # loss = self.loss_func(y_hat[:, ~mask], y[unroll_steps][:, ~mask])  
                        loss = self.loss_func(y_hat, y[unroll_steps])  
                        loss.backward(retain_graph=True)

                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=.5)

                        del y_hat, y_hat_mappings

                    self.optimizer.step()

                self.writer.add_scalar("Loss/train", loss.item(), batch_step)

                step += 1
                batch_step += 1
                running_loss += loss.item()
                torch.cuda.empty_cache()

            # Loop over test set
            running_loss_test = 0
            step_test = 0
            self.model.eval()
            for x, y, launch_date in tqdm(loader_test, leave=True):

                x, y = x.squeeze(0).to(self.device), y.squeeze(0).to(self.device)

                if climatology is not None:
                    concat_layers = self.get_climatology_array(climatology_test if climatology_test is not None else climatology, launch_date)
                else:
                    concat_layers = None

                with torch.no_grad():
                    with amp.autocast():
                        y_hat, y_hat_mappings = self.model(
                            x, 
                            y, 
                            concat_layers, 
                            teacher_forcing_ratio=0, 
                            mask=mask, 
                            high_interest_region=high_interest_region,
                            graph_structure=graph_structure_test if graph_structure_test is not None else graph_structure
                            )
                        # with amp.autocast(enabled=False):
                        #     if graph_structure is not None:
                        #         y_hat = unflatten(torch.stack(y_hat), graph_structure['mapping'], image_shape, mask)
                        #     else:
                        #         y_hat = [unflatten(y_hat[i], y_hat_mappings[i], image_shape, mask) for i in range(self.output_timesteps)]
                        #         y_hat = torch.stack(y_hat, dim=0)
                    
                        # loss = self.loss_func(y_hat[:, ~mask], y[:, ~mask])  
                        y_hat = torch.stack(y_hat, dim=0)
                        loss = self.loss_func(y_hat,
                                              y,
                                              weights=graph_structure_test['n_pixels_per_node'] if graph_structure_test is not None else graph_structure['n_pixels_per_node'])  

                step_test += 1
                running_loss_test += loss

                del y_hat
                del y_hat_mappings
                torch.cuda.empty_cache()


            running_loss = running_loss / (step + 1)
            running_loss_test = running_loss_test / (step_test + 1)
            
            if running_loss_test < self.min_loss:
                self.save()
                self.min_loss = running_loss_test

            if np.isnan(running_loss_test.item()):
                raise ValueError('NaN loss :(')

            # if running_loss_test.item() > 4:
            #     raise ValueError('Diverged :(')

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
        """
        Get the daily climate normals for each day of the year in the output timesteps

        climatology (np.ndarray): Climate normals tensor of shape (365, w, h)
        launch_date (np.datetime64 (?)): The launch date
        """
        doys = [int_to_datetime(launch_date.numpy()[0] + 8.640e13 * t).timetuple().tm_yday - 1 for t in range(1, self.output_timesteps+1)]

        out = climatology[:, doys]
        out = torch.moveaxis(out, 0, -1)
        return out
        
    def predict(self, loader, climatology=None, mask=None, high_interest_region=None, graph_structure=None):
        """
        Use model in inference mode.
        """
        
        image_shape = loader.dataset.image_shape
            
        self.model.to(self.device)
        
        y_pred = []
        for x, y, launch_date in tqdm(loader, leave=False):

            x = x.squeeze(0).to(self.device)

            if climatology is not None:
                concat_layers = self.get_climatology_array(climatology, launch_date)
            else:
                concat_layers = None

            with torch.no_grad():
                with amp.autocast():
                    y_hat, y_hat_mappings = self.model(
                        x,
                        concat_layers=concat_layers, 
                        teacher_forcing_ratio=0,
                        mask=mask, 
                        high_interest_region=high_interest_region, 
                        graph_structure=graph_structure
                        )
                
                y_hat = [unflatten(y_hat[i], y_hat_mappings[i], image_shape, mask).detach().cpu() for i in range(self.output_timesteps)]
                
                y_hat = np.stack(y_hat)#.squeeze(1)
                
                y_pred.append(y_hat)

                del y_hat_mappings
                torch.cuda.empty_cache()
            
        return np.stack(y_pred, 0)

    def score(self, x, y, rollout=None):
        pass
