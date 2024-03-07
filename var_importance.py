import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import datetime
import os
import time
import glob
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta
import copy

import argparse

from model.utils import normalize, int_to_datetime

from model.mpnnlstm import NextFramePredictorS2S
from model.seq2seq import Seq2Seq

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from ice_dataset import IceDataset

from model.graph_functions import create_static_heterogeneous_graph, create_static_homogeneous_graph, flatten, unflatten

import torch.nn as nn
class MSE_SIP(nn.Module):
    def __init__(self):
        super(MSE_SIP, self).__init__()
        
    def forward(self, output, target, mask=None, weights=None):
        if mask is not None:
            output, target = output[:, :, ~mask], target[:, :, ~mask]
        loss = (output - target) ** 2
        return np.sqrt(loss.mean())

class MSE_SIP_bin_sep(nn.Module):
    def __init__(self):
        super(MSE_SIP_bin_sep, self).__init__()
        self.mse = MSE_SIP()
        
    def forward(self, output, target, mask=None, weights=None):
        mse_loss = self.mse(output[..., [0]], target, mask, weights)
        # bce_loss = self.bce(output[..., [1]], target, mask, weights) * 0.1
        # loss = mse_loss + bce_loss
        return mse_loss


np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

cache_dir='/home/zgoussea/scratch/data_cache/'

# Defaults
convolution_type = 'GCNConv'
lr = 0.001
multires_training = False
truncated_backprop = 0

training_years_1 = training_years_4 = range(1993, 2014)

month = 6

x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf', 'usi', 'vsi', 'sithick', 'thetao', 'so']
y_vars = ['siconc']
input_features = len(x_vars)
input_timesteps = 10
output_timesteps= 90
preset_mesh = False
rnn_type = 'LSTM'
n_conv_layers = 3
# n_conv_layers = 6

binary=False

convolution_type = 'TransformerConv'
# convolution_type = 'GCNConv'
# convolution_type = 'GINEConv'
multires_training = False
preset_mesh = 'heterogeneous'
# preset_mesh = 'homogeneous'
rnn_type = 'NoConvLSTM'
directory = f'/home/zgoussea/scratch/results/ice_results_20years_glorys_3conv_noconv_20yearsstraight_splitgconvlstm_adam_nodecay_lr001_1decoders_transformer_multitask'

directory = f'/home/zgoussea/scratch/results/ice_results_20years_glorys_3conv_noconv_20yearsstraight_splitgconvlstm_adam_nodecay_lr001_1decoders_gine'
directory = f'/home/zgoussea/scratch/results/ice_results_20years_glorys_3conv_noconv_20yearsstraight_splitgconvlstm_adam_nodecay_lr001_1decoders_transformer_homo_multitask'
directory = f'results/transformer'

results_dir = '/home/zgoussea/scratch/results/'
directory = f'{results_dir}/transformer_0'
# directory = f'results/gcn'

use_edge_attrs = False if convolution_type == 'GCNConv' else True

# -------------------------------------------

# Full resolution dataset
ds = xr.open_mfdataset(glob.glob('data/ERA5_GLORYS/*.nc'))  # ln -s /home/zgoussea/scratch/ERA5_GLORYS data/ERA5_GLORYS
mask = np.isnan(xr.open_dataset('data/ERA5_GLORYS/ERA5_GLORYS_1993.nc').siconc.isel(time=0)).values
high_interest_region = None

image_shape = mask.shape

if preset_mesh == 'heterogeneous':
    graph_structure = create_static_heterogeneous_graph(image_shape, 4, mask, high_interest_region=high_interest_region, use_edge_attrs=use_edge_attrs, resolution=1/12, device=device)
elif preset_mesh == 'homogeneous':
    graph_structure = create_static_homogeneous_graph(image_shape, 4, mask, high_interest_region=high_interest_region, use_edge_attrs=use_edge_attrs, resolution=1/12, device=device)

# Set threshold 
thresh = -np.inf  # 0.15
print(f'Threshold is {thresh}')

# Note: irrelevant if thresh = -np.inf
def dist_from_05(arr):
    return abs(abs(arr - 0.5) - 0.5)


# Arguments passed to Seq2Seq constructor
model_kwargs = dict(
    hidden_size=32,
    dropout=0.1,
    n_layers=1,
    transform_func=dist_from_05,
    dummy=False,
    n_conv_layers=n_conv_layers,
    rnn_type=rnn_type,
    convolution_type=convolution_type,
    multitask=True,
)

experiment_name = f'M{str(month)}_Y{training_years_1[0]}_Y{training_years_4[-1]}_I{input_timesteps}O{output_timesteps}'

model = NextFramePredictorS2S(
    thresh=thresh,
    experiment_name=experiment_name,
    directory=directory,
    input_features=input_features,
    input_timesteps=input_timesteps,
    output_timesteps=output_timesteps,
    transform_func=dist_from_05,
    device=device,
    binary=binary,
    debug=False, 
    model_kwargs=model_kwargs)

model.load(directory)

climatology_grid = ds[y_vars].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).to_array().values
climatology_grid = torch.tensor(np.nan_to_num(climatology_grid)).to(device)
climatology_grid = torch.moveaxis(climatology_grid, 0, -1)
# climatology_grid = torch.load('data/climatology.pt').to(device)
climatology = flatten(climatology_grid, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
climatology = torch.moveaxis(climatology, -1, 0)

cache_dir='/home/zgoussea/scratch/data_cache/'
data_val = IceDataset(ds, range(2014, 2020), month, input_timesteps, output_timesteps, x_vars, y_vars, graph_structure=graph_structure, mask=mask, cache_dir=cache_dir, flatten_y=False)
data_val.y = unflatten(torch.Tensor(data_val.y).to(device), graph_structure['mapping'], image_shape)

var_importance = {}
var_preds = {}

# Base
loader_val = DataLoader(data_val, batch_size=1, shuffle=False)
model.model = model.model.eval()
val_preds = model.predict(
    loader_val,
    climatology,
    mask=mask,
    high_interest_region=high_interest_region,
    graph_structure=graph_structure
    )
loss_func = MSE_SIP_bin_sep()
loss = loss_func(val_preds, data_val.y.cpu().numpy())
print('base', loss)
var_importance['base'] = loss
var_preds['base'] = val_preds

# With noise variables
for i in range(len(x_vars)):
    data_val_var = copy.deepcopy(data_val)
    data_val_var.x = torch.Tensor(data_val_var.x)
    var = data_val_var.x[..., i]
    noise = torch.randn(*var.shape) * var.std() + var.mean()
    if x_vars[i] != 'siconc':
        data_val_var.x[..., i] = noise
    else:
        data_val_var.x[:, :-1, :, i] = noise[:, :-1, :]
    loader_val = DataLoader(data_val_var, batch_size=1, shuffle=False)
    val_preds = model.predict(
        loader_val,
        climatology,
        mask=mask,
        high_interest_region=high_interest_region,
        graph_structure=graph_structure
        )
    loss_func = MSE_SIP_bin_sep()
    loss = loss_func(val_preds, data_val.y.cpu().numpy())
    print(x_vars[i], loss)
    var_importance[x_vars[i]] = loss
    var_preds[x_vars[i]] = val_preds

var_importance_df = pd.DataFrame(index=range(1, 91))
for var_ in x_vars + ['base']:
    val_preds = var_preds[var_]
    loss = [loss_func(val_preds[:, [t]], data_val.y[:, [t]].cpu().numpy()) for t in range(output_timesteps)]
    var_importance_df[var_] = loss

var_importance_df.to_csv(f'scratch/var_importance_{month}.csv')
import seaborn as sns
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(var_importance_df.T, ax=ax)
plt.savefig(f'scratch/var_importance_{month}.png')

fig, ax = plt.subplots(figsize=(8, 4))
var_importance_diff = var_importance_df.sub(var_importance_df['base'], axis=0)
sns.heatmap(var_importance_diff.T, ax=ax, cmap='coolwarm', center=0)
plt.savefig(f'scratch/var_importance_diff_{month}.png')