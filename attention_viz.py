import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import datetime
import os
import tqdm
import time
import glob
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta
from matplotlib.colors import LogNorm

os.chdir('/Users/zach/Documents/Quadtree-MPNNLSTM')

from model.graph_functions import unflatten
from model.utils import normalize, int_to_datetime
from model.mpnnlstm import NextFramePredictorS2S
from model.seq2seq import Seq2Seq
from model.graph_functions import create_static_heterogeneous_graph, create_static_homogeneous_graph

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from ice_dataset import IceDataset

device = torch.device('cpu')

# Defaults
convolution_type = 'TransformerConv'

training_years = range(2010, 2011)
x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf']
y_vars = ['siconc']
input_features = len(x_vars)
# input_timesteps = 10
# output_timesteps= 90
input_timesteps = 3
output_timesteps= 45

binary=False

# Set threshold 
thresh = -np.inf

# Arguments passed to Seq2Seq constructor
# model_kwargs = dict(
#     hidden_size=32,
#     dropout=0.1,
#     n_layers=1,
#     dummy=False,
#     n_conv_layers=3,
#     rnn_type='LSTM',
#     convolution_type=convolution_type,
# )

model_kwargs = dict(
    hidden_size=16,
    dropout=0.1,
    n_layers=1,
    n_conv_layers=2,
    dummy=False,
    convolution_type=convolution_type,
    rnn_type='LSTM',
)

month = 6
ds = xr.open_mfdataset(glob.glob('/Users/zach/Documents/Quadtree-MPNNLSTM/data/ERA5_GLORYS/*.nc'))
mask = np.isnan(ds.siconc.isel(time=0)).values

climatology = ds[y_vars].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).to_array().values
climatology = torch.tensor(np.nan_to_num(climatology)).to(device)

graph_structure = create_static_heterogeneous_graph(mask.shape, 4, mask, use_edge_attrs=True, resolution=0.25, device=device)
# graph_structure = create_static_homogeneous_graph(mask.shape, 4, mask, use_edge_attrs=True, resolution=0.25, device=device)

data_val = IceDataset(ds, [2015], month, input_timesteps, output_timesteps, x_vars, y_vars)

experiment_name = f'M{str(month)}_Y{training_years[0]}_Y{training_years[-1]}_I{input_timesteps}O{output_timesteps}'

model = NextFramePredictorS2S(
    thresh=thresh,
    experiment_name=experiment_name,
    input_features=input_features,
    input_timesteps=input_timesteps,
    output_timesteps=output_timesteps,
    device=device,
    binary=binary,
    debug=False, 
    model_kwargs=model_kwargs)

results_dir = f'results/ice_results_profile/'

model.load(results_dir)

class Sampler(SubsetRandomSampler):
    def __init__(self, indices):
        super().__init__(indices[:1])
        
xs = []
att_maps_to = []
att_maps_from = []
for i in tqdm.tqdm(range(10)):
    loader_val = DataLoader(data_val, batch_size=1, shuffle=False, sampler=Sampler([i]))
    model.predict(loader_val, climatology, mask=mask, graph_structure=graph_structure)
    with open(f'scratch/attention_maps_0.npy', 'rb') as f:
        alpha = np.load(f)
        x = np.load(f)
        att_map_from = np.load(f)
        att_map_to = np.load(f)
        
        xs.append(x)
        att_maps_from.append(att_map_from)
        att_maps_to.append(att_map_to)
        
tmp = xr.zeros_like(ds.isel(time=0).siconc)

i = 0
fns = []
for x, att_map_from, att_map_to in zip(xs, att_maps_from, att_maps_to):
    
    att_map_from = unflatten(torch.Tensor(np.nan_to_num(att_map_from)), graph_structure['mapping'], mask.shape)
    att_map_to = unflatten(torch.Tensor(np.nan_to_num(att_map_to)), graph_structure['mapping'], mask.shape)
    x = unflatten(torch.Tensor(x), graph_structure['mapping'], mask.shape)
    
    fig, axs = plt.subplots(1, 2, figsize=(182, 3))

    tmp = xr.zeros_like(ds.isel(time=0).siconc)

    tmp.values = att_map_from[..., 0]
    tmp.where(~mask).plot(ax=axs[0], cmap='coolwarm', center=1)

    tmp.values = x[..., 0]
    tmp.where(~mask).plot(ax=axs[1])

    axs[0].axis('off')
    axs[1].axis('off')

    axs[0].set_title('Attention map')
    axs[1].set_title('Input')
        
    fn = f'scratch/att_gif/{i}.png'
    plt.savefig(fn)
    fns.append(fn)
    
    plt.show()
    
    i += 1
    
    
from PIL import Image
frames = []
for fn in fns:
    new_frame = Image.open(fn)
    frames.append(new_frame)
            
frames[0].save(f'scratch/att_gif/att_map.gif',
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=300,
            loop=0)

for fn in fns:
    os.remove(fn)