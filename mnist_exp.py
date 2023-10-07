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

from model.utils import normalize, int_to_datetime

from model.mpnnlstm import NextFramePredictorS2S
from model.seq2seq import Seq2Seq

from data.mod_moving_mnist import ModMovingMNISTDataset
from torch.utils.data import Dataset, DataLoader

from model.graph_functions import create_static_heterogeneous_graph, create_static_homogeneous_graph, flatten, unflatten

np.random.seed(21)
random.seed(21)
torch.manual_seed(21)

input_timesteps = 10
output_timesteps = 10
canvas_size = (32, 32)
digit_size = (12, 12)
velocity_noise = 0.25
pixel_noise = 0.05
gap = 0
n_digits=1

dataset_train = ModMovingMNISTDataset(
                 300,
                 input_timesteps,
                 output_timesteps,
                 n_digits=n_digits,
                 gap=gap,
                 canvas_size=canvas_size,
                 digit_size=digit_size,
                 pixel_noise=pixel_noise,
                 velocity_noise=velocity_noise,
                 as_torch=False
)

dataset_test = ModMovingMNISTDataset(
                 100,
                 input_timesteps,
                 output_timesteps,
                 n_digits=n_digits,
                 gap=gap,
                 canvas_size=canvas_size,
                 digit_size=digit_size,
                 pixel_noise=pixel_noise,
                 velocity_noise=velocity_noise,
                 as_torch=False
)

dataset_val = ModMovingMNISTDataset(
                 100,
                 input_timesteps,
                 output_timesteps,
                 n_digits=n_digits,
                 gap=gap,
                 canvas_size=canvas_size,
                 digit_size=digit_size,
                 pixel_noise=pixel_noise,
                 velocity_noise=velocity_noise,
                 as_torch=False
)


loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

convolution_type = 'TransformerConv'
lr = 0.001
multires_training = False
truncated_backprop = 0
n_conv_layers = 3

cache_dir = None

binary=False

rnn_type = 'NoConvLSTM'
n_epochs = [35]

use_edge_attrs = False if convolution_type == 'GCNConv' else True

# Set threshold 
thresh = -np.inf  # 0.15

# Note: irrelevant if thresh = -np.inf
def dist_from_05(arr):
    return abs(abs(arr - 0.5) - 0.5)


# Arguments passed to Seq2Seq constructor
model_kwargs = dict(
    hidden_size=8,
    dropout=0.1,
    n_layers=1,
    transform_func=dist_from_05,
    dummy=False,
    n_conv_layers=n_conv_layers,
    rnn_type=rnn_type,
    convolution_type=convolution_type,
    concat_layers_dim=1,
)

experiment_name = f'test_mnist'

model = NextFramePredictorS2S(
    thresh=thresh,
    experiment_name=experiment_name,
    directory='scratch/test',
    input_features=1,
    input_timesteps=input_timesteps,
    output_timesteps=output_timesteps,
    transform_func=dist_from_05,
    device=device,
    binary=binary,
    debug=False, 
    model_kwargs=model_kwargs)

print('Num. parameters:', model.get_n_params())
# print('Model:\n', model.model)

model.model.train()

# Train with full resolution. Use high interest region.
model.train(
    loader_train,
    loader_test,
    lr=lr,
    n_epochs=100,
    mask=None,
    truncated_backprop=truncated_backprop,
    graph_structure=None,
    )