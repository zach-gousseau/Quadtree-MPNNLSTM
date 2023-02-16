import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import datetime
import os
import xarray as xr

from utils import normalize

from mod_moving_mnist import ModMovingMNIST
from mpnnlstm import NextFramePredictorAR, NextFramePredictorS2S
from model import MPNNLSTMI, MPNNLSTM
from seq2seq import Seq2Seq

ds = xr.open_zarr('data/era5_hb_daily.zarr')
mask = np.isnan(ds.siconc.isel(time=0)).values

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Number of frames to read as input
input_timesteps = 5
output_timesteps= 1

def xarray_to_x_y(ds, start_date, end_date, input_timesteps, output_timesteps, coarsen=0, x_vars=None, y_vars=None):
    ds = ds.sel(time=slice(start_date, end_date))
    ds = (ds - ds.min()) / (ds.max() - ds.min())

    if coarsen != 0:
        ds = ds.coarsen(latitude=coarsen, longitude=coarsen, boundary='trim').mean()
    num_samples = ds.time.size-output_timesteps-input_timesteps
    
    x_vars = list(ds.data_vars) if x_vars is None else x_vars
    y_vars = list(ds.data_vars) if y_vars is None else y_vars
    
    i = 0
    x = np.ndarray((num_samples, input_timesteps, ds.latitude.size, ds.longitude.size, len(x_vars)))
    y = np.ndarray((num_samples, output_timesteps, ds.latitude.size, ds.longitude.size, len(y_vars)))
    while i + output_timesteps + input_timesteps < ds.time.size:
        x[i] = np.moveaxis(np.nan_to_num(ds[x_vars].isel(time=slice(i, i+input_timesteps)).to_array().to_numpy()), 0, -1)
        y[i] = np.moveaxis(np.nan_to_num(ds[y_vars].isel(time=slice(i+input_timesteps, i+input_timesteps+output_timesteps)).to_array().to_numpy()), 0, -1)
        i += 1

    return x, y

def xarray_to_y(ds, start_date, end_date, input_timesteps, output_timesteps, coarsen=0, y_vars=None):
    ds = ds.sel(time=slice(start_date, end_date))
    ds = (ds - ds.min()) / (ds.max() - ds.min())

    if coarsen != 0:
        ds = ds.coarsen(latitude=coarsen, longitude=coarsen, boundary='trim').mean()
    num_samples = ds.time.size-output_timesteps-input_timesteps
    
    y_vars = list(ds.data_vars) if y_vars is None else y_vars
        
    i = 0
    y = np.ndarray((num_samples, output_timesteps, ds.latitude.size, ds.longitude.size, len(y_vars)))
    while i + output_timesteps + input_timesteps < ds.time.size:
        y[i] = np.moveaxis(np.nan_to_num(ds[y_vars].isel(time=slice(i+input_timesteps, i+input_timesteps+output_timesteps)).to_array().to_numpy()), 0, -1)
        i += 1

    return y

coarsen = 2
month = 7
x_vars = ['siconc', 't2m']#, 'v10', 'u10', 'sshf']
y_vars = ['siconc']  # ['siconc', 't2m']
training_years = (2010, 2016)

x, y = [], []
for year in range(training_years[0], training_years[1]):
    x_year, y_year = xarray_to_x_y(ds,
                                   datetime.datetime(year, month-1, 1),
                                   datetime.datetime(year, month+2, 1),
                                   input_timesteps,
                                   output_timesteps,
                                   coarsen=coarsen,
                                   x_vars=x_vars,
                                   y_vars=y_vars)
    x.append(x_year)
    y.append(y_year)
x = np.concatenate(x, 0)
y = np.concatenate(y, 0)

x_test, y_test = xarray_to_x_y(ds,
                               datetime.datetime(2016, month, 1),
                               datetime.datetime(2016, month+1, 1),
                               input_timesteps,
                               output_timesteps,
                               coarsen=coarsen,
                               x_vars=x_vars,
                               y_vars=y_vars)

x_val, y_val = xarray_to_x_y(ds,
                             datetime.datetime(2017, month, 1),
                             datetime.datetime(2017, month+1, 1), 
                             input_timesteps,
                             output_timesteps, 
                             coarsen=coarsen,
                             x_vars=x_vars,
                             y_vars=y_vars)

# y_viz = xarray_to_y(ds[x_vars], datetime.datetime(2017, 7, 1), datetime.datetime(2017, 10, 15), input_timesteps, output_timesteps, coarsen=coarsen)

# Add 3 to the number of input features since we add positional encoding (x, y) and node size (s)
nn = Seq2Seq(
    hidden_size=64,
    dropout=0.1,
    input_features=input_features+3,
    output_timesteps=output_timesteps,
    n_layers=3).float()


model = NextFramePredictorS2S(
    nn,
    experiment_name='test',
    decompose=True,
    input_features=input_features,
    output_timesteps=output_timesteps)

print('Num. parameters:', model.get_n_params())
print('Model:\n', model.model)

model.set_thresh(0.15)
# model.set_thresh(-np.inf)

lr = 0.01

model.model.train()
model.train(x, y, x_test, y_test, lr=lr, n_epochs=15, mask=mask)  # Train for 20 epochs

# model.model.eval()
# model.score(x_val, y_val[:, :1])  # Check the MSE on the validation set
# Unfinished