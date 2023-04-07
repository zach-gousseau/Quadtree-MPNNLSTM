import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import datetime
import os
import time
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta

import argparse

from utils import normalize

from mpnnlstm import NextFramePredictorS2S
from seq2seq import Seq2Seq

from torch.utils.data import Dataset, DataLoader

# torch.autograd.set_detect_anomaly(True)

class IceDataset(Dataset):
    def __init__(self, ds, years, month, input_timesteps, output_timesteps, x_vars=None, y_vars=None, train=False):
        self.train = train
        
        self.x, self.y, self.launch_dates = self.get_xy(ds, years, month, input_timesteps, output_timesteps, x_vars=x_vars, y_vars=y_vars)
        self.image_shape = self.x[0].shape[1:-1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.launch_dates[idx]

    def get_xy(self, ds, years, month, input_timesteps, output_timesteps, x_vars=None, y_vars=None):

        x, y = [], []
        launch_dates = []
        for year in years:
            
            x_vars = list(ds.data_vars) if x_vars is None else x_vars
            y_vars = list(ds.data_vars) if y_vars is None else y_vars
            
            if self.train:
                # 3 months around the month of interest
                start_date = datetime.datetime(year, month, 1) - relativedelta(months=1)
                end_date = datetime.datetime(year, month, 1) + relativedelta(months=2)
            else:
                start_date = datetime.datetime(year, month, 1)
                end_date = datetime.datetime(year, month, 1) + relativedelta(months=1)
                

            # Add buffer for input timesteps and output timesteps 
            start_date -= relativedelta(days=input_timesteps)
            end_date += relativedelta(days=output_timesteps-1)

            # Slice dataset & normalize
            ds_year = ds.sel(time=slice(start_date, end_date))
            
            # Add DOY
            ds_year['doy'] = (('time', 'latitude', 'longitude'), ds_year.time.dt.dayofyear.values.reshape(-1, 1, 1) * np.ones(shape=(ds_year[x_vars[0]].shape)))
            
            ds_year = (ds_year - ds_year.min()) / (ds_year.max() - ds_year.min())

            num_samples = ds_year.time.size - output_timesteps - input_timesteps

            i = 0
            x_year = np.ndarray((num_samples, input_timesteps, ds.latitude.size, ds.longitude.size, len(x_vars)))
            y_year = np.ndarray((num_samples, output_timesteps, ds.latitude.size, ds.longitude.size, len(y_vars)))
            while i + output_timesteps + input_timesteps < ds_year.time.size:
                x_year[i] = np.moveaxis(np.nan_to_num(ds_year[x_vars].isel(time=slice(i, i+input_timesteps)).to_array().to_numpy()), 0, -1)
                y_year[i] = np.moveaxis(np.nan_to_num(ds_year[y_vars].isel(time=slice(i+input_timesteps, i+input_timesteps+output_timesteps)).to_array().to_numpy()), 0, -1)
                i += 1

            x.append(x_year)
            y.append(y_year)
            launch_dates.append(ds_year.time[input_timesteps:-output_timesteps].values)

        x, y, launch_dates = np.concatenate(x, 0), np.concatenate(y, 0), np.concatenate(launch_dates, 0)
        
        return x.astype('float32'), y.astype('float32'), launch_dates.astype(int)


if __name__ == '__main__':

    month = 4

    ds = xr.open_zarr('data/era5_hb_daily.zarr').load()    # ln -s /home/zgoussea/scratch/era5_hb_daily.zarr data/era5_hb_daily.zarr

    coarsen = 0

    if coarsen != 0:
        ds = ds.coarsen(latitude=coarsen, longitude=coarsen, boundary='trim').mean()

    mask = np.isnan(ds.siconc.isel(time=0)).values

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # Number of frames to read as input
    input_timesteps = 5
    output_timesteps= 30

    start = time.time()

    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf']
    y_vars = ['siconc']  # ['siconc', 't2m']
    training_years = range(2011, 2012)

    climatology = ds[y_vars].groupby('time.month').mean('time', skipna=True).to_array().values
    climatology = np.nan_to_num(climatology)

    input_features = len(x_vars)
    
    data_train = IceDataset(ds, training_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True)
    data_test = IceDataset(ds, [training_years[-1]+1], month, input_timesteps, output_timesteps, x_vars, y_vars)

    loader_profile = DataLoader(data_train, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(range(10)))
    loader_test = DataLoader(data_train, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(range(5)))

    thresh = 0.15
    # thresh = -np.inf

    def dist_from_05(arr):
        return abs(abs(arr - 0.5) - 0.5)

    # Add 3 to the number of input features since weadd positional encoding (x, y) and node size (s)
    model_kwargs = dict(
        hidden_size=64,
        dropout=0.1,
        n_layers=3,
        transform_func=dist_from_05
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('mps')
    print('device:', device)

    experiment_name = f'M{str(month)}_Y{training_years[0]}_Y{training_years[-1]}_I{input_timesteps}O{output_timesteps}'

    model = NextFramePredictorS2S(
        thresh=thresh,
        experiment_name=experiment_name,
        input_features=input_features,
        output_timesteps=output_timesteps,
        transform_func=dist_from_05,
        device=device,
        model_kwargs=model_kwargs)

    lr = 0.01

    model.model.train()

    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    model.train(loader_profile, loader_test, climatology, lr=lr, n_epochs=1, mask=mask)  # Train for 20 epochs
    pr.disable()
    stats = pstats.Stats(pr).sort_stats('time')
    stats.print_stats(10)


"""
GPU
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        5   59.771   11.954   59.771   11.954 {method 'run_backward' of 'torch._C._EngineBase' objects}
     1210   35.624    0.029   37.355    0.031 /Users/zach/Documents/Quadtree-MPNNLSTM/graph_functions.py:263(flatten)
     7920   21.319    0.003   26.378    0.003 /Users/zach/opt/miniconda3/envs/thesis/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:103(__call__)
847906/846976   20.496    0.000   20.546    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
      310   11.740    0.038   13.361    0.043 /Users/zach/Documents/Quadtree-MPNNLSTM/graph_functions.py:60(quadtree_decompose)
      630   10.514    0.017   10.514    0.017 {method 'cpu' of 'torch._C._TensorBase' objects}
     4050    8.422    0.002    8.422    0.002 {method 'to' of 'torch._C._TensorBase' objects}
      310    6.060    0.020   45.341    0.146 /Users/zach/Documents/Quadtree-MPNNLSTM/graph_functions.py:200(create_graph_structure)
    15840    5.181    0.000    5.181    0.000 {method 'scatter_add_' of 'torch._C._TensorBase' objects}
     7920    4.758    0.001    8.576    0.001 /Users/zach/opt/miniconda3/envs/thesis/lib/python3.10/site-packages/torch_geometric/nn/conv/gcn_conv.py:33(gcn_norm)

NOT GPU
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        5   21.630    4.326   21.630    4.326 {method 'run_backward' of 'torch._C._EngineBase' objects}
847906/846976   18.110    0.000   18.146    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
     1210   16.517    0.014   16.724    0.014 /Users/zach/Documents/Quadtree-MPNNLSTM/graph_functions.py:263(flatten)
     1200   11.134    0.009   11.161    0.009 /Users/zach/Documents/Quadtree-MPNNLSTM/graph_functions.py:289(unflatten)
      310    9.973    0.032   11.382    0.037 /Users/zach/Documents/Quadtree-MPNNLSTM/graph_functions.py:60(quadtree_decompose)
      310    5.377    0.017   33.108    0.107 /Users/zach/Documents/Quadtree-MPNNLSTM/graph_functions.py:200(create_graph_structure)
    15840    5.037    0.000    5.037    0.000 {method 'scatter_add_' of 'torch._C._TensorBase' objects}
     7920    3.357    0.000   17.175    0.002 /Users/zach/opt/miniconda3/envs/thesis/lib/python3.10/site-packages/torch_geometric/nn/conv/gcn_conv.py:168(forward)
     7920    2.656    0.000    2.656    0.000 {method 'index_select' of 'torch._C._TensorBase' objects}
  3382930    2.459    0.000    2.459    0.000 {method 'reduce' of 'numpy.ufunc' objects}
"""