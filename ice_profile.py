import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import datetime
import glob
import os
import time
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta

import argparse

from utils import normalize

from mpnnlstm import NextFramePredictorS2S
from seq2seq import Seq2Seq
from ice_test import IceDataset

from torch.utils.data import Dataset, DataLoader

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('mps')
    print('device:', device)

    month = 6
    convolution_type = 'TransformerConv'

    ds = xr.open_zarr('data/era5_hb_daily.zarr')    # ln -s /home/zgoussea/scratch/era5_hb_daily.zarr data/era5_hb_daily.zarr
    # ds = xr.open_mfdataset(glob.glob('data/era5_hb_daily_nc/*.nc'))  # ln -s /home/zgoussea/scratch/era5_hb_daily_nc data/era5_hb_daily_nc
    # ds = xr.open_zarr('/home/zgoussea/scratch/era5_arctic_daily.zarr')
    # ds = xr.open_mfdataset(glob.glob('/home/zgoussea/scratch/ERA5/*/*.nc'))

    coarsen = 1

    if coarsen > 1:
        ds = ds.coarsen(latitude=coarsen, longitude=coarsen, boundary='trim').mean()
    elif coarsen < 1:
        newres = 0.25 * coarsen
        newlat = np.arange(ds.latitude.min(), ds.latitude.max() + newres, newres)
        newlon = np.arange(ds.longitude.min(), ds.longitude.max() + newres, newres)
        ds = ds.interp(latitude=newlat, longitude=newlon, method='nearest')

    mask = np.isnan(ds.siconc.isel(time=0)).values

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    binary = False
    binary_thresh = 0.15

    # Number of frames to read as input
    input_timesteps = 10
    output_timesteps= 90

    start = time.time()

    x_vars = ['siconc', 't2m', 'v10', 'u10']#, 'slhf']
    y_vars = ['siconc']  # ['siconc', 't2m']
    training_years = range(2015, 2016)

    climatology = ds[y_vars].groupby('time.dayofyear').mean('time', skipna=True).to_array().values
    climatology = torch.tensor(np.nan_to_num(climatology)).to(device)

    input_features = len(x_vars)
    
    data_train = IceDataset(ds, training_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True, y_binary_thresh=binary_thresh if binary else None)
    data_test = IceDataset(ds, [training_years[-1]+1], month, input_timesteps, output_timesteps, x_vars, y_vars, y_binary_thresh=binary_thresh if binary else None)

    loader_profile = DataLoader(data_train, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(range(25)))
    loader_test = DataLoader(data_train, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(range(5)))

    # thresh = 0.15
    thresh = -np.inf

    def dist_from_05(arr):
        return abs(abs(arr - 0.5) - 0.5)

    # Add 3 to the number of input features since weadd positional encoding (x, y) and node size (s)
    model_kwargs = dict(
        hidden_size=32,
        dropout=0.1,
        n_layers=1,
        transform_func=dist_from_05,
        dummy=False,
        convolution_type=convolution_type,
        debug=False,
    )

    experiment_name = f'M{str(month)}_Y{training_years[0]}_Y{training_years[-1]}_I{input_timesteps}O{output_timesteps}'

    model = NextFramePredictorS2S(
        thresh=thresh,
        experiment_name=experiment_name,
        input_features=input_features,
        input_timesteps=input_timesteps,
        output_timesteps=output_timesteps,
        transform_func=dist_from_05,
        device=device,
        model_kwargs=model_kwargs)

    print('Num. parameters:', model.get_n_params())
    print(model.model)

    lr = 0.01

    model.model.train()

    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()

    model.train(
        loader_profile,
        loader_test,
        climatology,
        lr=lr, 
        n_epochs=1, 
        mask=mask, 
        truncated_backprop=truncated_backprop
        )

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