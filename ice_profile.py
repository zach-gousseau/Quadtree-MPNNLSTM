import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import datetime
import glob
import pickle
import os
import time
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta

import argparse

from model.utils import normalize
from model.mpnnlstm import NextFramePredictorS2S
from model.seq2seq import Seq2Seq

from ice_dataset import IceDataset
from torch.utils.data import Dataset, DataLoader

from model.graph_functions import create_static_heterogeneous_graph, create_static_homogeneous_graph

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    # device = torch.device('mps')
    print('device:', device)

    month = 6
    convolution_type = 'TransformerConv'
    # convolution_type = 'GCNConv'
    # convolution_type = 'Dummy'
    generate_predictions = False

    ds = xr.open_mfdataset(glob.glob('data/ERA5_GLORYS/*.nc'))  # ln -s /home/zgoussea/scratch/ERA5_GLORYS data/ERA5_GLORYS

    # ds = ds.isel(latitude=slice(175, 275), longitude=slice(125, 225))

    coarsen = 1

    if coarsen > 1:
        ds = ds.coarsen(latitude=coarsen, longitude=coarsen, boundary='trim').mean()
    elif coarsen < 1:
        newres = 0.25 * coarsen
        newlat = np.arange(ds.latitude.min(), ds.latitude.max() + newres, newres)
        newlon = np.arange(ds.longitude.min(), ds.longitude.max() + newres, newres)
        ds = ds.interp(latitude=newlat, longitude=newlon, method='nearest')

    mask = np.isnan(ds.siconc.isel(time=0)).values
    # high_interest_region = xr.open_dataset('data/shipping_corridors/primary_route_mask.nc').band_data.values
    high_interest_region = None

    image_shape = mask.shape
    # graph_structure = create_static_heterogeneous_graph(image_shape, 4, mask, high_interest_region, use_edge_attrs=True, resolution=0.25, device=device)
    graph_structure = create_static_homogeneous_graph(image_shape, 4, mask, use_edge_attrs=True, resolution=0.25, device=device)

    print(f'Num nodes: {len(graph_structure["graph_nodes"])}')


    # graph_structure = None

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    binary = False
    binary_thresh = 0.15

    truncated_backprop = 0

    # Number of frames to read as input
    input_timesteps = 3
    output_timesteps= 45

    start = time.time()

    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf']
    y_vars = ['siconc']  # ['siconc', 't2m']
    training_years = range(2001, 2002)
    
    cache_dir='/home/zgoussea/scratch/data_cache/'

    climatology = ds[y_vars].groupby('time.dayofyear').mean('time', skipna=True).to_array().values
    climatology = torch.tensor(np.nan_to_num(climatology)).to(device)

    input_features = len(x_vars)
    
    data_train = IceDataset(ds, training_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True, y_binary_thresh=binary_thresh if binary else None, cache_dir=cache_dir)
    data_test = IceDataset(ds, [training_years[-1]+1], month, input_timesteps, output_timesteps, x_vars, y_vars, y_binary_thresh=binary_thresh if binary else None, cache_dir=cache_dir)

    loader_profile = DataLoader(data_train, batch_size=1)#, sampler=torch.utils.data.SubsetRandomSampler(range(15)))
    loader_test = DataLoader(data_test, batch_size=1)#, sampler=torch.utils.data.SubsetRandomSampler(range(5)))

    thresh = 0.15
    thresh = -np.inf

    def dist_from_05(arr):
        return abs(abs(arr - 0.5) - 0.5)

    model_kwargs = dict(
        hidden_size=16,
        dropout=0.1,
        n_layers=1,
        n_conv_layers=1,
        transform_func=dist_from_05,
        dummy=False,
        convolution_type=convolution_type,
        rnn_type='LSTM',
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
        debug=False,
        model_kwargs=model_kwargs)

    print('Num. parameters:', model.get_n_params())

    # print(model.model)

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
        high_interest_region=high_interest_region, 
        graph_structure=graph_structure, 
        truncated_backprop=False,#truncated_backprop
        )

    pr.disable()
    stats = pstats.Stats(pr).sort_stats('time')
    stats.print_stats(10)

    if generate_predictions:

        data_val = IceDataset(ds, [training_years[-1]+2], month, input_timesteps, output_timesteps, x_vars, y_vars, cache_dir=cache_dir)
        loader_val = DataLoader(data_val, batch_size=1, shuffle=False)

        # Generate predictions
        model.model.eval()
        val_preds = model.predict(
            loader_val, 
            climatology, 
            mask=mask, 
            high_interest_region=high_interest_region, 
            graph_structure=graph_structure
            )
        
        # Save results
        launch_dates = loader_val.dataset.launch_dates
        
        ds = xr.Dataset(
            data_vars=dict(
                y_hat=(["launch_date", "timestep", "latitude", "longitude"], val_preds.squeeze(-1)),
                y_true=(["launch_date", "timestep", "latitude", "longitude"], loader_val.dataset.y.squeeze(-1)),
            ),
            coords=dict(
                longitude=ds.longitude,
                latitude=ds.latitude,
                launch_date=launch_dates,
                timestep=np.arange(1, output_timesteps+1),
            ),
        )

        results_dir = f'results/ice_results_profile'

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        ds.to_netcdf(f'{results_dir}/valpredictions_{experiment_name}.nc')

        model.loss.to_csv(f'{results_dir}/loss_{experiment_name}.csv')
        model.save(results_dir)