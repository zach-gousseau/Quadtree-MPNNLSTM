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

import argparse

from model.utils import normalize, int_to_datetime

from model.mpnnlstm import NextFramePredictorS2S
from model.seq2seq import Seq2Seq

from torch.utils.data import Dataset, DataLoader

from ice_dataset import IceDataset

from model.graph_functions import create_static_heterogeneous_graph, create_static_homogeneous_graph


if __name__ == '__main__':

    np.random.seed(21)
    random.seed(21)
    torch.manual_seed(21)

    start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--month')  # Month number
    parser.add_argument('-e', '--exp')

    args = vars(parser.parse_args())
    month = int(args['month'])
    exp = int(args['exp'])

    # Defaults
    convolution_type = 'TransformerConv'
    lr = 0.0001
    multires_training = False
    truncated_backprop = 0

    training_years = range(2007, 2013)
    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf']
    y_vars = ['siconc']
    input_features = len(x_vars)
    input_timesteps = 10
    output_timesteps= 90
    preset_mesh = False

    binary=False

    # Experiment definitions --------------------
    if exp == 1:
        convolution_type = 'GCNConv'
    elif exp == 2:
        lr = 0.001
    elif exp == 3:
        multires_training = True
    elif exp == 4:
        lr = 0.0001
    elif exp == 5:
        truncated_backprop = 45
    elif exp == 6:
        truncated_backprop = 30
    elif exp == 7:
        lr = 0.001
        input_timesteps = 30
    elif exp == 8:
        lr = 0.001
        input_timesteps = 90
    elif exp == 9:
        multires_training = True
        preset_mesh = 'heterogeneous'
    elif exp == 10:
        multires_training = True
        preset_mesh = 'homogeneous'
        
    # -------------------------------------------

    if multires_training:

        # Half resolution dataset
        # ds_half = xr.open_dataset('data/era5_hb_daily_coarsened_2.zarr') # ln -s /home/zgoussea/scratch/era5_hb_daily_coarsened_2.zarr data/era5_hb_daily_coarsened_2.zarr
        ds_half = xr.open_mfdataset(glob.glob('data/hb_era5_glorys_nc_2x/*.nc'))
        mask_half = np.isnan(ds_half.siconc.isel(time=0)).values

        data_train_half = IceDataset(ds_half, training_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True)
        data_test_half = IceDataset(ds_half, [training_years[-1]+1], month, input_timesteps, output_timesteps, x_vars, y_vars)
        data_val_half = IceDataset(ds_half, [training_years[-1]+5], month, input_timesteps, output_timesteps, x_vars, y_vars)

        loader_train_half = DataLoader(data_train_half, batch_size=1, shuffle=True)
        loader_test_half = DataLoader(data_test_half, batch_size=1, shuffle=True)
        loader_val_half = DataLoader(data_val_half, batch_size=1, shuffle=False)

        # climatology_half = ds_half[y_vars].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).to_array().values
        # climatology_half = torch.tensor(np.nan_to_num(climatology_half)).to(device)

        if preset_mesh == 'heterogeneous':
            graph_structure_half = create_static_heterogeneous_graph(mask_half.shape, 4, mask_half, use_edge_attrs=True, resolution=1/6, device=device)
        elif preset_mesh == 'homogeneous':
            graph_structure_half = create_static_homogeneous_graph(mask_half.shape, 4, mask_half, use_edge_attrs=True, resolution=1/6, device=device)


    # Full resolution dataset
    # ds = xr.open_zarr('data/era5_hb_daily.zarr')    # ln -s /home/zgoussea/scratch/era5_hb_daily.zarr data/era5_hb_daily.zarr
    # ds = xr.open_mfdataset(glob.glob('data/era5_hb_daily_nc/*.nc'))  # ln -s /home/zgoussea/scratch/era5_hb_daily_nc data/era5_hb_daily_nc
    # ds = xr.open_mfdataset(glob.glob('data/hb_era5_glorys_nc/*.nc'))  # ln -s /home/zgoussea/scratch/hb_era5_glorys_nc data/hb_era5_glorys_nc
    # ds = xr.open_zarr('data/hb_era5_glorys.zarr')  # ln -s /home/zgoussea/scratch/hb_era5_glorys.zarr/  data/hb_era5_glorys.zarr
    ds = xr.open_mfdataset(glob.glob('data/hb_era5_glorys_nc/*.nc'))
    mask = np.isnan(ds.siconc.isel(time=0)).values
    high_interest_region = xr.open_dataset('data/shipping_corridors/primary_route_mask.nc').band_data.values

    image_shape = mask.shape
    graph_structure = None

    if preset_mesh == 'heterogeneous':
        graph_structure = create_static_heterogeneous_graph(image_shape, 4, mask, use_edge_attrs=True, resolution=1/12, device=device)
    elif preset_mesh == 'homogeneous':
        graph_structure = create_static_homogeneous_graph(image_shape, 4, mask, use_edge_attrs=True, resolution=1/12, device=device)
    
    # Full resolution datasets
    data_train = IceDataset(ds, training_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True)
    data_test = IceDataset(ds, [training_years[-1]+1], month, input_timesteps, output_timesteps, x_vars, y_vars)
    data_val = IceDataset(ds, range(training_years[-1]+2, training_years[-1]+2+4), month, input_timesteps, output_timesteps, x_vars, y_vars)

    loader_train = DataLoader(data_train, batch_size=1, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=1, shuffle=True)
    loader_val = DataLoader(data_val, batch_size=1, shuffle=False)

    climatology = ds[y_vars].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).to_array().values
    climatology = torch.tensor(np.nan_to_num(climatology)).to(device)

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
        n_conv_layers=3,
        rnn_type='LSTM',
        convolution_type=convolution_type,
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
        binary=binary,
        debug=True, 
        model_kwargs=model_kwargs)

    print('Num. parameters:', model.get_n_params())
    print('Model:\n', model.model)

    model.model.train()

    # Train with half resolution first. Don't use the high interest region
    if multires_training:
        model.train(
            loader_train_half,
            loader_test_half,
            # climatology_half,
            lr=lr,
            n_epochs=5,
            mask=mask_half,
            truncated_backprop=truncated_backprop,
            graph_structure=graph_structure_half) 

    # Train with full resolution. Use high interest region.
    model.train(
        loader_train,
        loader_test,
        climatology,
        lr=lr,
        n_epochs=15 if not multires_training else 10,
        mask=mask,
        high_interest_region=high_interest_region,
        truncated_backprop=truncated_backprop,
        graph_structure=graph_structure,
        ) 

    # Save model and losses
    results_dir = f'ice_results_jun3_{exp}_multires_noclim'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model.loss.to_csv(f'{results_dir}/loss_{experiment_name}.csv')
    model.save(results_dir)
    
    # Generate predictions
    model.model.eval()
    val_preds = model.predict(
        loader_val,
        climatology,
        mask=mask,
        graph_structure=graph_structure
        )
    
    # Save results
    launch_dates = [int_to_datetime(t) for t in loader_val.dataset.launch_dates]
    
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
    ds.to_netcdf(f'{results_dir}/valpredictions_{experiment_name}.nc')
    print(f'Finished model {month} in {((time.time() - start) / 60)} minutes')
