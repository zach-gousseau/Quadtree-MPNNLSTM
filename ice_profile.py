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

def remove_lone_pixels(arr):
    # Create a padded version of the arr with False values around the edges
    padded_arr = np.pad(~arr, ((1, 1), (1, 1)), mode='constant', constant_values=False)
    # Create a mask that is True for any element that is True and has False neighbors
    mask = ~arr & ~padded_arr[:-2,:-2] & ~padded_arr[:-2,1:-1] & ~padded_arr[:-2,2:] & \
           ~padded_arr[1:-1,:-2] & ~padded_arr[1:-1,2:] & \
           ~padded_arr[2:,:-2] & ~padded_arr[2:,1:-1] & ~padded_arr[2:,2:]
    # Return the indices of the True elements in the mask
    return arr + mask

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    # device = torch.device('mps')
    print('device:', device)

    month = 6
    convolution_type = 'TransformerConv'
    convolution_type = 'GCNConv'
    # convolution_type = 'Dummy'
    generate_predictions = False

    ds = xr.open_zarr('data/era5_hb_daily.zarr')    # ln -s /home/zgoussea/scratch/era5_hb_daily.zarr data/era5_hb_daily.zarr
    # ds = xr.open_dataset('data/era5_hb_daily_coarsened_2.zarr')
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
    mask = remove_lone_pixels(mask)


    # mask = np.zeros_like(ds.siconc.isel(time=0).values).astype(bool)

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    binary = False
    binary_thresh = 0.15

    truncated_backprop = 0

    # Number of frames to read as input
    input_timesteps = 10
    output_timesteps= 10

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

    thresh = 0.15
    # thresh = -1
    thresh = -np.inf

    def dist_from_05(arr):
        return abs(abs(arr - 0.5) - 0.5)

    # Add 3 to the number of input features since weadd positional encoding (x, y) and node size (s)
    model_kwargs = dict(
        hidden_size=32,
        dropout=0.1,
        n_layers=1,
        n_conv_layers=3,
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
        debug=True,
        model_kwargs=model_kwargs)

    print('Num. parameters:', model.get_n_params())

    # print(model.model)

    lr = 0.001

    model.model.train()

    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()

    model.train(
        loader_profile,
        loader_test,
        climatology,
        lr=lr, 
        n_epochs=10, 
        mask=mask, 
        truncated_backprop=truncated_backprop
        )

    pr.disable()
    stats = pstats.Stats(pr).sort_stats('time')
    stats.print_stats(10)

    if generate_predictions:

        data_val = IceDataset(ds, [training_years[-1]+2], month, input_timesteps, output_timesteps, x_vars, y_vars)
        loader_val = DataLoader(data_val, batch_size=1, shuffle=False)

        # Generate predictions
        model.model.eval()
        val_preds = model.predict(loader_val, climatology, mask=mask)
        
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

        results_dir = f'ice_results_profile'

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        ds.to_netcdf(f'{results_dir}/valpredictions_{experiment_name}.nc')

        model.loss.to_csv(f'{results_dir}/loss_{experiment_name}.csv')
        model.save(results_dir)