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

from utils import normalize

from mpnnlstm import NextFramePredictorS2S
from seq2seq import Seq2Seq

from torch.utils.data import Dataset, DataLoader

class IceDataset(Dataset):
    def __init__(self, ds, years, month, input_timesteps, output_timesteps, x_vars=None, y_vars=None, train=False, y_binary_thresh=None):
        self.train = train
        
        self.x, self.y, self.launch_dates = self.get_xy(ds, years, month, input_timesteps, output_timesteps, x_vars=x_vars, y_vars=y_vars, y_binary_thresh=y_binary_thresh)
        self.image_shape = self.x[0].shape[1:-1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.launch_dates[idx]

    def get_xy(self, ds, years, month, input_timesteps, output_timesteps, x_vars=None, y_vars=None, y_binary_thresh=None):

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

        if y_binary_thresh is not None:
            y = y > y_binary_thresh
        
        return x.astype('float32'), y.astype('float32'), launch_dates.astype(int)


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('mps')
    print('device:', device)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--month')  # Month number
    parser.add_argument('-c', '--conv')

    args = vars(parser.parse_args())
    month = int(args['month'])
    convolution_type = str(args['conv'])

    # Half resolution dataset
    ds0 = xr.open_dataset('data/era5_hb_daily_coarsened_2.zarr') # ln -s /home/zgoussea/scratch/era5_hb_daily_coarsen_2.zarr data/era5_hb_daily_coarsen_2.zarr

    # Full resolution dataset
    ds = xr.open_zarr('data/era5_hb_daily.zarr')    # ln -s /home/zgoussea/scratch/era5_hb_daily.zarr data/era5_hb_daily.zarr
    # ds = xr.open_mfdataset(glob.glob('data/era5_hb_daily_nc/*.nc'))  # ln -s /home/zgoussea/scratch/era5_hb_daily_nc data/era5_hb_daily_nc
    # ds = xr.open_mfdataset(glob.glob('data/hb_era5_glorys_nc/*.nc'))  # ln -s /home/zgoussea/scratch/hb_era5_glorys_nc data/hb_era5_glorys_nc
    # ds = xr.open_zarr('data/hb_era5_glorys.zarr')  # ln -s /home/zgoussea/scratch/hb_era5_glorys.zarr/  data/hb_era5_glorys.zarr

    training_years = range(2011, 2016)

    mask_0 = np.isnan(ds0.siconc.isel(time=0)).values
    mask = np.isnan(ds.siconc.isel(time=0)).values

    np.random.seed(21)
    random.seed(21)
    torch.manual_seed(21)

    # Number of frames to read as input
    input_timesteps = 10
    output_timesteps= 90

    print(input_timesteps, output_timesteps)

    start = time.time()

    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf']
    y_vars = ['siconc']  # ['siconc', 't2m']

    input_features = len(x_vars)

    # Half resolution datasets
    data_train_0 = IceDataset(ds0, training_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True)
    data_test_0 = IceDataset(ds0, [training_years[-1]+1], month, input_timesteps, output_timesteps, x_vars, y_vars)
    data_val_0 = IceDataset(ds0, [training_years[-1]+2], month, input_timesteps, output_timesteps, x_vars, y_vars)

    loader_train_0 = DataLoader(data_train_0, batch_size=1, shuffle=True)
    loader_test_0 = DataLoader(data_test_0, batch_size=1, shuffle=True)
    loader_val_0 = DataLoader(data_val_0, batch_size=1, shuffle=False)

    climatology_0 = ds0[y_vars].groupby('time.dayofyear').mean('time', skipna=True).to_array().values
    climatology_0 = torch.tensor(np.nan_to_num(climatology_0)).to(device)
    
    # Full resolution datasets
    data_train = IceDataset(ds, training_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True)
    data_test = IceDataset(ds, [training_years[-1]+1], month, input_timesteps, output_timesteps, x_vars, y_vars)
    data_val = IceDataset(ds, [training_years[-1]+2], month, input_timesteps, output_timesteps, x_vars, y_vars)

    loader_train = DataLoader(data_train, batch_size=1, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=1, shuffle=True)
    loader_val = DataLoader(data_val, batch_size=1, shuffle=False)

    climatology = ds[y_vars].groupby('time.dayofyear').mean('time', skipna=True).to_array().values
    climatology = torch.tensor(np.nan_to_num(climatology)).to(device)

    # Set threshold 
    # thresh = 0.15
    thresh = -np.inf
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
        model_kwargs=model_kwargs)

    print('Num. parameters:', model.get_n_params())
    print('Model:\n', model.model)

    lr = 0.01

    # Train model
    model.model.train()

    model.train(
        loader_train,
        loader_test,
        climatology,
        lr=lr,
        n_epochs=5,
        mask=mask) 

    

    # Train with half resolution first
    model.train(
        loader_train_0,
        loader_test_0,
        climatology_0,
        lr=lr,
        n_epochs=5,
        mask=mask_0) 

    # Train with full resolution 
    model.train(
        loader_train,
        loader_test,
        climatology,
        lr=lr,
        n_epochs=5,
        mask=mask) 
    
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

    results_dir = f'ice_results_{convolution_type}_apr20'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    ds.to_netcdf(f'{results_dir}/valpredictions_{experiment_name}.nc')

    model.loss.to_csv(f'{results_dir}/loss_{experiment_name}.csv')
    model.save(results_dir)

    print(f'Finished model {month} in {((time.time() - start) / 60)} minutes')
