import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import datetime
import os
import time
import xarray as xr
from dateutil.relativedelta import relativedelta

import argparse

from utils import normalize

from mpnnlstm import NextFramePredictorS2S
from seq2seq import Seq2Seq

from torch.utils.data import Dataset, DataLoader

class IceDataset(Dataset):
    def __init__(self, ds, years, month, input_timesteps, output_timesteps, x_vars=None, y_vars=None):
        self.x, self.y = self.get_xy(ds, years, month, input_timesteps, output_timesteps, x_vars=x_vars, y_vars=y_vars)
        self.image_shape = self.x[0].shape[1:-1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_xy(self, ds, years, month, input_timesteps, output_timesteps, x_vars=None, y_vars=None):

        x, y = [], []
        for year in years:
            # 3 months around the month of interest
            start_date = datetime.datetime(year, month, 1) - relativedelta(months=1)
            end_date = datetime.datetime(year, month, 1) + relativedelta(months=2)

            # Add buffer for input timesteps and output timesteps 
            start_date -= relativedelta(days=input_timesteps)
            end_date += relativedelta(days=output_timesteps-1)

            # Slice dataset & normalize
            ds_year = ds.sel(time=slice(start_date, end_date))
            # print(ds)
            ds_year = (ds_year - ds_year.min()) / (ds_year.max() - ds_year.min())

            num_samples = ds_year.time.size - output_timesteps - input_timesteps

            x_vars = list(ds.data_vars) if x_vars is None else x_vars
            y_vars = list(ds.data_vars) if y_vars is None else y_vars

            i = 0
            x_year = np.ndarray((num_samples, input_timesteps, ds.latitude.size, ds.longitude.size, len(x_vars)))
            y_year = np.ndarray((num_samples, output_timesteps, ds.latitude.size, ds.longitude.size, len(y_vars)))
            while i + output_timesteps + input_timesteps < ds_year.time.size:
                x_year[i] = np.moveaxis(np.nan_to_num(ds_year[x_vars].isel(time=slice(i, i+input_timesteps)).to_array().to_numpy()), 0, -1)
                y_year[i] = np.moveaxis(np.nan_to_num(ds_year[y_vars].isel(time=slice(i+input_timesteps, i+input_timesteps+output_timesteps)).to_array().to_numpy()), 0, -1)
                i += 1

            x.append(x_year)
            y.append(y_year)

        return np.concatenate(x, 0), np.concatenate(y, 0)


if __name__ == '__main__':
    

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--month")  # Month number

    # args = vars(parser.parse_args())
    # month = int(args['month'])

    ds = xr.open_zarr('data/era5_hb_daily.zarr')    # ln -s /home/zgoussea/scratch/era5_hb_daily.zarr data/era5_hb_daily.zarr

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
    training_years = range(2010, 2016)

    input_features = len(x_vars)
    
    months = range(1, 13)
    for month in months:

        data_train = IceDataset(ds, training_years, month, input_timesteps, output_timesteps, x_vars, y_vars)
        data_test = IceDataset(ds, [training_years[-1]+1], month, input_timesteps, output_timesteps, x_vars, y_vars)
        data_val = IceDataset(ds, [training_years[-1]+2], month, input_timesteps, output_timesteps, x_vars, y_vars)
        
        loader_train = DataLoader(data_train, batch_size=1, shuffle=True)
        loader_test = DataLoader(data_test, batch_size=1, shuffle=True)
        loader_val = DataLoader(data_val, batch_size=1, shuffle=True)

        # Add 3 to the number of input features since we add positional encoding (x, y) and node size (s)
        nn = Seq2Seq(
            hidden_size=64,
            dropout=0.1,
            thresh=0.15,
            input_features=input_features+3,
            output_timesteps=output_timesteps,
            n_layers=3).float()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device:', device)
        
        experiment_name = str(month) + '_test'

        model = NextFramePredictorS2S(
            nn,
            thresh=0.15,
            experiment_name=experiment_name,
            input_features=input_features,
            output_timesteps=output_timesteps)

        print('Num. parameters:', model.get_n_params())
        print('Model:\n', model.model)

        lr = 0.05

        model.model.train()
        model.train(loader_train, loader_test, lr=lr, n_epochs=15, mask=mask)  # Train for 20 epochs

        # model.model.eval()
        # model.score(x_val, y_val[:, :1])  # Check the MSE on the validation set
        # Unfinished

        model.loss.to_csv(f'ice_results/loss_{experiment_name}.csv')
        model.save('ice_results')

        print(f'Finished model {month} in {time.time() - start}')