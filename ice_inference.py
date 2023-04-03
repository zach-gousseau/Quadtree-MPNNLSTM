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

from ice_test import IceDataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--month')  # Month number

    args = vars(parser.parse_args())
    month = int(args['month'])

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
    inference_years = range(2011, 2018)

    climatology = ds[y_vars].groupby('time.month').mean('time', skipna=True).to_array().values
    climatology = np.nan_to_num(climatology)

    input_features = len(x_vars)
    
    data_inf = IceDataset(ds, inference_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=False)

    loader_inf = DataLoader(data_inf, batch_size=1, shuffle=False)#, collate_fn=lambda x: x[0])

    thresh = 0.15
    print(f'threshold is {thresh}')

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
    # device = torch.device('mps')
    print('device:', device)

    experiment_name = f'M{str(month)}_Y2011_Y2015_I{input_timesteps}O{output_timesteps}'

    results_dir = 'ice_results_gnn_out'

    model = NextFramePredictorS2S(
        thresh=thresh,
        experiment_name=experiment_name,
        input_features=input_features,
        output_timesteps=output_timesteps,
        transform_func=dist_from_05,
        device=device,
        model_kwargs=model_kwargs)
    
    model.load(results_dir)
    
    model.model.eval()
    inf_preds = model.predict(loader_inf, climatology, mask=mask)
    
    launch_dates = loader_inf.dataset.launch_dates
    
    ds = xr.Dataset(
        data_vars=dict(
            y_hat=(["launch_date", "timestep", "latitude", "longitude"], inf_preds.squeeze(-1)),
            y_true=(["launch_date", "timestep", "latitude", "longitude"], loader_inf.dataset.y.squeeze(-1)),
        ),
        coords=dict(
            longitude=ds.longitude,
            latitude=ds.latitude,
            launch_date=launch_dates,
            timestep=np.arange(1, output_timesteps+1),
        ),
    )

    results_dir = 'ice_results_gnn_out'
    
    ds.to_netcdf(f'{results_dir}/infpredictions_{experiment_name}.nc')

    print(f'Finished model {month} in {(time.time() - start / 60)} minutes')
