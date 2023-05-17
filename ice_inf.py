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

from ice_test import IceDataset


if __name__ == '__main__':

    np.random.seed(21)
    random.seed(21)
    torch.manual_seed(21)

    start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    

    # Defaults
    convolution_type = 'TransformerConv'
    lr = 0.001
    multires_training = False
    truncated_backprop = 0

    training_years = range(2002, 2010)
    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf']
    y_vars = ['siconc']  # ['siconc', 't2m']
    input_features = len(x_vars)
    input_timesteps = 10
    output_timesteps= 90

    binary=False

    for month in range(11, 13):

        # Full resolution dataset
        ds = xr.open_mfdataset(glob.glob('data/era5_hb_daily_nc/*.nc'))  # ln -s /home/zgoussea/scratch/era5_hb_daily_nc data/era5_hb_daily_nc
        
        mask = np.isnan(ds.siconc.isel(time=0)).values

        climatology = ds[y_vars].groupby('time.dayofyear').mean('time', skipna=True).to_array().values
        climatology = torch.tensor(np.nan_to_num(climatology)).to(device)

        # Set threshold 
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
            n_conv_layers=3,
            rnn_type='LSTM',
            convolution_type=convolution_type,
        )

        print(month)
        data_val = IceDataset(ds, range(training_years[-1]+2, training_years[-1]+2+7), month, input_timesteps, output_timesteps, x_vars, y_vars)
        loader_val = DataLoader(data_val, batch_size=1, shuffle=False)

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

        # print('Num. parameters:', model.get_n_params())

        results_dir = f'ice_results_may10_exp_8y_train_5y_'

        model.load(results_dir)
        
        # Generate predictions
        model.model.eval()
        val_preds = model.predict(loader_val, climatology, mask=mask)
        
        # Save results
        launch_dates = loader_val.dataset.launch_dates
        
        ds_pred = xr.Dataset(
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

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        ds_pred.to_netcdf(f'{results_dir}/valpredictions_{experiment_name}.nc')

        print(f'Finished model {month} in {((time.time() - start) / 60)} minutes')
