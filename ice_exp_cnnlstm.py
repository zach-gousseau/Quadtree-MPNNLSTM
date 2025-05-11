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
from model.cnnlstm import NextFramePredictorCNNLSTM

from torch.utils.data import Dataset, DataLoader

from ice_dataset import IceDataset


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
    lr = 0.0001
    training_years = range(2007, 2013)
    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf']
    y_vars = ['siconc']
    input_features = len(x_vars)
    input_timesteps = 10
    output_timesteps = 90
    hidden_size = 32
    n_layers = 2
    kernel_size = 3
    dropout = 0.1
    binary = False

    # Experiment definitions --------------------
    if exp == 1:
        kernel_size = 5
    elif exp == 2:
        lr = 0.001
    elif exp == 3:
        hidden_size = 64
    elif exp == 4:
        n_layers = 3
    elif exp == 5:
        dropout = 0.2
    elif exp == 6:
        input_timesteps = 30
    elif exp == 7:
        lr = 0.001
        input_timesteps = 30
    elif exp == 8:
        lr = 0.001
        input_timesteps = 90
    # -------------------------------------------

    # Full resolution dataset
    ds = xr.open_mfdataset(glob.glob('data/hb_era5_glorys_nc/*.nc'))
    mask = np.isnan(ds.siconc.isel(time=0)).values
    high_interest_region = xr.open_dataset('data/shipping_corridors/primary_route_mask.nc').band_data.values

    image_shape = mask.shape
    
    # Full resolution datasets
    data_train = IceDataset(ds, training_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True)
    data_test = IceDataset(ds, [training_years[-1]+1], month, input_timesteps, output_timesteps, x_vars, y_vars)
    data_val = IceDataset(ds, range(training_years[-1]+2, training_years[-1]+2+4), month, input_timesteps, output_timesteps, x_vars, y_vars)

    loader_train = DataLoader(data_train, batch_size=1, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=1, shuffle=True)
    loader_val = DataLoader(data_val, batch_size=1, shuffle=False)

    climatology = ds[y_vars].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).to_array().values
    climatology = torch.tensor(np.nan_to_num(climatology)).to(device)

    experiment_name = f'CNNLSTM_M{str(month)}_Y{training_years[0]}_Y{training_years[-1]}_I{input_timesteps}O{output_timesteps}'

    model = NextFramePredictorCNNLSTM(
        experiment_name=experiment_name,
        input_features=input_features,
        hidden_size=hidden_size,
        input_timesteps=input_timesteps,
        output_timesteps=output_timesteps,
        n_layers=n_layers,
        dropout=dropout,
        kernel_size=kernel_size,
        binary=binary,
        debug=True,
        device=device)

    print('Num. parameters:', model.get_n_params())
    print('Model:\n', model.model)

    # Train the model
    model.train(
        loader_train,
        loader_test,
        climatology,
        lr=lr,
        n_epochs=15,
        mask=mask,
        high_interest_region=high_interest_region,
    )

    # Save model and losses
    results_dir = f'ice_results_cnnlstm_{exp}'

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
    )
    
    # Save results
    launch_dates = [int_to_datetime(t) for t in loader_val.dataset.launch_dates]
    
    ds_output = xr.Dataset(
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
    ds_output.to_netcdf(f'{results_dir}/valpredictions_{experiment_name}.nc')
    print(f'Finished model {month} in {((time.time() - start) / 60)} minutes') 