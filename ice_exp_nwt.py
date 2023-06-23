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

    args = vars(parser.parse_args())
    month = int(args['month'])

    # Defaults
    convolution_type = 'TransformerConv'
    lr = 0.001
    truncated_backprop = 0

    training_years = range(2007, 2013)
    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf']
    y_vars = ['siconc']
    input_features = len(x_vars)
    input_timesteps = 10
    output_timesteps= 90
    preset_mesh = False

    binary=False

    # Full resolution dataset
    ds = xr.open_mfdataset(glob.glob('path_to_nwt_netcdf_files/*.nc'))
    mask = np.isnan(ds.siconc.isel(time=0)).values

    image_shape = mask.shape
    graph_structure = None
    
    # Create Pytorch loaders
    data_train = IceDataset(ds, training_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True)
    data_test = IceDataset(ds, [training_years[-1]+1], month, input_timesteps, output_timesteps, x_vars, y_vars)
    data_val = IceDataset(ds, range(training_years[-1]+2, training_years[-1]+2+4), month, input_timesteps, output_timesteps, x_vars, y_vars)

    loader_train = DataLoader(data_train, batch_size=1, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=1, shuffle=True)
    loader_val = DataLoader(data_val, batch_size=1, shuffle=False)

    # climatology = ds[y_vars].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).to_array().values
    # climatology = torch.tensor(np.nan_to_num(climatology)).to(device)

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

    # Train with full resolution 
    model.train(
        loader_train,
        loader_test,
        # climatology,
        lr=lr,
        n_epochs=15,
        mask=mask,
        truncated_backprop=truncated_backprop,
        graph_structure=graph_structure,
        ) 

    # Save model and losses
    results_dir = f'ice_results_nwt_example'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model.loss.to_csv(f'{results_dir}/loss_{experiment_name}.csv')
    model.save(results_dir)
    
    # Generate predictions
    model.model.eval()
    val_preds = model.predict(
        loader_val,
        # climatology,
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
