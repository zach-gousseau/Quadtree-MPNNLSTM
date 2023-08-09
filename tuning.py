import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import datetime
import os
import time
from calendar import monthrange, month_name
import seaborn as sns
import glob
import pandas as pd
import xarray as xr
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

import argparse

from model.utils import normalize, int_to_datetime

from model.mpnnlstm import NextFramePredictorS2S
from model.seq2seq import Seq2Seq

from torch.utils.data import Dataset, DataLoader

from ice_dataset import IceDataset

from model.graph_functions import create_static_heterogeneous_graph, create_static_homogeneous_graph, flatten, unflatten

def masked_RMSE_along_axis(mask):
    def loss(y_true, y_pred):
        sq_diff = np.multiply((y_pred - y_true)**2, mask)
        return np.sqrt(np.mean(sq_diff, (1, 2)))
    return loss

def create_heatmap_fast(ds, accuracy=False):
    timestep_values = ds.timestep.values.astype(int)
    launch_date_values = ds.launch_date.values
    launch_months = pd.DatetimeIndex(launch_date_values).month
    heatmap = np.zeros((12, len(timestep_values)))
    heatmap_n = np.zeros_like(heatmap)
    
    for i, timestep in enumerate(tqdm(timestep_values)):
        arr = ds.sel(timestep=timestep).to_array().values
        arr = np.nan_to_num(arr)
        err = masked_RMSE_along_axis(~mask)(arr[0], arr[1])
            
        for j, e in enumerate(err):
            heatmap[launch_months[j]-1, i] += e
            heatmap_n[launch_months[j]-1, i] += 1
    
    heatmap /= heatmap_n
    heatmap = pd.DataFrame(heatmap, index=range(1, 13), columns=ds.timestep.values)
    return heatmap


if __name__ == '__main__':
    
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp')

    args = vars(parser.parse_args())
    exp = int(args['exp'])

    start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    month = 6
    
    # Full resolution dataset
    # ds = xr.open_mfdataset(glob.glob('data/ERA5_GLORYS/*.nc'))  # ln -s /home/zgoussea/scratch/ERA5_GLORYS data/ERA5_GLORYS
    ds = xr.open_mfdataset(glob.glob('/home/zgoussea/scratch/ERA5_D/*.nc'))
    mask = np.isnan(ds.siconc.isel(time=0)).values
    # high_interest_region = xr.open_dataset('data/shipping_corridors/primary_route_mask.nc').band_data.values
    high_interest_region = None

    image_shape = mask.shape
    preset_mesh = 'heterogeneous'
    training_years = range(2009, 2012)
    
    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf', 'usi', 'vsi', 'sithick']
    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf']
    y_vars = ['siconc']
    input_features = len(x_vars)
    input_timesteps = 5
    output_timesteps= 45

    if preset_mesh == 'heterogeneous':
        graph_structure = create_static_heterogeneous_graph(image_shape, 1, mask, high_interest_region=high_interest_region, use_edge_attrs=False, resolution=1/12, device=device)
    elif preset_mesh == 'homogeneous':
        graph_structure = create_static_homogeneous_graph(image_shape, 1, mask, high_interest_region=high_interest_region, use_edge_attrs=False, resolution=1/12, device=device)
    
    # Full resolution datasets
    data_train = IceDataset(ds, training_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True, graph_structure=graph_structure, mask=mask)
    data_test = IceDataset(ds, range(training_years[-1]+1, training_years[-1]+1+2), month, input_timesteps, output_timesteps, x_vars, y_vars, graph_structure=graph_structure, mask=mask)

    loader_train = DataLoader(data_train, batch_size=1, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=1, shuffle=True)

    climatology = ds[y_vars].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).to_array().values
    climatology = torch.tensor(np.nan_to_num(climatology)).to(device)
    climatology = torch.moveaxis(climatology, 0, -1)
    climatology = flatten(climatology, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
    climatology = torch.moveaxis(climatology, -1, 0)
    
    # exps = range(0, 7)
    # for exp in exps:   

    # Defaults
    convolution_type = 'GCNConv'
    lr = 0.001
    multires_training = False
    truncated_backprop = 0
    rnn_type = 'GRU'
    
    hidden_size=16
    dropout=0.1
    n_layers=1
    n_conv_layers=1

    binary=False

    np.random.seed(21)
    random.seed(21)
    torch.manual_seed(21)
    
    # Experiment definitions --------------------
    if exp == 1:
        convolution_type = 'TransformerConv'
    elif exp == 2:
        rnn_type = 'LSTM'
    elif exp == 3:
        hidden_size = 32
    elif exp == 4:
        n_layers = 2
    elif exp == 5:
        n_conv_layers = 2
    elif exp == 6:
        n_conv_layers = 3
    elif exp == 7:
        n_conv_layers = 3

    experiment_name = f'M{str(month)}_Y{training_years[0]}_Y{training_years[-1]}_I{input_timesteps}O{output_timesteps}_EXP_{exp}'
    
    
    use_edge_attrs = True if convolution_type == 'TransformerConv' else False
    if preset_mesh == 'heterogeneous':
        graph_structure = create_static_heterogeneous_graph(image_shape, 1, mask, high_interest_region=high_interest_region, use_edge_attrs=use_edge_attrs, resolution=1/12, device=device)
    elif preset_mesh == 'homogeneous':
        graph_structure = create_static_homogeneous_graph(image_shape, 1, mask, high_interest_region=high_interest_region, use_edge_attrs=use_edge_attrs, resolution=1/12, device=device)
    
    # Arguments passed to Seq2Seq constructor
    model_kwargs = dict(
        hidden_size=hidden_size,
        dropout=dropout,
        n_layers=n_layers,
        dummy=False,
        n_conv_layers=n_conv_layers,
        rnn_type=rnn_type,
        convolution_type=convolution_type,
    )

    model = NextFramePredictorS2S(
        thresh=-np.inf,
        experiment_name=experiment_name,
        input_features=input_features,
        input_timesteps=input_timesteps,
        output_timesteps=output_timesteps,
        device=device,
        binary=binary,
        debug=False, 
        model_kwargs=model_kwargs)

    print('Num. parameters:', model.get_n_params())
    print('Model:\n', model.model)

    model.model.train()

    # Train with full resolution. Use high interest region.
    training_time = time.time()
    model.train(
        loader_train,
        loader_test,
        climatology,
        lr=lr,
        n_epochs=30,
        mask=mask,
        high_interest_region=high_interest_region,  # This should not be necessary
        truncated_backprop=truncated_backprop,
        graph_structure=graph_structure,
        ) 
    
    training_time = time.time() - training_time

    # Save model and losses
    results_dir = f'results/ice_results_tuning'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model.loss.to_csv(f'{results_dir}/loss_{experiment_name}.csv')
    
    # Generate predictions
    model.model.eval()
    val_preds = model.predict(
        loader_test,
        climatology,
        mask=mask,
        high_interest_region=high_interest_region,
        graph_structure=graph_structure
        )
    
    # Save results
    launch_dates = [int_to_datetime(t) for t in loader_test.dataset.launch_dates]
    
    y_true = torch.Tensor(loader_test.dataset.y)
    
    if graph_structure is not None:
        y_true = torch.Tensor(loader_test.dataset.y)
        y_true = torch.stack([unflatten(y_true[i].to(device), graph_structure['mapping'], image_shape, mask).detach().cpu() for i in range(y_true.shape[0])])
        y_true = np.array(y_true)

    ds = xr.Dataset(
        data_vars=dict(
            y_hat=(["launch_date", "timestep", "latitude", "longitude"], val_preds.squeeze(-1).astype('float')),
            y_true=(["launch_date", "timestep", "latitude", "longitude"], y_true.squeeze(-1).astype('float')),
        ),
        coords=dict(
            longitude=ds.longitude,
            latitude=ds.latitude,
            launch_date=launch_dates,
            timestep=np.arange(1, output_timesteps+1),
        ),
    )
    
    heatmap = create_heatmap_fast(ds)
    
    plt.figure(dpi=80)
    sns.heatmap(heatmap, yticklabels=[month_name[i][:3] for i in range(1, 13)], vmax=0.18, vmin=0.02)
    plt.xlabel('Lead time (days)')
    plt.savefig(f'{results_dir}/heatmap_{exp}.png')
    plt.close()

    heatmap.to_csv(f'{results_dir}/heatmap_{exp}.csv')
    
    plt.figure(figsize=(6, 3))
    plt.plot(model.loss.train_loss, label='train')
    plt.plot(model.loss.test_loss, label='test')
    plt.legend()
    
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.title(f'Experiment {exp}')
    plt.savefig(f'{results_dir}/losses_{exp}.png')
    
    pd.DataFrame({'training_time': [training_time], 'num_params': model.get_n_params()}).to_csv(f'{results_dir}/data_{exp}.png')
    
    print(f'Finished model {exp} in {((time.time() - start) / 60)} minutes')
