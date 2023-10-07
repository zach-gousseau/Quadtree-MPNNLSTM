import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import datetime
import os
import time
from tqdm import tqdm
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

from model.graph_functions import create_static_heterogeneous_graph, create_static_homogeneous_graph, flatten, unflatten

def round_to_day(dt):
    return datetime.datetime(*dt.timetuple()[:3])

def masked_accuracy(mask):
    def loss(y_true, y_pred):
        y_true_masked = np.multiply(y_true, mask)
        y_pred_masked = np.multiply(y_pred, mask)
        return accuracy_score(y_true_masked, y_pred_masked)
    return loss

def masked_MSE(mask):
    def loss(y_true, y_pred):
        sq_diff = np.multiply((y_pred - y_true)**2, mask)
        return np.mean(sq_diff)
    return loss

def masked_RMSE(mask):
    def loss(y_true, y_pred):
        sq_diff = np.multiply((y_pred - y_true)**2, mask)
        return np.sqrt(np.mean(sq_diff))
    return loss

def masked_RMSE_along_axis(mask):
    def loss(y_true, y_pred):
        sq_diff = ((y_pred - y_true)**2)[:, mask]
        return np.sqrt(np.mean(sq_diff, (1)))
    return loss

def create_heatmap(ds, accuracy=False):
    heatmap = pd.DataFrame(0.0, index=range(1, 13), columns=ds.timestep)
    heatmap_n = pd.DataFrame(0.0, index=range(1, 13), columns=ds.timestep)

    for timestep in tqdm(ds.timestep):
        timestep = int(timestep.values)
        for launch_date in ds.launch_date:
            try:
                arr = ds.sel(timestep=timestep, launch_date=launch_date).to_array().values
                arr = np.nan_to_num(arr)
            except ValueError:
                continue

            if accuracy:
                arr = arr > 0.5
                err = masked_accuracy(~mask)(arr[0], arr[1])
            else:
                err = masked_RMSE(~mask)(arr[0], arr[1])

            launch_month = pd.Timestamp(launch_date.values).month

            heatmap[timestep][launch_month] += err
            heatmap_n[timestep][launch_month] += 1

    heatmap = heatmap.div(heatmap_n)
    return heatmap

def create_heatmap_fast(ds, accuracy=False):
    timestep_values = ds.timestep.values.astype(int)
    launch_date_values = ds.launch_date.values
    launch_months = pd.DatetimeIndex(launch_date_values).month
    heatmap = np.zeros((12, len(timestep_values)))
    heatmap_n = np.zeros_like(heatmap)
    
    for i, timestep in enumerate(tqdm(timestep_values)):
        arr = ds.sel(timestep=timestep).to_array().values
        arr = np.nan_to_num(arr)
        
        if accuracy:
            arr = arr > 0.5
            err = masked_accuracy(~mask)(arr[0], arr[1])
        else:
            err = masked_RMSE_along_axis(~mask)(arr[0], arr[1])
            
        for j, e in enumerate(err):
            heatmap[launch_months[j]-1, i] += e
            heatmap_n[launch_months[j]-1, i] += 1
    
    heatmap /= heatmap_n
    heatmap = pd.DataFrame(heatmap, index=range(1, 13), columns=ds.timestep.values)
    return heatmap

def round_to_day(dt):
    return datetime.datetime(*dt.timetuple()[:3])

def flatten_unflatten(arr, graph_structure, mask):
    arr = flatten(arr, graph_structure['mapping'], graph_structure['n_pixels_per_node'], mask=~mask)
    arr = unflatten(arr, graph_structure['mapping'], mask.shape, mask=~mask)
    return arr

if __name__ == '__main__':

    np.random.seed(21)
    random.seed(21)
    torch.manual_seed(21)

    start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ys', '--yearstart')
    parser.add_argument('-ye', '--yearend')

    args = vars(parser.parse_args())
    ys = int(args['yearstart'])
    ye = int(args['yearend'])
    

    convolution_type = 'GCNConv'
    lr = 0.001
    multires_training = False
    truncated_backprop = 0
    
    training_years_1 = training_years_4 = range(1993, 2013)
    inference_years = range(ys, ye)
    
    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf']
    y_vars = ['siconc']
    input_features = len(x_vars)
    input_timesteps = 10
    output_timesteps= 90
    preset_mesh = False
    rnn_type = 'LSTM'
    
    # EXPERIMENT 20 
    multires_training = False
    preset_mesh = 'heterogeneous'
    rnn_type = 'NoConvLSTM'
    n_epochs = [50]
    results_dir = f'results/ice_results_20years_era5_6conv_noconv_20yearsstraight_splitgconvlstm_adam'

    
    cache_dir='/home/zgoussea/scratch/data_cache/'

    binary=False



    ds = xr.open_mfdataset(glob.glob('/home/zgoussea/scratch/ERA5_D/*.nc'))  # ln -s /home/zgoussea/scratch/ERA5_GLORYS data/ERA5_GLORYS
    mask = np.isnan(ds.siconc.isel(time=0)).values

    use_edge_attrs = False if convolution_type == 'GCNConv' else True
    
    # high_interest_region = xr.open_dataset('data/shipping_corridors/primary_route_mask.nc').band_data.values
    high_interest_region = None

    image_shape = mask.shape
    if preset_mesh == 'heterogeneous':
        graph_structure = create_static_heterogeneous_graph(image_shape, 1, mask, high_interest_region=high_interest_region, use_edge_attrs=use_edge_attrs, resolution=1/12, device=device)
    elif preset_mesh == 'homogeneous':
        graph_structure = create_static_homogeneous_graph(image_shape, 1, mask, high_interest_region=high_interest_region, use_edge_attrs=use_edge_attrs, resolution=1/12, device=device)

    climatology = ds[y_vars].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).to_array().values
    climatology = torch.tensor(np.nan_to_num(climatology)).to(device)
    climatology = torch.moveaxis(climatology, 0, -1)
    climatology = flatten(climatology, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
    climatology = torch.moveaxis(climatology, -1, 0)

    # Set threshold 
    thresh = -np.inf
    print(f'Threshold is {thresh}')

    # Note: irrelevant if thresh = -np.inf
    def dist_from_05(arr):
        return abs(abs(arr - 0.5) - 0.5)

    # Arguments passed to Seq2Seq constructor
    model_kwargs = dict(
        hidden_size=64,
        dropout=0.1,
        n_layers=1,
        transform_func=dist_from_05,
        dummy=False,
        n_conv_layers=6,
        rnn_type=rnn_type,
        convolution_type=convolution_type,
    )
    ds_pred = [] 
    for month in range(1, 13):
        experiment_name = f'M{str(month)}_Y{training_years_1[0]}_Y{training_years_4[-1]}_I{input_timesteps}O{output_timesteps}'

        model = NextFramePredictorS2S(
            thresh=thresh,
            experiment_name=experiment_name,
            input_features=input_features,
            input_timesteps=input_timesteps,
            output_timesteps=output_timesteps,
            transform_func=dist_from_05,
            device=device,
            binary=binary,
            debug=False, 
            model_kwargs=model_kwargs)

        print('Doing month', month)
        data_val = IceDataset(ds, inference_years, month, input_timesteps, output_timesteps, x_vars, y_vars, graph_structure=graph_structure, mask=mask, cache_dir=cache_dir)
        loader_val = DataLoader(data_val, batch_size=1, shuffle=False)

        model.load(results_dir)
        
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
        launch_dates = [int_to_datetime(t) for t in loader_val.dataset.launch_dates]
        
        y_true = loader_val.dataset.y
        
        if graph_structure is not None:
            y_true = torch.stack([unflatten(torch.Tensor(y_true)[i].to(device), graph_structure['mapping'], image_shape, mask).detach().cpu() for i in range(y_true.shape[0])])
            y_true = np.array(y_true)

        ds_ = xr.Dataset(
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
        ds_pred.append(ds_)
        print(f'Finished model {month} in {((time.time() - start) / 60)} minutes')
        

    ds_pred = xr.concat(ds_pred, dim='launch_date')
    ds_pred = ds_pred.rio.set_crs(4326)
    ds_pred['launch_date'] = [round_to_day(pd.Timestamp(dt)) + datetime.timedelta(days=1) for dt in ds_pred.launch_date.values]
    image_shape = mask.shape
        
    # HEATMAP ----------------------

    heatmap = create_heatmap_fast(ds_pred)
    heatmap.to_csv(f'{results_dir}/heatmap_{inference_years[0]}_{inference_years[-1]}.csv')