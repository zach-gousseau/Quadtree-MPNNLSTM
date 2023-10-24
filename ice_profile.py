import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import datetime
import glob
import pickle
import os
import time
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta

import argparse

from model.utils import normalize
from model.mpnnlstm import NextFramePredictorS2S
from model.seq2seq import Seq2Seq

from ice_dataset import IceDataset
from torch.utils.data import Dataset, DataLoader

from model.graph_functions import create_static_heterogeneous_graph, create_static_homogeneous_graph, flatten, unflatten, flatten_pixelwise

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    # device = torch.device('mps')
    print('device:', device)

    month = 6
    # convolution_type = 'TransformerConv'
    # convolution_type = 'GCNConv'
    convolution_type = 'GINEConv'
    # convolution_type = 'Dummy'
    generate_predictions = True

    ds = xr.open_mfdataset(glob.glob('data/ERA5_GLORYS_2x/*.nc'))  # ln -s /home/zgoussea/scratch/ERA5_GLORYS data/ERA5_GLORYS
    # ds = xr.open_mfdataset(glob.glob('/home/zgoussea/scratch/ERA5_D/*.nc'))

    # ds = ds.isel(latitude=slice(175, 275), longitude=slice(125, 225))

    coarsen = 1

    if coarsen > 1:
        ds = ds.coarsen(latitude=coarsen, longitude=coarsen, boundary='trim').mean()
    elif coarsen < 1:
        newres = 0.25 * coarsen
        newlat = np.arange(ds.latitude.min(), ds.latitude.max() + newres, newres)
        newlon = np.arange(ds.longitude.min(), ds.longitude.max() + newres, newres)
        ds = ds.interp(latitude=newlat, longitude=newlon, method='nearest')

    mask = np.isnan(ds.siconc.isel(time=0)).values
    # high_interest_region = xr.open_dataset('data/shipping_corridors/primary_route_mask.nc').band_data.values
    high_interest_region = None

    image_shape = mask.shape
    graph_structure = create_static_heterogeneous_graph(image_shape, 4, mask, use_edge_attrs=True, resolution=1/12, device=device)
    # graph_structure = create_static_homogeneous_graph(image_shape, 4, mask, use_edge_attrs=True, resolution=1/12, device=device)

    print(f'Num nodes: {len(graph_structure["graph_nodes"])}')


    # graph_structure = None

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    binary = False
    binary_thresh = 0.15

    truncated_backprop = 0

    # Number of frames to read as input
    input_timesteps = 10
    output_timesteps= 45

    start = time.time()

    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf']#, 'usi', 'vsi', 'sithick']
    y_vars = ['siconc']  # ['siconc', 't2m']
    training_years = range(2012, 2013)
    
    cache_dir=None#'/home/zgoussea/scratch/data_cache/'
    directory = f'results/ice_results_profile'

    climatology = ds[y_vars].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).to_array().values
    climatology = torch.tensor(np.nan_to_num(climatology)).to(device)
    
    if graph_structure is not None:
        climatology = torch.moveaxis(climatology, 0, -1)
        climatology = flatten(climatology, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
        climatology = torch.moveaxis(climatology, -1, 0)
    else:
        climatology = flatten_pixelwise(climatology[0], mask)
        climatology = climatology.unsqueeze(0)

    input_features = len(x_vars)# + (len(x_vars)+3)
    
    data_train = IceDataset(ds, 
                            training_years, 
                            month, 
                            input_timesteps, 
                            output_timesteps,
                            x_vars,
                            y_vars, 
                            train=True, 
                            y_binary_thresh=binary_thresh if binary else None, 
                            graph_structure=graph_structure,
                            mask=mask, 
                            cache_dir=cache_dir,
                            flatten_y=True
                            )
    data_test = IceDataset(ds, 
                           range(2013, 2014),
                           month, 
                           input_timesteps, 
                           output_timesteps,
                           x_vars,
                           y_vars, 
                           y_binary_thresh=binary_thresh if binary else None,
                           graph_structure=graph_structure,
                           mask=mask, 
                           cache_dir=cache_dir,
                           flatten_y=True
                           )

    loader_profile = DataLoader(data_train, batch_size=1)#, sampler=torch.utils.data.SubsetRandomSampler(range(30)))
    loader_test = DataLoader(data_test, batch_size=1, shuffle=False)#, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(range(5)))

    thresh = 0.15
    thresh = -np.inf

    def dist_from_05(arr):
        return abs(abs(arr - 0.5) - 0.5)

    model_kwargs = dict(
        hidden_size=16,
        dropout=0.1,
        n_layers=1,
        n_conv_layers=1,
        transform_func=dist_from_05,
        dummy=False,
        convolution_type=convolution_type,
        rnn_type='NoConvLSTM',
        image_shape=image_shape
    )

    experiment_name = f'M{str(month)}_Y{training_years[0]}_Y{training_years[-1]}_I{input_timesteps}O{output_timesteps}'

    model = NextFramePredictorS2S(
        thresh=thresh,
        experiment_name=experiment_name,
        directory=directory,
        input_features=input_features,
        input_timesteps=input_timesteps,
        output_timesteps=output_timesteps,
        transform_func=dist_from_05,
        device=device,
        debug=False,
        model_kwargs=model_kwargs)

    print('Num. parameters:', model.get_n_params())

    # print(model.model)

    lr = 0.01

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
        high_interest_region=high_interest_region, 
        graph_structure=graph_structure, 
        truncated_backprop=False,#truncated_backprop
        )

    pr.disable()
    stats = pstats.Stats(pr).sort_stats('time')
    stats.print_stats(10)

    if generate_predictions:

        # data_val = IceDataset(ds, [training_years[-1]+2], month, input_timesteps, output_timesteps, x_vars, y_vars, cache_dir=cache_dir)
        # loader_val = DataLoader(data_val, batch_size=1, shuffle=False)

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
        launch_dates = loader_test.dataset.launch_dates
        
        y_true = torch.Tensor(loader_test.dataset.y).to(device)
        
        if graph_structure is not None:
            y_true = torch.stack([unflatten(y_true[i], graph_structure['mapping'], image_shape, mask).detach().cpu() for i in range(y_true.shape[0])])

        
        ds = xr.Dataset(
            data_vars=dict(
                y_hat=(["launch_date", "timestep", "latitude", "longitude"], val_preds[..., 0]),
                y_hat_sip=(["launch_date", "timestep", "latitude", "longitude"], val_preds[..., 1]),
                y_true=(["launch_date", "timestep", "latitude", "longitude"], y_true.squeeze(-1)),
            ),
            coords=dict(
                longitude=ds.longitude,
                latitude=ds.latitude,
                launch_date=launch_dates,
                timestep=np.arange(1, output_timesteps+1),
            ),
        )

        fns = []
        for ts in range(output_timesteps):
            year = 2013
            fig, axs=plt.subplots(1, 4, figsize=(20, 4))
            ds.isel(launch_date=5, timestep=ts).y_hat.where(~mask).plot(vmin=0, vmax=1, ax=axs[2])
            ds.isel(launch_date=5, timestep=ts).y_hat_sip.where(~mask).plot(vmin=0, vmax=1, ax=axs[3])
            ds.isel(launch_date=5, timestep=ts).y_true.where(~mask).plot(vmin=0, vmax=1, ax=axs[0])
            (ds.isel(launch_date=5, timestep=ts).y_true>0.15).where(~mask).plot(vmin=0, vmax=1, ax=axs[1])
            axs[0].set_title(f'True ({str(datetime.datetime(year, month, 5))[:10]}, step {ts})')
            axs[1].set_title(f'True ({str(datetime.datetime(year, month, 5))[:10]}, step {ts})')
            axs[2].set_title(f'Pred ({str(datetime.datetime(year, month, 5))[:10]}, step {ts})')
            axs[3].set_title(f'Pred ({str(datetime.datetime(year, month, 5))[:10]}, step {ts})')
            plt.tight_layout()
            fn = f'scratch/gif/{str(datetime.datetime(year, month, 5))[:10]}_{ts}.png'
            fns.append(fn)
            plt.savefig(fn)
            plt.close()
        from PIL import Image
        frames = []
        for fn in fns:
            new_frame = Image.open(fn)
            frames.append(new_frame)
        frames[0].save(f'scratch/gif/{str(datetime.datetime(year, month, 15))[:10]}.gif',
                    format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=300,
                    loop=0)
        for fn in fns:
            os.remove(fn)
            

        ds.to_netcdf(f'{directory}/valpredictions_{experiment_name}.nc')

        model.loss.to_csv(f'{directory}/loss_{experiment_name}.csv')
        model.save(directory)