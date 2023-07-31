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
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from model.graph_functions import create_static_heterogeneous_graph, create_static_homogeneous_graph, flatten, unflatten

class Sampler(SubsetRandomSampler):
    def __init__(self, indices):
        super().__init__(indices[:1])

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    month = 1
    # convolution_type = 'TransformerConv'
    convolution_type = 'GCNConv'
    # convolution_type = 'Dummy'
    generate_predictions = True
    
    scalability_stats = []

    for size in np.linspace(10, 350, 15)[::-1]:
        size = int(size)
        print(size)
        
        # Define the dimensions and coordinates
        longitude = np.linspace(-95.0, -65.0, size)
        latitude = np.linspace(51.0, 70.0, size)
        times = np.arange('2014-01-01', '2015-01-01', dtype='datetime64[D]')

        # Create the DataArray filled with zeros
        num_var = 5
        var_data = []
        for _ in range(num_var):
            var_data.append(np.zeros((len(times), len(latitude), len(longitude)), dtype=np.float32))

        # Create the xarray Dataset
        ds = xr.Dataset(
            {f'var_{i}': (['time', 'latitude', 'longitude'], data) for i, data in enumerate(var_data)},
            coords={
                'longitude': longitude,
                'latitude': latitude,
                'time': times,
            }
        )
        

        mask = np.zeros_like(ds.var_1.isel(time=0).values).astype(bool)
        high_interest_region = None

        image_shape = mask.shape
        # graph_structure = create_static_heterogeneous_graph(image_shape, 4, mask, use_edge_attrs=True, resolution=1/12, device=device)
        graph_structure = create_static_homogeneous_graph(image_shape, 1, mask, use_edge_attrs=False, resolution=1/12, device=device)

        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)

        truncated_backprop = 0

        # Number of frames to read as input
        input_timesteps = 10
        output_timesteps= 90

        start = time.time()

        x_vars = [f'var_{i}' for i in range(num_var)]
        y_vars = ['var_1']  # ['siconc', 't2m']
        training_years = range(2014, 2015)
        
        cache_dir=None#'/home/zgoussea/scratch/data_cache/'

        climatology = ds[y_vars].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).to_array().values
        climatology = torch.tensor(np.nan_to_num(climatology)).to(device)
        climatology = torch.moveaxis(climatology, 0, -1)
        climatology = flatten(climatology, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
        climatology = torch.moveaxis(climatology, -1, 0)

        input_features = len(x_vars)
        
        data_train = IceDataset(ds, 
                                training_years, 
                                month, 
                                input_timesteps, 
                                output_timesteps,
                                x_vars,
                                y_vars, 
                                train=True, 
                                y_binary_thresh=None, 
                                graph_structure=graph_structure,
                                mask=mask, 
                                cache_dir=cache_dir
                                )
        data_test = IceDataset(ds, 
                            training_years,
                            month, 
                            input_timesteps, 
                            output_timesteps,
                            x_vars,
                            y_vars, 
                            y_binary_thresh=None,
                            graph_structure=graph_structure,
                            mask=mask, 
                            cache_dir=cache_dir
                            )

        loader_profile = DataLoader(data_train, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(range(10)))
        loader_test = DataLoader(data_test, batch_size=1, shuffle=False, sampler=Sampler([0]))

        thresh = 0.15
        thresh = -np.inf

        def dist_from_05(arr):
            return abs(abs(arr - 0.5) - 0.5)

        model_kwargs = dict(
            hidden_size=32,
            dropout=0.1,
            n_layers=1,
            n_conv_layers=3,
            transform_func=dist_from_05,
            dummy=False,
            convolution_type=convolution_type,
            rnn_type='GRU',
            image_shape=image_shape
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
            debug=False,
            model_kwargs=model_kwargs)

        lr = 0.01

        model.model.train()

        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
        ss = time.time()
        model.train(
            loader_profile,
            loader_test,
            climatology,
            lr=lr, 
            n_epochs=1, 
            mask=mask, 
            high_interest_region=high_interest_region, 
            graph_structure=graph_structure, 
            truncated_backprop=False,#truncated_backprop
            )
        train_time = time.time() - ss

        pr.disable()
        stats = pstats.Stats(pr).sort_stats('time')
        
        backwards_time = stats.__dict__['stats'][[p for p in list(stats.__dict__['stats'].keys()) if 'run_backward' in p[-1]][0]][3]
        num_params = model.get_n_params()
        num_nodes = len(graph_structure['graph_nodes'])
        
        print(num_params, num_nodes, backwards_time, train_time)
        
        scalability_stats.append((num_params, num_nodes, backwards_time, train_time))
    print(scalability_stats)
    pd.DataFrame(scalability_stats).to_csv('scrap/scalability.csv')