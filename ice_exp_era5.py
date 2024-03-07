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

from model.graph_functions import create_static_heterogeneous_graph, create_static_homogeneous_graph, flatten, unflatten


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
    # convolution_type = 'TransformerConv'
    convolution_type = 'GCNConv'
    lr = 0.001
    multires_training = False
    truncated_backprop = 0

    # training_years_1 = range(1993, 1998)
    # training_years_2 = range(2000, 2003)
    # training_years_4 = range(2003, 2013)
    
    # training_years_1 = range(2002, 2003)
    # training_years_4 = range(2002, 2010)
    
    training_years_1 = training_years_4 = range(1993, 2013)
    
    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf']
    y_vars = ['siconc']
    input_features = len(x_vars)
    input_timesteps = 10
    output_timesteps= 90
    preset_mesh = False
    rnn_type = 'LSTM'
    n_conv_layers = 3
    
    cache_dir='/home/zgoussea/scratch/data_cache/'
    cache_dir = None

    binary=False

    # Experiment definitions --------------------
    if exp == 1:
        convolution_type = 'GCNConv'
    elif exp == 2:
        lr = 0.001
    elif exp == 3:
        multires_training = True
    elif exp == 4:
        lr = 0.0001
    elif exp == 5:
        truncated_backprop = 45
    elif exp == 6:
        truncated_backprop = 30
    elif exp == 7:
        lr = 0.001
        input_timesteps = 30
    elif exp == 8:
        lr = 0.001
        input_timesteps = 90
    elif exp == 9:
        multires_training = False
        preset_mesh = 'heterogeneous'
    elif exp == 10:
        multires_training = True
        preset_mesh = 'homogeneous'
    elif exp == 11:
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'LSTM'
        n_epochs = [5, 5, 10, 10]
        directory = f'results/ice_results_20years_era5_LSTM_6conv'
    elif exp == 12:
        convolution_type = 'TransformerConv'
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'GRU'
        n_epochs = [5, 10]
        directory = f'results/ice_results_20years_smaller_era5_transformer'
    elif exp == 13:
        convolution_type = 'TransformerConv'
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'LSTM'
        n_epochs = [5, 5, 5, 10]
        directory = f'results/ice_results_15years_era5_transformer'
    elif exp == 14:
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [5, 10, 20, 30]
        directory = f'results/ice_results_20years_era5_6conv_noconv'
    elif exp == 15:
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [15, 15, 20]
        directory = f'results/ice_results_20years_era5_6conv_noconv_newyears'
    elif exp == 16:
        convolution_type = 'TransformerConv'
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [7, 7, 12]
        directory = f'results/ice_results_20years_era5_3convtransformer_noconv_newyears'
    elif exp == 17:
        convolution_type = 'TransformerConv'
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'GRU'
        n_epochs = [5, 10, 5]
        directory = f'results/ice_results_20years_smaller_era5_transformer_2'
    elif exp == 18:
        convolution_type = 'TransformerConv'
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'LSTM'
        n_epochs = [0, 15]
        directory = f'results/ice_results_20years_smaller_era5_transformer_8y'
    elif exp == 19:
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [15, 15, 20]
        directory = f'results/ice_results_20years_era5_6conv_noconv_newyears_actually6conv'
    elif exp == 20:
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [50]
        directory = f'results/ice_results_20years_era5_6conv_noconv_20yearsstraight_splitgconvlstm'
    elif exp == 21:
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [50]
        directory = f'results/ice_results_20years_era5_6conv_noconv_20yearsstraight_splitgconvlstm_30_input'
        input_timesteps = 30
    elif exp == 22:
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [50]
        directory = f'results/ice_results_20years_era5_6conv_noconv_20yearsstraight_splitgconvlstm_adam'
    elif exp == 23:
        convolution_type = 'TransformerConv'
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [35]
        directory = f'results/ice_results_20years_era5_3conv_noconv_20yearsstraight_splitgconvlstm_adam_transformer_decay'
        n_conv_layers = 3
    elif exp == 24:
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [50]
        lr = 0.0001
        directory = f'results/ice_results_20years_era5_3conv_noconv_20yearsstraight_splitgconvlstm_adam_decay_lr0001'
        n_conv_layers = 6
    elif exp == 25:
        convolution_type = 'TransformerConv'
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [35]
        lr = 0.0001
        directory = f'results/ice_results_20years_era5_3conv_noconv_20yearsstraight_splitgconvlstm_adam_transformer_decay_lr0001'
        n_conv_layers = 3
    elif exp == 26:
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [50]
        lr = 0.0001
        directory = f'results/ice_results_20years_era5_3conv_noconv_20yearsstraight_splitgconvlstm_adam_decay_lr0001_2decoders'
        n_conv_layers = 6
    elif exp == 27:
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [50]
        lr = 0.001
        directory = f'results/ice_results_20years_era5_3conv_noconv_20yearsstraight_splitgconvlstm_adam_decay_lr001_4decoders'
        n_conv_layers = 6
    elif exp == 28:
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [50]
        lr = 0.001
        directory = f'results/ice_results_20years_era5_3conv_noconv_20yearsstraight_splitgconvlstm_adam_nodecay_lr001_4decoders'
        n_conv_layers = 6
    elif exp == 29:
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [1]
        lr = 0.001
        directory = f'results/ice_results_20years_era5_3conv_noconv_20yearsstraight_splitgconvlstm_adam_nodecay_lr001_4decoders_1'
        n_conv_layers = 6
    elif exp == 30:
        multires_training = False
        preset_mesh = 'heterogeneous'
        rnn_type = 'NoConvLSTM'
        n_epochs = [35]
        lr = 0.001
        directory = f'results/ice_results_20years_era5_3conv_noconv_20yearsstraight_splitgconvlstm_adam_nodecay_lr001_4decoders_1'
        n_conv_layers = 3
        output_timesteps = 10
        
    use_edge_attrs = False if convolution_type == 'GCNConv' else True
        
    # -------------------------------------------

    # Full resolution dataset
    ds = xr.open_mfdataset(glob.glob('/home/zgoussea/scratch/ERA5_D/*.nc'))  # ln -s /home/zgoussea/scratch/ERA5_GLORYS data/ERA5_GLORYS
    mask = np.isnan(ds.siconc.isel(time=0)).values
    # high_interest_region = xr.open_dataset('data/shipping_corridors/primary_route_mask.nc').band_data.values
    high_interest_region = None

    image_shape = mask.shape
    # graph_structure = None

    if preset_mesh == 'heterogeneous':
        graph_structure = create_static_heterogeneous_graph(image_shape, 1, mask, high_interest_region=high_interest_region, use_edge_attrs=use_edge_attrs, resolution=1/12, device=device)
    elif preset_mesh == 'homogeneous':
        graph_structure = create_static_homogeneous_graph(image_shape, 1, mask, high_interest_region=high_interest_region, use_edge_attrs=use_edge_attrs, resolution=1/12, device=device)
    
    # Full resolution datasets
    data_train_1 = IceDataset(ds, training_years_1, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True, graph_structure=graph_structure, mask=mask, cache_dir=cache_dir)
    # data_train_2 = IceDataset(ds, training_years_2, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True, graph_structure=graph_structure, mask=mask, cache_dir=cache_dir)
    # data_train_3 = IceDataset(ds, training_years_3, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True, graph_structure=graph_structure, mask=mask, cache_dir=cache_dir)
    # data_train_4 = IceDataset(ds, training_years_4, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True, graph_structure=graph_structure, mask=mask, cache_dir=cache_dir)
    data_test = IceDataset(ds, range(training_years_4[-1]+1, training_years_4[-1]+1+2), month, input_timesteps, output_timesteps, x_vars, y_vars, graph_structure=graph_structure, mask=mask, cache_dir=cache_dir)
    data_val = IceDataset(ds, range(training_years_4[-1]+1+2+1-2, training_years_4[-1]+1+2+1+4), month, input_timesteps, output_timesteps, x_vars, y_vars, graph_structure=graph_structure, mask=mask, cache_dir=cache_dir)

    loader_train_1 = DataLoader(data_train_1, batch_size=1, shuffle=True, num_workers=2)
    # loader_train_2 = DataLoader(data_train_2, batch_size=1, shuffle=True)
    # loader_train_3 = DataLoader(data_train_3, batch_size=1, shuffle=True)
    # loader_train_4 = DataLoader(data_train_4, batch_size=1, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=1, shuffle=True, num_workers=2)
    loader_val = DataLoader(data_val, batch_size=1, shuffle=False, num_workers=2)

    climatology = ds[y_vars].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).to_array().values
    climatology = torch.tensor(np.nan_to_num(climatology)).to(device)
    climatology = torch.moveaxis(climatology, 0, -1)
    climatology = flatten(climatology, graph_structure['mapping'], graph_structure['n_pixels_per_node'])
    climatology = torch.moveaxis(climatology, -1, 0)

    # Set threshold 
    thresh = -np.inf  # 0.15
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
        n_conv_layers=n_conv_layers,
        rnn_type=rnn_type,
        convolution_type=convolution_type,
    )

    experiment_name = f'M{str(month)}_Y{training_years_1[0]}_Y{training_years_4[-1]}_I{input_timesteps}O{output_timesteps}'

    model = NextFramePredictorS2S(
        thresh=thresh,
        experiment_name=experiment_name,
        directory=directory,
        input_features=input_features,
        input_timesteps=input_timesteps,
        output_timesteps=output_timesteps,
        transform_func=dist_from_05,
        device=device,
        binary=binary,
        debug=False, 
        model_kwargs=model_kwargs)

    print('Num. parameters:', model.get_n_params())
    # print('Model:\n', model.model)

    model.model.train()

    # Train with full resolution. Use high interest region.
    model.train(
        loader_train_1,
        loader_test,
        climatology,
        lr=lr,
        n_epochs=n_epochs[0],
        mask=mask,
        truncated_backprop=truncated_backprop,
        graph_structure=graph_structure,
        )
    
    # model.train(
    #     loader_train_2,
    #     loader_test,
    #     climatology,
    #     lr=lr,
    #     n_epochs=n_epochs[1],
    #     mask=mask,
    #     truncated_backprop=truncated_backprop,
    #     graph_structure=graph_structure,
    #     ) 
 
    # model.train(
    #     loader_train_3,
    #     loader_test,
    #     climatology,
    #     lr=lr,
    #     n_epochs=n_epochs[2],
    #     mask=mask,
    #     truncated_backprop=truncated_backprop,
    #     graph_structure=graph_structure,
    #     ) 
  
    # model.train(
    #     loader_train_4,
    #     loader_test,
    #     climatology,
    #     lr=lr,
    #     n_epochs=n_epochs[1],
    #     mask=mask,
    #     truncated_backprop=truncated_backprop,
    #     graph_structure=graph_structure,
    #     ) 

    # Save model and losses
    model.loss.to_csv(f'{directory}/loss_{experiment_name}.csv')
    model.load(directory)  # Load best model
    
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
    
    if graph_structure is not None:
        y_true = torch.Tensor(loader_val.dataset.y)
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
    ds.to_netcdf(f'{directory}/valpredictions_{experiment_name}.nc')
    print(f'Finished model {month} in {((time.time() - start) / 60)} minutes')
