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

from model.cnn_mpnnlstm import NextFramePredictorCNNS2S
from model.cnn_seq2seq import CNNSeq2Seq

from torch.utils.data import Dataset, DataLoader

from cnn_ice_dataset import CNNIceDataset


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
    lr = 0.001
    truncated_backprop = 0

    training_years = range(1994, 2014)
    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf', 'usi', 'vsi', 'sithick']
    y_vars = ['siconc']
    input_features = len(x_vars)
    input_timesteps = 10
    output_timesteps= 90
    rnn_type = 'LSTM'
    
    cache_dir='/home/zgoussea/scratch/data_cache/'

    binary=False
    
    n_epochs = 50
        
    # -------------------------------------------

    # Full resolution dataset
    ds = xr.open_mfdataset(glob.glob('/home/zgoussea/scratch/ERA5_GLORYS/*.nc'))
    mask = np.isnan(ds.siconc.isel(time=0)).values
    high_interest_region = None

    image_shape = mask.shape
    
    # CNN datasets - no graph structure needed
    data_train = CNNIceDataset(ds, training_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True, mask=mask, cache_dir=cache_dir)    
    data_test = CNNIceDataset(ds, [training_years[-1]+1], month, input_timesteps, output_timesteps, x_vars, y_vars, mask=mask, cache_dir=cache_dir)    
    data_val = CNNIceDataset(ds, range(training_years[-1]+2, training_years[-1]+2+4), month, input_timesteps, output_timesteps, x_vars, y_vars, mask=mask, cache_dir=cache_dir)
    
    loader_train = DataLoader(data_train, batch_size=1, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=1, shuffle=True)
    loader_val = DataLoader(data_val, batch_size=1, shuffle=False)

    # Climatology for CNN (no flattening needed)
    climatology = ds[y_vars].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).to_array().values
    climatology = torch.tensor(np.nan_to_num(climatology)).to(device)
    # Shape is (variables, height, width, days) -> (height, width, days)
    climatology = climatology.squeeze(0)  # Remove variable dimension since y_vars has only 1 variable
    
    # The climatology from xarray groupby has shape (days, height, width)
    # We need to transpose it to (height, width, days)
    if len(climatology.shape) == 3 and climatology.shape[0] == 366:  # 366 days in a year
        print(f"Transposing climatology from {climatology.shape} to (height, width, days)")
        climatology = climatology.permute(1, 2, 0)  # (days, height, width) -> (height, width, days)
        print(f"Final climatology shape: {climatology.shape}")
    
    # Now shape is (height, width, days) which is what get_climatology_array expects

    # Arguments passed to CNNSeq2Seq constructor
    model_kwargs = dict(
        hidden_size=32,
        dropout=0.1,
        n_layers=1,
        dummy=False,
        n_conv_layers=3,
        rnn_type=rnn_type,
        kernel_size=3,
        padding=1,
        multitask=False,  # Only predicting siconc, not multitask
    )

    experiment_name = f'CNN_M{str(month)}_Y{training_years[0]}_Y{training_years[-1]}_I{input_timesteps}O{output_timesteps}'
    
    # Save model and losses    
    results_dir = '/home/zgoussea/projects/def-ka3scott/zgoussea/Quadtree-MPNNLSTM/results/cnn_new'   

    model = NextFramePredictorCNNS2S(
        experiment_name=experiment_name,
        directory=results_dir,
        input_features=input_features,
        input_timesteps=input_timesteps,
        output_timesteps=output_timesteps,
        device=device,
        binary=binary,
        debug=False, 
        model_kwargs=model_kwargs)

    print('Num. parameters:', model.get_n_params())

    model.model.train()

    # Train with full resolution
    model.train(    
        loader_train,    
        loader_test,    
        climatology,    
        lr=lr,    
        n_epochs=n_epochs,    
        mask=mask,    
        truncated_backprop=truncated_backprop,    
        )       

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
    
    # For CNN, y_true is already in the right format
    y_true = loader_val.dataset.y

    # Debug shapes
    print(f"val_preds shape: {val_preds.shape}")
    print(f"y_true shape: {y_true.shape}")
    print(f"val_preds type: {type(val_preds)}")
    print(f"y_true type: {type(y_true)}")

    try:
        # Try to create xarray Dataset
        # val_preds: (num_samples, timesteps, height, width, channels)
        # y_true: (num_samples, output_timesteps, height, width, channels)
        
        # No need to squeeze batch dimension anymore since it's handled in predict method
        val_preds_squeezed = val_preds
        
        # Squeeze channel dimension if it's 1
        if val_preds_squeezed.shape[-1] == 1:
            val_preds_squeezed = val_preds_squeezed.squeeze(-1)
        if y_true.shape[-1] == 1:
            y_true_squeezed = y_true.squeeze(-1)
        else:
            y_true_squeezed = y_true
        
        print(f"After processing - val_preds_squeezed shape: {val_preds_squeezed.shape}")
        print(f"After processing - y_true_squeezed shape: {y_true_squeezed.shape}")
        
        ds_results = xr.Dataset(
            data_vars=dict(
                y_hat=(["launch_date", "timestep", "latitude", "longitude"], val_preds_squeezed.astype('float')),
                y_true=(["launch_date", "timestep", "latitude", "longitude"], y_true_squeezed.astype('float')),
            ),
            coords=dict(
                longitude=ds.longitude,
                latitude=ds.latitude,
                launch_date=launch_dates,
                timestep=np.arange(1, output_timesteps+1),
            ),
        )
        ds_results.to_netcdf(f'{results_dir}/valpredictions_{experiment_name}.nc')
        print("Successfully saved xarray Dataset")
        
    except Exception as e:
        print(f"Failed to create xarray Dataset: {e}")
        print("Saving as numpy arrays instead...")
        
        # Save as numpy arrays
        np.save(f'{results_dir}/val_preds_{experiment_name}.npy', val_preds)
        np.save(f'{results_dir}/y_true_{experiment_name}.npy', y_true)
        np.save(f'{results_dir}/launch_dates_{experiment_name}.npy', np.array(launch_dates))
        
        # Also save metadata
        metadata = {
            'val_preds_shape': val_preds.shape,
            'y_true_shape': y_true.shape,
            'output_timesteps': output_timesteps,
            'experiment_name': experiment_name
        }
        
        import pickle
        with open(f'{results_dir}/metadata_{experiment_name}.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print("Saved as numpy arrays with metadata")

    print(f'Finished CNN model {month} in {((time.time() - start) / 60)} minutes') 