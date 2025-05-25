import numpy as np
import torch
import random
import os
import time
import glob
import pandas as pd
import xarray as xr
import argparse
from torch.utils.data import Dataset, DataLoader

from model.utils import int_to_datetime
from model.cnnlstm import SimpleCNNLSTMPredictor

from ice_dataset import IceDataset


class SyntheticIceDataset(Dataset):
    """Synthetic dataset with the same interface as IceDataset but using random data"""
    def __init__(self, input_timesteps, output_timesteps, image_size=(32, 32), n_samples=20):
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.image_size = image_size
        self.n_samples = n_samples
        self.channels = 5  # Same as len(x_vars) in the real dataset
        
        # Generate launch dates (just for compatibility)
        self.launch_dates = np.array([int(time.time() * 1000) + i * 86400000 for i in range(n_samples)])
        
        # Generate random data
        self.x = torch.rand(n_samples, input_timesteps, self.channels, *image_size)
        self.y = torch.rand(n_samples, output_timesteps, 1, *image_size) 
        
        # Create a synthetic mask to simulate land/water
        self.mask = torch.zeros(*image_size, dtype=torch.bool)
        # Add some "land" areas
        self.mask[0:5, 0:5] = True
        self.mask[-5:, -5:] = True
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], torch.tensor(self.launch_dates[idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true', help="Use synthetic data to test the pipeline")
    parser.add_argument('-m', '--month', type=int, default=1, help="Month to run the experiment for")
    args = parser.parse_args()

    np.random.seed(21)
    random.seed(21)
    torch.manual_seed(21)

    start = time.time()
    
    cache_dir='/home/zgoussea/scratch/data_cache/'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # Model parameters
    month = args.month  # January
    lr = 0.001
    training_years = range(1994, 2014)
    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf']
    y_vars = ['siconc']
    input_features = len(x_vars)
    input_timesteps = 10
    output_timesteps = 90
    hidden_size = 64
    n_layers = 1
    kernel_size = 5
    dropout = 0.1
    binary = False

    if args.synthetic:
        print("Using synthetic data to test the pipeline")
        # Use small image size for faster testing
        image_size = (16, 16)
        
        output_timesteps = 7
        input_timesteps = 3
        
        # Create synthetic datasets
        data_train = SyntheticIceDataset(input_timesteps, output_timesteps, image_size, n_samples=5)
        data_test = SyntheticIceDataset(input_timesteps, output_timesteps, image_size, n_samples=2)
        data_val = SyntheticIceDataset(input_timesteps, output_timesteps, image_size, n_samples=2)
        
        # Use synthetic mask
        mask = data_train.mask
        
        # Reduce epochs for faster testing
        n_epochs = 2
    else:
        # Load real dataset
        ds = xr.open_mfdataset(glob.glob('/home/zgoussea/scratch/ERA5_GLORYS/*.nc'))
        mask = np.isnan(ds.siconc.isel(time=0)).values

        # Create datasets
        data_train = IceDataset(ds, training_years, month, input_timesteps, output_timesteps, x_vars, y_vars, train=True, cache_dir=cache_dir)
        data_test = IceDataset(ds, [training_years[-1]+1], month, input_timesteps, output_timesteps, x_vars, y_vars, cache_dir=cache_dir)
        data_val = IceDataset(ds, range(training_years[-1]+2, training_years[-1]+2+4), month, input_timesteps, output_timesteps, x_vars, y_vars, cache_dir=cache_dir)

        n_epochs = 30

    # Create dataloaders
    loader_train = DataLoader(data_train, batch_size=1, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=1, shuffle=True)
    loader_val = DataLoader(data_val, batch_size=1, shuffle=False)

    # Create experiment name
    experiment_name = f'CNNLSTM_M{str(month)}_Y{training_years[0]}_Y{training_years[-1]}_I{input_timesteps}O{output_timesteps}'
    if args.synthetic:
        experiment_name = 'Synthetic_' + experiment_name

    # Initialize model
    model = SimpleCNNLSTMPredictor(
        experiment_name=experiment_name,
        input_features=input_features,
        hidden_size=hidden_size,
        input_timesteps=input_timesteps,
        output_timesteps=output_timesteps,
        n_layers=n_layers,
        dropout=dropout,
        kernel_size=kernel_size,
        binary=binary,
        device=device)

    print('Num. parameters:', model.get_n_params())
    print('Model:\n', model.model)

    # Train the model
    model.train(
        loader_train,
        loader_test,
        lr=lr,
        n_epochs=n_epochs,
    )

    # Save model and losses
    results_dir = '/home/zgoussea/projects/def-ka3scott/zgoussea/Quadtree-MPNNLSTM/results/cnnlstm_64_1_layer'
    if args.synthetic:
        results_dir = 'synthetic_' + results_dir

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model.loss.to_csv(f'{results_dir}/loss_{experiment_name}.csv')
    model.save(results_dir)
    
    # Generate predictions
    model.model.eval()
    val_preds = model.predict(
        loader_val,
        mask=mask,
    )
    
    # Save results
    launch_dates = [int_to_datetime(t) for t in loader_val.dataset.launch_dates]
    
    if args.synthetic:
        # For synthetic data, create a simple xarray dataset
        ds_output = xr.Dataset(
            data_vars=dict(
                y_hat=(["launch_date", "timestep", "latitude", "longitude"], val_preds.squeeze()),
                y_true=(["launch_date", "timestep", "latitude", "longitude"], data_val.y.squeeze()),
            ),
            coords=dict(
                longitude=np.arange(data_val.image_size[1]),
                latitude=np.arange(data_val.image_size[0]),
                launch_date=launch_dates,
                timestep=np.arange(1, output_timesteps+1),
            ),
        )
    else:
        try:
            ds_output = xr.Dataset(
                data_vars=dict(
                    y_hat=(["launch_date", "timestep", "latitude", "longitude"], val_preds.squeeze()),
                    y_true=(["launch_date", "timestep", "latitude", "longitude"], loader_val.dataset.y.squeeze()),
                ),
                coords=dict(
                    longitude=ds.longitude,
                    latitude=ds.latitude,
                    launch_date=launch_dates,
                    timestep=np.arange(1, output_timesteps+1),
                ),
            )
        except Exception as e:
            print(f'Error creating xarray dataset: {e}')
            # just write y_hat, y_true and launch_dates as numpy arrays
            np.save(f'{results_dir}/valpredictions_{experiment_name}.npy', val_preds)
            np.save(f'{results_dir}/valtrue_{experiment_name}.npy', loader_val.dataset.y)
            np.save(f'{results_dir}/vallaunchdates_{experiment_name}.npy', launch_dates)
    
    ds_output.to_netcdf(f'{results_dir}/valpredictions_{experiment_name}.nc')
    print(f'Finished model {month} in {((time.time() - start) / 60)} minutes') 
