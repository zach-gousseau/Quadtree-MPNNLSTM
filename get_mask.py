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
    hidden_size = 32
    n_layers = 1
    kernel_size = 3
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
        xr.DataArray(mask, dims=('latitude','longitude'), coords={'latitude': ds.latitude, 'longitude': ds.longitude}).to_dataset(name='mask').to_netcdf('/home/zgoussea/scratch/mask.nc')
