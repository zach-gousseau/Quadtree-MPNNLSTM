import matplotlib.pyplot as plt
import numpy as np
import netCDF4
import torch
import random
import datetime
import glob
import pandas as pd
import os
import seaborn as sns
import time
from calendar import monthrange, month_name
import xarray as xr
import rioxarray
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

import argparse
import re

from model.utils import normalize

from model.mpnnlstm import NextFramePredictorS2S
from model.seq2seq import Seq2Seq

from torch.utils.data import Dataset, DataLoader

from ice_dataset import IceDataset
from model.utils import int_to_datetime


def masked_accuracy_along_axis(mask):
    def loss(y_true, y_pred):
        return [accuracy_score(y_true[i, mask], y_pred[i, mask]) for i in range(y_true.shape[0])]
    return loss


mask = np.isnan(xr.open_dataset('data/ERA5_GLORYS/ERA5_GLORYS_1993.nc').siconc.isel(time=0)).values
results_dir = f'results/ice_results_20years_glorys_3conv_noconv_20yearsstraight_splitgconvlstm_adam_nodecay_lr001_1decoders_transformer_homo_multitask'
year_start, year_end, timestep_in, timestep_out = re.search(r'Y(\d+)_Y(\d+)_I(\d+)O(\d+)', glob.glob(results_dir+'/*.nc')[0]).groups()

timesteps = [30, 60]
months, ds = [], []
for month in range(1, 13):
    print(month)
    try:
        ds.append(xr.open_dataset(f'{results_dir}/valpredictions_M{month}_Y{year_start}_Y{year_end}_I{timestep_in}O{timestep_out}.nc', engine='netcdf4').sel(timestep=timesteps))
        months.append(month)
    except Exception as e: #FileNotFoundError:
        print(e)
        pass

ds = xr.concat(ds, dim='launch_date')
ds = ds.rio.set_crs(4326)

# mode = 'clim'
mode = 'model'

if mode == 'clim':
    climatology = xr.open_mfdataset(glob.glob('data/ERA5_GLORYS/*.nc'))
    # climatology = xr.open_mfdataset(glob.glob('/home/zgoussea/scratch/ERA5_D/*.nc'))
    climatology = climatology.sel(time=slice(datetime.datetime(1993, 1, 1), datetime.datetime(2014, 1, 1)))
    climatology = climatology['siconc'].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).values
    climatology = np.nan_to_num(climatology)
    arr_clim = np.array([[climatology[(doy.item()-1+i)%365, :, :] for i in timesteps] for doy in ds.launch_date.dt.dayofyear])
    ds['y_hat_sip'].values = arr_clim
    ds['y_hat_sip'] = ds['y_hat_sip'] < 0.5
else:
    ds['y_hat_sip'] = ds['y_hat_sip'] < 0.5

ds['y_true'] = ds['y_true'] < 0.15
ds = ds[['y_true', 'y_hat_sip']]

# ds = xr.open_dataset('/Users/zach/Documents/thesis-paper/ports/churchill.nc')
timestep = 60

correct_est = None
for year, ds_ in ds.sel(timestep=timestep).groupby('launch_date.year'):
    if year == 2013:
        continue
    ds_ = ds_.rolling(launch_date=15).construct('ts').all('ts')
    true_bu = ds_['y_true'].idxmax(dim='launch_date', skipna=True)
    est_bu = ds_['y_hat_sip'].idxmax(dim='launch_date', skipna=True)
    correct_est_year = abs(true_bu - est_bu).astype('float') <= 86400000000000*7
    if correct_est is None:
        correct_est = correct_est_year.astype(int)
    else: 
        correct_est += correct_est_year.astype(int)
    
    est = abs(true_bu - est_bu).astype('float') / 86400000000000#<= 86400000000000*7
    est.where(~mask).plot(vmin=0, vmax=14)
    plt.savefig(f'{results_dir}/est_bu_{year}_{timestep}_{mode}.png')
    plt.close()

correct_est = correct_est / 6
correct_est.where(~mask).plot(vmin=0, vmax=1)
plt.savefig(f'{results_dir}/bu_acc_{timestep}_{mode}.png')
plt.close()
correct_est.to_netcdf(f'{results_dir}/bu_acc_{timestep}_{mode}.nc')

    
correct_est = None
for year, ds_ in ds.sel(timestep=timestep).groupby('launch_date.year'):
    
    if year == 2013:
        continue
    
    # For freeze-up 
    ds_ = ds_.isel(launch_date=slice(244, None))
    ds_ = ~ds_
    
    ds_ = ds_.rolling(launch_date=15).construct('ts').all('ts')
    true_bu = ds_['y_true'].idxmax(dim='launch_date', skipna=True)
    est_bu = ds_['y_hat_sip'].idxmax(dim='launch_date', skipna=True)
    correct_est_year = abs(true_bu - est_bu).astype('float') <= 86400000000000*7
    if correct_est is None:
        correct_est = correct_est_year.astype(int)
    else: 
        correct_est += correct_est_year.astype(int)
    
    est = abs(true_bu - est_bu).astype('float') / 86400000000000#<= 86400000000000*7
    est.where(~mask).plot(vmin=0, vmax=14)
    plt.savefig(f'{results_dir}/est_fu_{year}_{timestep}_{mode}.png')
    plt.close()

correct_est = correct_est / 6
correct_est.where(~mask).plot(vmin=0, vmax=1)
plt.savefig(f'{results_dir}/fu_acc_{timestep}.png')
plt.close()
correct_est.to_netcdf(f'{results_dir}/fu_acc_{timestep}_{mode}.nc')