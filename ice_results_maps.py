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

from model.graph_functions import create_static_heterogeneous_graph, create_static_homogeneous_graph, flatten, unflatten

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

def masked_accuracy_along_axis(mask):
    def loss(y_true, y_pred):
        return [accuracy_score(y_true[i, mask], y_pred[i, mask]) for i in range(y_true.shape[0])]
    return loss


def round_to_day(dt):
    return datetime.datetime(*dt.timetuple()[:3])

def flatten_unflatten(arr, graph_structure, mask):
    arr = flatten(arr, graph_structure['mapping'], graph_structure['n_pixels_per_node'], mask=~mask)
    arr = unflatten(arr, graph_structure['mapping'], mask.shape, mask=~mask)
    return arr
mask = np.isnan(xr.open_dataset('data/ERA5_GLORYS/ERA5_GLORYS_1993.nc').siconc.isel(time=0)).values
results_dir = f'results/ice_results_20years_glorys_3conv_noconv_20yearsstraight_splitgconvlstm_adam_nodecay_lr001_1decoders_transformer_multitask'

accuracy = False

year_start, year_end, timestep_in, timestep_out = re.search(r'Y(\d+)_Y(\d+)_I(\d+)O(\d+)', glob.glob(results_dir+'/*.nc')[0]).groups()

timesteps = [15, 30, 60, 90]

months, ds = [], []
for month in range(1, 13):
    print(month)
    try:
        ds.append(xr.open_dataset(f'{results_dir}/valpredictions_M{month}_Y{year_start}_Y{year_end}_I{timestep_in}O{timestep_out}.nc', engine='netcdf4').sel(timestep=(timesteps)))
        months.append(month)
    except Exception as e: #FileNotFoundError:
        print(e)
        pass

ds = xr.concat(ds, dim='launch_date')
ds = ds.rio.set_crs(4326)

ds['launch_date'] = [round_to_day(pd.Timestamp(dt)) + datetime.timedelta(days=1) for dt in ds.launch_date.values]

# fc_month = np.array([(ds.launch_date + 86400000000000 * timestep).dt.month for timestep in timesteps])
# ds = ds.assign_coords(fc_month=(('timestep', 'launch_date'), fc_month))

image_shape = mask.shape

num_timesteps = ds.timestep.size

sq_diff = (ds.y_hat_sic - ds.y_true)**2
rmses = np.sqrt(sq_diff.mean('launch_date'))
rmses.to_netcdf(f'{results_dir}/rmse_map.nc')

# rmses = [[] for _ in range(12)]
# for timestep in timesteps:
#     for month, ds_month in ds.sel(timestep=timestep).groupby('fc_month'):
#         sq_diff = (ds_month.y_hat_sic - ds_month.y_true)**2
#         rmses[month-1].append(np.sqrt(sq_diff.mean('launch_date')))
        
rmses = [[] for _ in range(12)]
for timestep in timesteps:
    for month, ds_month in ds.sel(timestep=timestep).groupby('launch_date.month'):
        sq_diff = (ds_month.y_hat_sic - ds_month.y_true)**2
        rmses[month-1].append(np.sqrt(sq_diff.mean('launch_date')))
        
for i in range(12):
    xr.concat(rmses[i], 'timestep').to_netcdf(f'{results_dir}/rmse_map_{i+1}.nc')



# climatology = xr.open_mfdataset(glob.glob('data/ERA5_GLORYS/*.nc'))
# # climatology = xr.open_mfdataset(glob.glob('/home/zgoussea/scratch/ERA5_D/*.nc'))
# climatology = climatology.sel(time=slice(datetime.datetime(1993, 1, 1), datetime.datetime(2014, 1, 1)))
# climatology = climatology['siconc'].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).values
# climatology = np.nan_to_num(climatology)
# climatology = flatten_unflatten(torch.Tensor(climatology.reshape((-1, *image_shape, 1))), graph_structure, mask)

# arr_clim = np.array([[climatology[(doy.item()-1+i)%365, :, :, 0].numpy() for i in range(num_timesteps)] for doy in ds.launch_date.dt.dayofyear])
# ds['y_hat_sip'].values = arr_clim



# arr_pers = flatten_unflatten(torch.Tensor(ds.y_true.isel(timestep=0).values).unsqueeze(-1), graph_structure, mask)
# arr_pers = arr_pers.type(torch.float16)
# arr_pers = np.array(arr_pers.repeat(1, 1, 1, 90))
# arr_pers = np.moveaxis(arr_pers, -1, 1)
# arr_pers = np.concatenate([arr_pers[[0]], arr_pers[:-1]])  # Shift one forward
# ds['y_hat_sip'].values = arr_pers
