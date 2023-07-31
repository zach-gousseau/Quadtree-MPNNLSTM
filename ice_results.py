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
        sq_diff = np.multiply((y_pred - y_true)**2, mask)
        return np.sqrt(np.mean(sq_diff, (1, 2)))
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

mask = np.isnan(xr.open_mfdataset(glob.glob('data/ERA5_GLORYS/*.nc')).siconc.isel(time=0)).values
# mask = np.isnan(xr.open_mfdataset(glob.glob('data/ERA5_GLORYS/*.nc')).isel(latitude=slice(175, 275), longitude=slice(125, 225)).siconc.isel(time=0)).values

results_dir = f'results/ice_results_20years_small'
accuracy = False

year_start, year_end, timestep_in, timestep_out = re.search(r'Y(\d+)_Y(\d+)_I(\d+)O(\d+)', glob.glob(results_dir+'/*.nc')[0]).groups()

months, ds = [], []
for month in range(1, 13):
    print(month)
    try:
        ds.append(xr.open_dataset(f'{results_dir}/valpredictions_M{month}_Y{year_start}_Y{year_end}_I{timestep_in}O{timestep_out}.nc', engine='netcdf4').astype('float16'))
        months.append(month)
    except Exception as e: #FileNotFoundError:
        print(e)
        pass

ds = xr.concat(ds, dim='launch_date')
ds = ds.rio.set_crs(4326)
ds['launch_date'] = [round_to_day(pd.Timestamp(dt)) + datetime.timedelta(days=1) for dt in ds.launch_date.values]
image_shape = mask.shape


# graph_structure = create_static_heterogeneous_graph(image_shape, 4, mask, use_edge_attrs=True, resolution=1/12)
graph_structure = create_static_heterogeneous_graph(image_shape, 4, mask, high_interest_region=None, use_edge_attrs=False, resolution=1/12, device=None)
# graph_structure = create_static_homogeneous_graph(image_shape, 4, mask, use_edge_attrs=True, resolution=1/12)

num_timesteps = ds.timestep.size

# GIF 
if not os.path.exists(f'{results_dir}/gif'):
    os.makedirs(f'{results_dir}/gif')

generate_gif = False
year = int(ds.launch_date.dt.year.values[0])
if generate_gif:
    ld = 15

    for month in months:
        fns = []
        for ts in range(1, 91):
            
            fig, axs = plt.subplots(1, 2, figsize=(8, 3))
            
            ds.sel(launch_date=datetime.datetime(year, month, 15), timestep=ts).where(~mask).y_true.plot(ax=axs[0], vmin=0, vmax=1)
            ds.sel(launch_date=datetime.datetime(year, month, 15), timestep=ts).where(~mask).y_hat.plot(ax=axs[1], vmin=0, vmax=1)
            axs[0].set_title(f'True ({str(datetime.datetime(year, month, 15))[:10]}, step {ts})')
            axs[1].set_title(f'Pred ({str(datetime.datetime(year, month, 15))[:10]}, step {ts})')
            plt.tight_layout()
            fn = f'{results_dir}/gif/{str(datetime.datetime(year, month, 15))[:10]}_{ts}.png'
            fns.append(fn)
            plt.savefig(fn)
            plt.close()



        from PIL import Image
        frames = []
        for fn in fns:
            new_frame = Image.open(fn)
            frames.append(new_frame)

        frames[0].save(f'{results_dir}/gif/{str(datetime.datetime(year, month, 15))[:10]}.gif',
                    format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=300,
                    loop=0)

        for fn in fns:
            os.remove(fn)

# quit()

# LOSSES ----------------------------
months = range(1, 13)
losses = {}
for month in months:
    try:
        losses[month] = pd.read_csv(f'{results_dir}/loss_M{month}_Y{year_start}_Y{year_end}_I{timestep_in}O{timestep_out}.csv')
    except FileNotFoundError:
        pass

fig, axs = plt.subplots(3, 4, figsize=(14, 6))
for i, month in enumerate(months):
    try:
        axs.flatten()[i].plot(losses[month].train_loss, label='train')
        axs.flatten()[i].plot(losses[month].test_loss, label='test')
        axs.flatten()[i].legend()
        
        axs.flatten()[i].set_ylabel('Loss (MSE)')
        axs.flatten()[i].set_xlabel('Epoch')
        axs.flatten()[i].set_title(month_name[month][:3])
    except KeyError:
        pass
    
plt.tight_layout()
plt.savefig(f'{results_dir}/losses.png')


# HEATMAP ----------------------

heatmap = create_heatmap_fast(ds)

plt.figure(dpi=80)
sns.heatmap(heatmap, yticklabels=[month_name[i][:3] for i in range(1, 13)], vmax=0.18, vmin=0.02)
plt.xlabel('Lead time (days)')
plt.savefig(f'{results_dir}/heatmap.png')
plt.close()


# ds['y_hat'].values
arr_pers = np.swapaxes(xr.concat([ds.y_true.isel(timestep=0) for _ in range(num_timesteps)], dim='timestep').values, 0, 1)
shape_ = arr_pers.shape
arr_pers = flatten_unflatten(torch.Tensor(arr_pers.reshape((-1, *image_shape, 1))), graph_structure, mask)
arr_pers = arr_pers.reshape(shape_)
ds['y_hat'].values = arr_pers
heatmap_pers = create_heatmap_fast(ds)

plt.figure(dpi=80)
sns.heatmap(heatmap_pers, yticklabels=[month_name[i][:3] for i in range(1, 13)], vmax=0.18, vmin=0.02)
plt.xlabel('Lead time (days)')
plt.savefig(f'{results_dir}/heatmap_pers.png')
plt.close()

climatology = xr.open_mfdataset(glob.glob('data/ERA5_GLORYS/*.nc'))
# climatology = climatology.isel(latitude=slice(175, 275), longitude=slice(125, 225))
climatology = climatology['siconc'].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).values
climatology = np.nan_to_num(climatology)

arr_clim = np.array([[climatology[(doy-1+i)%365] for i in range(num_timesteps)] for doy in ds.launch_date.dt.dayofyear])
shape_ = arr_clim.shape
arr_clim = flatten_unflatten(torch.Tensor(arr_clim.reshape((-1, *image_shape, 1))), graph_structure, mask)
arr_clim = arr_clim.reshape(shape_)
ds['y_hat'].values = arr_clim
heatmap_clim = create_heatmap_fast(ds)

plt.figure(dpi=80)
sns.heatmap(heatmap_clim, yticklabels=[month_name[i][:3] for i in range(1, 13)], vmax=0.18, vmin=0.02)
plt.xlabel('Lead time (days)')
plt.savefig(f'{results_dir}/heatmap_clim.png')
plt.close()


plt.figure(dpi=80)
sns.heatmap((heatmap - heatmap_clim), yticklabels=[month_name[i][:3] for i in range(1, 13)], cmap='coolwarm', center=0)
plt.title('Blue -> Model outperforms climatology')
plt.xlabel('Lead time (days)')
plt.savefig(f'{results_dir}/heatmap_diff_clim.png')
plt.close()

plt.figure(dpi=80)
sns.heatmap((heatmap - heatmap_pers), yticklabels=[month_name[i][:3] for i in range(1, 13)], cmap='coolwarm', center=0, vmin=-0.05, vmax=0.05)
plt.title('Blue -> Model outperforms persistence')
plt.xlabel('Lead time (days)')
plt.savefig(f'{results_dir}/heatmap_diff_pers.png')
plt.close()

heatmap.to_csv(f'{results_dir}/heatmap.csv')
heatmap_clim.to_csv(f'{results_dir}/heatmap_clim.csv')
heatmap_pers.to_csv(f'{results_dir}/heatmap_pers.csv')