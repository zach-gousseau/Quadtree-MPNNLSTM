import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import datetime
import pandas as pd
import os
import seaborn as sns
import time
from calendar import monthrange, month_name
import xarray as xr
import rioxarray
from dateutil.relativedelta import relativedelta
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

import argparse

from utils import normalize

from mpnnlstm import NextFramePredictorS2S
from seq2seq import Seq2Seq

from torch.utils.data import Dataset, DataLoader

from ice_test import IceDataset
from utils import int_to_datetime

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

mask = np.isnan(xr.open_zarr('data/era5_hb_daily.zarr').siconc.isel(time=0)).values

results_dir = 'ice_results_gnn_out'

months = range(1, 13)
ds = []
for month in months:
    try:
        ds.append(xr.open_dataset(f'{results_dir}/valpredictions_M{month}_Y2011_Y2015_I5O30.nc', engine='netcdf4'))
    except Exception as e: #FileNotFoundError:
        print(e)
        pass
    
ds = xr.concat(ds, dim='launch_date')
ds = ds.rio.set_crs(4326)

ds['launch_date'] = [int_to_datetime(dt) for dt in ds.launch_date.values]


# LOSSES ----------------------------
months = range(1, 13)
losses = {}
for month in months:
    try:
        losses[month] = pd.read_csv(f'{results_dir}/loss_M{month}_Y2011_Y2015_I5O30.csv')
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
accuracy = False

heatmap = pd.DataFrame(0.0, index=range(1, 13), columns=ds.timestep)
heatmap_n = pd.DataFrame(0.0, index=range(1, 13), columns=ds.timestep)

for timestep in ds.timestep:
    timestep = int(timestep.values)
    for launch_date in ds.launch_date:
        arr = ds.sel(timestep=timestep, launch_date=launch_date).to_array().values
        
        if accuracy:
            arr = arr > 0.5
            err = masked_accuracy(~mask)(arr[0], arr[1])
        else:
            err = masked_RMSE(~mask)(arr[0], arr[1])
        
        launch_month = pd.Timestamp(launch_date.values).month
        
        heatmap[timestep][launch_month] += err
        heatmap_n[timestep][launch_month] += 1
        
heatmap = heatmap.div(heatmap_n)

plt.figure(dpi=80)
sns.heatmap(heatmap, yticklabels=[month_name[i][:3] for i in range(1, 13)], vmax=None, vmin=None)
plt.xlabel('Lead time (days)')
plt.savefig(f'{results_dir}/heatmap.png')
quit()
# Examples and GIF
for m in range(12):
    ld = 15 + m*30

    fns = []
    for ts in range(1, 30):
        
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        
        ds.sel(launch_date=ds.launch_date[ld], timestep=ts).where(~mask).y_true.plot(ax=axs[0], vmin=0, vmax=1)
        ds.sel(launch_date=ds.launch_date[ld], timestep=ts).where(~mask).y_hat.plot(ax=axs[1], vmin=0, vmax=1)
        axs[0].set_title(f'True ({str(ds.launch_date[ld].values)[:10]}, step {ts})')
        axs[1].set_title(f'Pred ({str(ds.launch_date[ld].values)[:10]}, step {ts})')
        plt.tight_layout()
        fn = f'{results_dir}/gif/{str(ds.launch_date[ld].values)[:10]}_{ts}.png'
        fns.append(fn)
        plt.savefig(fn)



    from PIL import Image
    frames = []
    for fn in fns:
        new_frame = Image.open(fn)
        frames.append(new_frame)

    frames[0].save(f'{results_dir}/gif/{str(ds.launch_date[ld].values)[:10]}.gif',
                format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=300,
                loop=0)