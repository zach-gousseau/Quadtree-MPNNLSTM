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
results_dir = f'results/ice_results_20years_glorys_3conv_noconv_20yearsstraight_splitgconvlstm_adam_nodecay_lr001_1decoders_transformer_multitask'
year_start, year_end, timestep_in, timestep_out = re.search(r'Y(\d+)_Y(\d+)_I(\d+)O(\d+)', glob.glob(results_dir+'/*.nc')[0]).groups()

timesteps = np.arange(10, 91, 5)
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

# Apply thresholds
y_hat_sic_binary = (ds['y_hat_sic'] >= 0.15)
y_hat_sip_binary = (ds['y_hat_sip'] >= 0)

# Calculate true/false positives/negatives
TP = ((y_hat_sic_binary == 1) & (y_hat_sip_binary == 1)).astype(int)
TN = ((y_hat_sic_binary == 0) & (y_hat_sip_binary == 0)).astype(int)
FP = ((y_hat_sic_binary == 1) & (y_hat_sip_binary == 0)).astype(int)
FN = ((y_hat_sic_binary == 0) & (y_hat_sip_binary == 1)).astype(int)
import gc
del ds
del y_hat_sic_binary
del y_hat_sip_binary
gc.collect()

# Calculate binary accuracy
tptn = TP + TN
del TP
del TN
gc.collect()
fpfn = FP + FN
del FP 
del FN
gc.collect()
binary_accuracy = tptn / (tptn + fpfn)

# Calculate mean binary accuracy along the 'launch_date' axis
mean_binary_accuracy = binary_accuracy.mean(dim='launch_date')

fig, axs = plt.subplots(1, 3, figsize=(18, 4))
mean_binary_accuracy.sel(timestep=30).plot(ax=axs[0])
mean_binary_accuracy.sel(timestep=60).plot(ax=axs[1])
mean_binary_accuracy.sel(timestep=90).plot(ax=axs[2])
plt.savefig('scrap/sip_sic.png')

plt.figure(figsize=(8, 2))
mean_binary_accuracy.where(~mask).mean(['latitude', 'longitude']).plot()
plt.xlabel('Lead time (days)')
plt.ylabel('Accuracy')
plt.savefig('scrap/sip_sic_ts.png')