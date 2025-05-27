import matplotlib.pyplot as plt
import numpy as np
import netCDF4
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

import re

from model.graph_functions import flatten, unflatten

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

def create_heatmap_fast(ds, accuracy=True):
    timestep_values = ds.timestep.values.astype(int)
    launch_date_values = ds.launch_date.values
    launch_months = pd.DatetimeIndex(launch_date_values).month
    heatmap = np.zeros((12, len(timestep_values)))
    heatmap_n = np.zeros_like(heatmap)
    
    for i, timestep in enumerate(tqdm(timestep_values)):
        arr = ds.sel(timestep=timestep).to_array().values
        arr = np.nan_to_num(arr)
        
        if accuracy:
            arr[0] = arr[0] > 0.15
            arr[1] = arr[1] > 0.5
            err = masked_accuracy_along_axis(~mask)(arr[0], arr[1])
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


mask = np.isnan(xr.open_dataset('/home/zgoussea/scratch/ERA5_GLORYS/ERA5_GLORYS_1994.nc').siconc.isel(time=0)).values

results_dir = '/home/zgoussea/projects/def-ka3scott/zgoussea/Quadtree-MPNNLSTM/results'
results_dir = f'{results_dir}/cnn_new'

accuracy = False

year_start, year_end, timestep_in, timestep_out = re.search(r'Y(\d+)_Y(\d+)_I(\d+)O(\d+)', glob.glob(results_dir+'/*.nc')[0]).groups()

months, ds = [], []
for month in range(1, 13):
    print(month)
    try:
        ds.append(xr.open_dataset(f'{results_dir}/valpredictions_CNN_M{month}_Y{year_start}_Y{year_end}_I{timestep_in}O{timestep_out}.nc', engine='netcdf4').isel(launch_date=slice(0, 365)).astype('float16'))
        months.append(month)
    except Exception as e: #FileNotFoundError:
        print(e)
        pass

ds = xr.concat(ds, dim='launch_date')
ds = ds.rio.set_crs(4326)

ds['launch_date'] = [round_to_day(pd.Timestamp(dt)) + datetime.timedelta(days=1) for dt in ds.launch_date.values]
image_shape = mask.shape

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
        # arr = []
        for ts in range(1, 91):
            fig, axs = plt.subplots(1, 2, figsize=(8, 3))
            (ds.sel(launch_date=datetime.datetime(year, month, 15), timestep=ts).where(~mask).y_true).plot(ax=axs[0], vmin=0, vmax=1)
            (ds.sel(launch_date=datetime.datetime(year, month, 15), timestep=ts).where(~mask).y_hat_sic).plot(ax=axs[1], vmin=0, vmax=1)
            axs[0].set_title(f'True ({str(datetime.datetime(year, month, 15))[:10]}, step {ts})')
            axs[1].set_title(f'Pred ({str(datetime.datetime(year, month, 15))[:10]}, step {ts})')
            plt.tight_layout()
            fn = f'/home/zgoussea/scratch/gif/{str(datetime.datetime(year, month, 15))[:10]}_{ts}.png'
            fns.append(fn)
            plt.savefig(fn)
            plt.close()
            # arr.append(ds.sel(launch_date=datetime.datetime(year, month, 15), timestep=ts).where(~mask).to_array().values)
        # with open(f'{results_dir}/gif/{str(datetime.datetime(year, month, 15))[:10]}.npy', 'wb') as f:
        #     for a in arr:
        #         np.save(f, a)
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
            

# HEATMAP ----------------------

heatmap = create_heatmap_fast(ds[['y_true', 'y_hat']], False)
heatmap.to_csv(f'{results_dir}/heatmap.csv')

plt.figure(dpi=80)
sns.heatmap(heatmap, yticklabels=[month_name[i][:3] for i in range(1, 13)], vmax=0.28, vmin=0.02)
plt.xlabel('Lead time (days)')
plt.savefig(f'{results_dir}/heatmap.png')
plt.close()


# #COMPARE
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from calendar import month_name
# results_dir1 = f'/home/zgoussea/projects/def-ka3scott/zgoussea/Quadtree-MPNNLSTM/results/cnnlstm'
# results_dir2 = f'/home/zgoussea/projects/def-ka3scott/zgoussea/Quadtree-MPNNLSTM/results/transformer'
# descriptor1 = 'CNNLSTM'
# descriptor2 = 'GraphSIFNET'
# heatmap1 = pd.read_csv(results_dir1 + '/heatmap.csv', index_col=0)
# heatmap2 = pd.read_csv(results_dir2 + '/heatmap.csv', index_col=0)

# plt.figure(dpi=80)
# sns.heatmap((heatmap2 - heatmap1), yticklabels=[month_name[i][:3] for i in range(1, 13)], cmap='coolwarm', center=0, vmin=-0.05, vmax=0.05)
# plt.title(f'Blue -> {descriptor2} outperforms {descriptor1}')
# plt.xlabel('Lead time (days)')
# plt.savefig(f'{results_dir2}/heatmap_diff_compare_{descriptor1}_{descriptor2}.png')
# plt.close()


