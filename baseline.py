import netCDF4
import xarray as xr 
import numpy as np
import glob
import argparse

def masked_RMSE(mask):
    def loss(y_true, y_pred):
        sq_diff = ((y_pred - y_true)**2)[mask]
        return np.sqrt(np.mean(sq_diff))
    return loss

def masked_RMSE_along_axis(mask):
    def loss(y_true, y_pred):
        sq_diff = ((y_pred - y_true)**2)[:, mask]
        return np.sqrt(np.mean(sq_diff, (1)))
    return loss

ds_full = xr.open_mfdataset(glob.glob('/home/zgoussea/scratch/ERA5_D/*.nc'))
mask = np.isnan(ds_full.isel(time=0).siconc.values)

climatology = ds_full[['siconc']].fillna(0).groupby('time.dayofyear').mean('time', skipna=True).to_array().values[0]

# parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--month')  # Month number

# args = vars(parser.parse_args())
# month = int(args['month'])

ds_full = ds_full.rolling(time=90).construct('timestep').siconc
ds_t = ds_full.time.values[:-89]
ds_full = ds_full.isel(time=slice(89, None))
ds_full = ds_full.assign_coords(time=ds_t)

def get_preds(lam=0.05):
    weights = [np.e**(-lam * t) for t in range(ds.timestep.size)]
    y_pred = np.array([[y_true[i, ..., 0] * weights[t] + climatology[(doys[i] + t)%366-1] * (1-weights[t]) for t in range(ds.timestep.size)] for i in range(ds.time.size)])
    y_pred = np.moveaxis(y_pred, 1, -1)
    return y_pred

def evaluate(lam=0.05):
    y_pred = get_preds(lam)
    return masked_RMSE_along_axis(~mask)(y_true, y_pred).mean()

from scipy.optimize import minimize 

for month in range(1, 2):
    ds = ds_full.sel(time=(ds_full.time.dt.month==month))
    doys = ds.time.dt.dayofyear.values
    y_true = ds.values
    print(month)
    print(minimize(evaluate, 0.02))
