import xarray as xr
import numpy as np

import datetime
import matplotlib.pyplot as plt
from utils import int_to_datetime

import glob

ports = {
    'churchill': (58.7745, -94.1935),
    'inukjuak': (58.4514, -78.1351),
    'quaqtaq': (61.0442, -69.6421),
    # 'sanirajak': (),
}

# Read dataset
ds = xr.open_mfdataset(glob.glob('data/hb_era5_glorys_nc/*.nc'))
mask = np.isnan(ds.siconc.isel(time=0))

# Read results
results_dir = 'ice_results_may26_9_multires'

months = list(range(1, 13))
ds_pred = []
for month in months:
    try:
        ds_pred.append(xr.open_dataset(f'{results_dir}/valpredictions_M{month}_Y2007_Y2012_I10O90.nc', engine='netcdf4'))
    except Exception as e: #FileNotFoundError:
        print(e)
        pass
    
ds_pred = xr.concat(ds_pred, dim='launch_date')
ds_pred = ds_pred.rio.set_crs(4326)
ds_pred['launch_date'] = ds_pred['launch_date'] + 3600000000000*8
# ds_pred['launch_date'] = [int_to_datetime(dt) for dt in ds_pred.launch_date.values]

# Remove (2018, 1, 1)
ds = ds.isel(time=slice(0, -1))
ds_pred = ds_pred.isel(launch_date=slice(0, -1))

# Rename to have consistency between preds and observation
ds_pred = ds_pred.rename({'launch_date': 'time', 'y_hat': 'siconc'})

def get_breakup_date(ds, port):
    lat = float(ds.sel(latitude=ports[port][0], longitude=ports[port][1], method='nearest').coords['latitude'].values)
    lon = float(ds.sel(latitude=ports[port][0], longitude=ports[port][1], method='nearest').coords['longitude'].values)

    lat_i, lon_i = np.argwhere(ds.latitude.values==lat)[0][0], np.argwhere(ds.longitude.values==lon)[0][0]

    ds_window = ds.isel(latitude=slice(lat_i-5, lat_i+5), longitude=slice(lon_i-5, lon_i+5)).siconc
    mask_window = mask.isel(latitude=slice(lat_i-5, lat_i+5), longitude=slice(lon_i-5, lon_i+5))
    ds_window = ds_window.fillna(0).where(~mask_window)
    proportion_ice = ((ds_window > 0.15).sum(['latitude', 'longitude']) / np.sum(~mask_window))

    years = np.unique(ds.time.dt.year)
    
    breakup_dates = {}
    for year in years:
        breakup_window = proportion_ice.sel(time=slice(datetime.datetime(year, 5, 15), datetime.datetime(year, 7, 15)))
        breakup_date = breakup_window.time[np.argwhere(breakup_window.values>0.3)[-1]]
        
        breakup_dates[year] = breakup_date.values
    return breakup_dates

def get_freezeup_date(ds, port):
    lat = float(ds.sel(latitude=ports[port][0], longitude=ports[port][1], method='nearest').coords['latitude'].values)
    lon = float(ds.sel(latitude=ports[port][0], longitude=ports[port][1], method='nearest').coords['longitude'].values)

    lat_i, lon_i = np.argwhere(ds.latitude.values==lat)[0][0], np.argwhere(ds.longitude.values==lon)[0][0]

    ds_window = ds.isel(latitude=slice(lat_i-5, lat_i+5), longitude=slice(lon_i-5, lon_i+5)).siconc
    mask_window = mask.isel(latitude=slice(lat_i-5, lat_i+5), longitude=slice(lon_i-5, lon_i+5))
    ds_window = ds_window.fillna(0).where(~mask_window)
    proportion_ice = ((ds_window > 0.15).sum(['latitude', 'longitude']) / np.sum(~mask_window))
    
    years = np.unique(ds.time.dt.year)
    
    freezeup_dates = {}
    for year in years:
        freezeup_window = proportion_ice.sel(time=slice(datetime.datetime(year, 10, 15), datetime.datetime(year, 12, 15)))
        freezeup_date = freezeup_window.time[np.argwhere(freezeup_window.values>0.3)[0]]
        
        freezeup_dates[year] = freezeup_date.values
    return freezeup_dates


ds_pred_15 = ds_pred.sel(timestep=15)
ds_pred_30 = ds_pred.sel(timestep=30)

get_breakup_date(ds_pred_15, 'churchill')
get_freezeup_date(ds_pred_15, 'churchill')