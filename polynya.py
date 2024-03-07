import netCDF4
import xarray as xr
import numpy as np
import pandas as pd
import glob
import rioxarray
import geopandas as gpd
import tqdm

kivaliq = gpd.read_file('data/kivaliq.gpkg')

# results_dir = f'results/ice_results_20years_glorys_3conv_noconv_20yearsstraight_splitgconvlstm_adam_nodecay_lr001_1decoders_transformer'
# results_dir = f'results/ice_results_20years_glorys_3conv_noconv_20yearsstraight_splitgconvlstm_adam_nodecay_lr001_1decoders'
results_dir = f'results/ice_results_20years_glorys_3conv_noconv_20yearsstraight_splitgconvlstm_adam_nodecay_lr001_1decoders_transformer_multitask'

mask = np.isnan(xr.open_dataset('data/ERA5_GLORYS/ERA5_GLORYS_1993.nc').siconc.isel(time=0)).values

ds = xr.open_mfdataset([results_dir + f'/valpredictions_M{m}_Y1993_Y2013_I10O90.nc' for m in [12, 1, 2, 3, 4, 5]], concat_dim='launch_date', combine='nested')
ds = ds.sortby('launch_date')

ds = ds.where(~mask)
ds = ds.rio.write_crs(4326)
ds = ds.rio.clip(kivaliq.geometry.values, kivaliq.crs)
# ds = (1-ds.sum(['latitude', 'longitude'])) / 2275
# ds['y_true'] = ds['y_true'] > 0.15
# ds['y_hat_sip'] = ds['y_hat_sip'] > 0.5
ds = (1-ds.sum(['latitude', 'longitude'])) / 2275
ds = ds.compute()


pred_openings = []
true_openings = []
for ld in tqdm.tqdm(ds.launch_date):
    ds_ = ds.sel(launch_date=ld)
    pred_opening = float(ds_.y_hat_sic.isel(timestep=7) - ds_.y_hat_sic.isel(timestep=0))
    true_opening = float(ds_.y_true.isel(timestep=7) - ds_.y_true.isel(timestep=0))
    
    # if abs(true_opening) > 0.05:
    #     pred_openings.append(pred_opening)
    #     true_openings.append(true_opening)
        
    pred_openings.append(pred_opening)
    true_openings.append(true_opening)
        
df = pd.DataFrame(dict(
    pred_opening=pred_openings,
    true_opening=true_openings
))

df.to_csv('scrap/polynya_openings_sip.csv')

        
    
    