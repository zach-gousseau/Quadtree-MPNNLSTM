import netCDF4
import xarray as xr
import dask
import os
import tqdm
import datetime
import glob

from glorys import get_glorys

"""
Combine ERA5 atmospheric data (stored locally) with GLORYS12 SIC data (from CMEMS online archive)
"""

# Set CMEMS credentials (https://data.marine.copernicus.eu/register)
CMEMS_USERNAME = 'your_cmems_username'
CMEMS_PASSWORD = 'your_cmems_password'

# Set region of interest
lats = (51, 70)
lons = (-95, -65)

# Set years of interest
years = (1993, 2020)
years = (2014, 2020)

if __name__ == '__main__':
    
    # Point to a directory of ERA5 data stored as netcdf files--download from CDS using download_era5.py
    era5 = xr.open_mfdataset(glob.glob("/home/zgoussea/scratch/ERA5/*/*.nc"))
    era5 = era5[['t2m', 'v10', 'u10', 'sshf']]  # Only these atmospheric variables

    # Get GLORYS12 over OPENDAP
    glorys = get_glorys(CMEMS_USERNAME, CMEMS_PASSWORD)
    # glorys = glorys['siconc']  # Only SIC
    glorys = glorys[['siconc', 'sithick', 'thetao', 'vsi', 'usi']].isel(depth=0)  # Only SIC
    
    # Slice by region
    glorys = glorys.sel(latitude=slice(*lats), longitude=slice(*lons))
    era5 = era5.sel(latitude=slice(*lats[::-1]), longitude=slice(*lons))

    # Slice by time
    era5 = era5.sel(time=slice(datetime.datetime(years[0], 1, 1), datetime.datetime(years[1]+1, 1, 1)))
    glorys = glorys.sel(time=slice(datetime.datetime(years[0], 1, 1), datetime.datetime(years[1]+1, 1, 1)))

    # Resample ERA5 to daily means from the 6-hourly observations
    era5 = era5.resample(time='D').mean()

    # Interpolate the ERA5 grid to match GLORYS's grid
    era5 = era5.interp(latitude=glorys.latitude, longitude=glorys.longitude)

    # Combine the two
    ds = xr.combine_by_coords([era5, glorys], coords=['latitude', 'longitude', 'time'], join="inner", combine_attrs='drop')

    # Save each year as a separate netcdf file.
    for year in tqdm.tqdm(range(years[0], years[1]+1)):
        ds_y = ds.sel(time=slice(datetime.datetime(year, 1, 1), datetime.datetime(year, 12, 31)))
        ds_y.to_netcdf(f'/home/zgoussea/scratch/ERA5_GLORYS/ERA5_GLORYS_{year}.nc')
