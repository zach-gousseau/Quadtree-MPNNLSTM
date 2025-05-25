#import xarray as xr
#import rioxarray
import numpy as np


CMEMS_USERNAME = 'zgousseau'#'your_cmems_username'
CMEMS_PASSWORD = 'Lopolmuk8!'
DATASET_ID = 'cmems_mod_glo_phy_my_0.083_P1D-m'

import copernicusmarine

def get_glorys(username, password):
    copernicusmarine.login(username=username, password=password, force_overwrite=True)
    ds = copernicusmarine.open_dataset(dataset_id='cmems_mod_glo_phy_my_0.083deg_P1D-m')
    ds = ds.rio.write_crs(4326)
    return ds

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--year', type=int, default=2024)
    parser.add_argument('-o', '--output_dir', type=str, default='.')
    args = parser.parse_args()
    year = args.year
    output_dir = args.output_dir
    ds = get_glorys(CMEMS_USERNAME, CMEMS_PASSWORD)
    ds.to_netcdf(f'{output_dir}/GLORYS_{year}.nc')
