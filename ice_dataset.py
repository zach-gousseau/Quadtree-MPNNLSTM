import numpy as np
import os
import datetime
from dateutil.relativedelta import relativedelta

from torch.utils.data import Dataset

class IceDataset(Dataset):
    def __init__(self, ds, years, month, input_timesteps, output_timesteps, x_vars=None, y_vars=None, train=False, y_binary_thresh=None, cache_dir=None):
        self.train = train
        
        self.x, self.y, self.launch_dates = self.get_xy(ds, years, month, input_timesteps, output_timesteps, x_vars=x_vars, y_vars=y_vars, y_binary_thresh=y_binary_thresh, cache_dir=cache_dir)
        self.image_shape = self.x[0].shape[1:-1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.launch_dates[idx]

    def get_xy(self, ds, years, month, input_timesteps, output_timesteps, x_vars=None, y_vars=None, y_binary_thresh=None, cache_dir=None):
        
        if cache_dir is not None:
            dataset_id = f'LA{ds.latitude.min().values.item()}_{ds.latitude.max().values.item()}_LO{ds.longitude.min().values.item()}_{ds.longitude.max().values.item()}' + \
                         f'_RES{(ds.latitude[1] - ds.latitude[0]).values.item().__round__(4)}' + \
                         f'_Y{years[0]}_Y{years[-1]}' + \
                         f'_M{month}' + \
                         f'_I' + '_'.join(x_vars) + '_O' + '_'.join(y_vars) + f'_T{self.train}' + f'_BIN{y_binary_thresh}'
                        
            cache_dir = os.path.join(cache_dir, dataset_id)
            
            try:
                x = np.load(os.path.join(cache_dir, 'x.npy'))
                y = np.load(os.path.join(cache_dir, 'y.npy'))
                launch_dates = np.load(os.path.join(cache_dir, 'launch_dates.npy'))
                x, y = x.astype('float16'), y.astype('float16')
                return x, y, launch_dates
            except FileNotFoundError:
                pass

        x, y = [], []
        launch_dates = []
        for year in years:
            
            x_vars = list(ds.data_vars) if x_vars is None else x_vars
            y_vars = list(ds.data_vars) if y_vars is None else y_vars
            
            if self.train:
                # 3 months around the month of interest
                start_date = datetime.datetime(year, month, 1) - relativedelta(months=1)
                end_date = datetime.datetime(year, month, 1) + relativedelta(months=2)
            else:
                start_date = datetime.datetime(year, month, 1)
                end_date = datetime.datetime(year, month, 1) + relativedelta(months=1)
                

            # Add buffer for input timesteps and output timesteps 
            start_date -= relativedelta(days=input_timesteps)
            end_date += relativedelta(days=output_timesteps-1)

            # Slice dataset & normalize
            ds_year = ds.sel(time=slice(start_date, end_date))
            
            # Add DOY
            ds_year['doy'] = (('time', 'latitude', 'longitude'), ds_year.time.dt.dayofyear.values.reshape(-1, 1, 1) * np.ones(shape=(ds_year[x_vars[0]].shape)))
            
            ds_year = (ds_year - ds_year.min()) / (ds_year.max() - ds_year.min())

            num_samples = ds_year.time.size - output_timesteps - input_timesteps

            i = 0
            x_year = np.ndarray((num_samples, input_timesteps, ds.latitude.size, ds.longitude.size, len(x_vars)), dtype='float32')
            y_year = np.ndarray((num_samples, output_timesteps, ds.latitude.size, ds.longitude.size, len(y_vars)), dtype='float32')
            while i + output_timesteps + input_timesteps < ds_year.time.size:
                x_year[i] = np.moveaxis(np.nan_to_num(ds_year[x_vars].isel(time=slice(i, i+input_timesteps)).to_array().to_numpy()), 0, -1).astype('float32')
                y_year[i] = np.moveaxis(np.nan_to_num(ds_year[y_vars].isel(time=slice(i+input_timesteps, i+input_timesteps+output_timesteps)).to_array().to_numpy()), 0, -1).astype('float32')
                i += 1

            x.append(x_year)
            y.append(y_year)
            launch_dates.append(ds_year.time[input_timesteps:-output_timesteps].values)

        x, y, launch_dates = np.concatenate(x, 0), np.concatenate(y, 0), np.concatenate(launch_dates, 0)

        if y_binary_thresh is not None:
            y = y > y_binary_thresh

        launch_dates = launch_dates.astype(int)
        
        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            np.save(os.path.join(cache_dir, 'x.npy'), x)
            np.save(os.path.join(cache_dir, 'y.npy'), y)
            np.save(os.path.join(cache_dir, 'launch_dates.npy'), launch_dates)
        
        return x, y, launch_dates