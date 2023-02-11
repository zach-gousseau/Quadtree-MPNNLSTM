import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import datetime
import xarray as xr

from mpnnlstm import NextFramePredictor

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else 'cpu')

    # ds = xr.open_dataset('data/glorys.nc')

    ds = xr.open_zarr('data/era5_hb_daily.zarr')
    mask = np.isnan(ds.siconc.isel(time=0)).values

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # Number of frames to read as input
    input_timesteps = 3
    output_timesteps= 1

    def xarray_to_x_y(ds, start_date, end_date, input_timesteps, output_timesteps, coarsen=0):
        ds = ds.sel(time=slice(start_date, end_date))
        ds = (ds - ds.min()) / (ds.max() - ds.min())

        if coarsen != 0:
            ds = ds.coarsen(latitude=coarsen, longitude=coarsen, boundary='trim').mean()
        num_samples = ds.time.size-output_timesteps-input_timesteps
        i = 0
        x = np.ndarray((num_samples, input_timesteps, ds.latitude.size, ds.longitude.size, len(ds.data_vars)))
        y = np.ndarray((num_samples, output_timesteps, ds.latitude.size, ds.longitude.size, len(ds.data_vars)))
        while i + output_timesteps + input_timesteps < ds.time.size:
            x[i] = np.moveaxis(np.nan_to_num(ds.isel(time=slice(i, i+input_timesteps)).to_array().to_numpy()), 0, -1)
            y[i] = np.moveaxis(np.nan_to_num(ds.isel(time=slice(i+input_timesteps, i+input_timesteps+output_timesteps)).to_array().to_numpy()), 0, -1)
            i += 1

        return x, y

    def xarray_to_y(ds, start_date, end_date, input_timesteps, output_timesteps, coarsen=0):
        ds = ds.sel(time=slice(start_date, end_date))
        ds = (ds - ds.min()) / (ds.max() - ds.min())

        if coarsen != 0:
            ds = ds.coarsen(latitude=coarsen, longitude=coarsen, boundary='trim').mean()
        num_samples = ds.time.size-output_timesteps-input_timesteps
        i = 0
        y = np.ndarray((num_samples, output_timesteps, ds.latitude.size, ds.longitude.size, len(ds.data_vars)))
        while i + output_timesteps + input_timesteps < ds.time.size:
            y[i] = np.moveaxis(np.nan_to_num(ds.isel(time=slice(i+input_timesteps, i+input_timesteps+output_timesteps)).to_array().to_numpy()), 0, -1)
            i += 1

        return y

    coarsen=0
    x_vars = ['siconc', 't2m', 'v10', 'u10', 'sshf']
    # x_vars = list(ds.data_vars)

    # x, y = xarray_to_x_y(ds[x_vars], datetime.datetime(1995, 6, 1), datetime.datetime(1995, 9, 1), input_timesteps, output_timesteps, coarsen=coarsen)
    x, y = [], []
    for year in range(2005, 2006):
        x_year, y_year = xarray_to_x_y(ds[x_vars], datetime.datetime(year, 5, 1), datetime.datetime(year, 8, 1), input_timesteps, output_timesteps, coarsen=coarsen)
        x.append(x_year)
        y.append(y_year)
    x = np.concatenate(x, 0)
    y = np.concatenate(y, 0)

    x_test, y_test = xarray_to_x_y(ds[x_vars], datetime.datetime(2004, 6, 1), datetime.datetime(2004, 7, 1), input_timesteps, output_timesteps, coarsen=coarsen)
    x_val, y_val = xarray_to_x_y(ds[x_vars], datetime.datetime(2005, 6, 1), datetime.datetime(2005, 7, 1), input_timesteps, output_timesteps, coarsen=coarsen)
    y_viz = xarray_to_y(ds[x_vars], datetime.datetime(2005, 6, 1), datetime.datetime(2005, 9, 15), input_timesteps, output_timesteps, coarsen=coarsen)

    print('Created datasets')

    model_kwargs = dict(
        hidden_size=64,
        dropout=0.1,
        multi_step_loss=output_timesteps,
        input_timesteps=input_timesteps
        )

    model = NextFramePredictor(experiment_name='test', decompose=True, input_features=len(x_vars), integrated_space_time=False, **model_kwargs)
    print('Num params:', model.get_n_params())

    # model.set_thresh(-np.inf)  # Set the threshold based on the plots above
    model.set_thresh(0.15)

    lr = 0.1

    print('Beginning training')

    # n_train_processes = 8

    # model.model.share_memory()

    # processes = []
    # for rank in range(n_train_processes + 1):  # + 1 for test process
    #     p = mp.Process(target=model.train, args=(x[:365], y[:365], x_test, y_test, 15, lr, 0.95))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()


    # model.model = torch.nn.DataParallel(model.model)
    model.train(x, y, x_test, y_test, lr=lr, n_epochs=1)  # Train for 20 epochs


    print('Getting MSE on val. set')
    model.model.eval()
    print(model.score(x_val, y_val[:, :1]))  # Check the MSE on the validation set

    print('Plotting loss')
    model.loss.iloc[1:].plot()
    model.loss.to_csv('ice_results/loss.csv')

    ar_steps = 90

    launch_step =0#w 90 + 40

    print('Getting predictions on val. set')
    # Get predictions on the validation set
    model.model.eval()
    y_hat = model.predict(x_val[[launch_step]], ar_steps)

    print('Plotting validation set predictions')
    for i in range(ar_steps):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        y_val_step = y_viz[launch_step + i].squeeze(0)[..., 0]
        y_hat_step = y_hat.squeeze(0)[i][..., 0]
        axs[0].imshow(np.where(~mask, y_val_step, np.nan), vmin=0, vmax=1)
        axs[1].imshow(np.where(~mask, y_hat_step, np.nan), vmin=0, vmax=1)
        axs[0].set_title('True')
        axs[1].set_title('Pred')
        plt.savefig(f'ice_results/example_{i}.png')
        plt.close()
    print('Finito.')