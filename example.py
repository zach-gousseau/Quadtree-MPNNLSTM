import numpy as np
import torch
import random
import os

from mod_moving_mnist import ModMovingMNIST
from mpnnlstm import NextFramePredictor

if __name__ == '__main__':

    # e.g. testing the difference between the MPNNLSTM with quadtree decomposition and without.

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # Number of frames to read as input
    input_timesteps = 4

    # Directory in which to save results
    out_directory = 'results'

    # Create a dataset using the modified MovingMNIST
    MMMNIST = ModMovingMNIST(
        canvas_size=(32, 32),
        digit_size=(12, 12),
        pixel_noise=0.05,
        velocity_noise=0.25
    )

    # Create a train set (500), test set (50), and validation set (50)
    x, y = MMMNIST.create_dataset(500, input_timesteps, n_digits=1)
    x_test, y_test = MMMNIST.create_dataset(50, input_timesteps, n_digits=1)
    x_val, y_val = MMMNIST.create_dataset(50, input_timesteps, n_digits=1)

    model_kwargs = dict(
        hidden_size=64,
        dropout=0.1,
        input_timesteps=input_timesteps
        )

    # Train with decomposition
    model = NextFramePredictor(experiment_name='reduced', decompose=True, input_features=1, **model_kwargs)
    model.set_thresh(0.01)  # Set the threshold -- make sure to check that this value is appropriate
    model.train(x, y, x_test, y_test, n_epochs=10)
    print('MSE:', model.score(x_val, y_val))
    model.save(out_directory)
    model.losses.to_csv(os.path.join(out_directory, f'losses_reduced.csv'))

    # Train without decomposition
    model = NextFramePredictor(experiment_name='full', decompose=False, input_features=1, **model_kwargs)
    model.train(x, y, x_test, y_test, n_epochs=10)
    print('MSE:', model.score(x_val, y_val))
    model.save(out_directory)
    model.losses.to_csv(os.path.join(out_directory, f'losses_full.csv'))
