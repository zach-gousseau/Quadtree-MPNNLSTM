import numpy as np
import numba
import torch
import datetime


@numba.jit
def minmax(x):
    maximum = x[0, 0]
    minimum = x[0, 0]
    for i in x[1:]:
        for j in i[1:]:
            if j > maximum:
                maximum = j
            elif j < minimum:
                minimum = j
    return (minimum, maximum)

def get_n_params(model):
    """Get number of parameters in a PyTorch model"""
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def add_positional_encoding(x):
    """(n_samples, w, h, c)"""
    assert len(x.shape) == 4, f'array should be 4-dimensional (n_samples, w, h, c); got {x.shape}'
    n_samples, w, h, c = x.shape

    image_shape = (w, h, c)

    # Position encoding
    ii = np.tile(np.array(range(image_shape[1])), (image_shape[0], 1))
    jj = np.tile(np.array(range(image_shape[0])), (image_shape[1], 1)).T

    # Normalize
    ii = ii / image_shape[1]
    jj = jj / image_shape[0]

    pos_encoding = np.moveaxis(np.array([[ii, jj]]*n_samples), 1, -1)
    if isinstance(x, torch.Tensor):
        pos_encoding = torch.Tensor(pos_encoding).type(x.dtype).to(x.device)
        x = torch.cat((x, pos_encoding), axis=-1)
    else:
        pos_encoding = pos_encoding.astype(x.dtype)
        x = np.concatenate((x, pos_encoding), axis=-1)
    return x


def gen_x_y(arr, input_ts=1, batch_size=8, num_vars=4):
    i = 0
    while i + batch_size + input_ts < arr.shape[0]:
        xs, ys = [], []
        for _ in range(batch_size):
            x = arr[i: i+input_ts].reshape(input_ts, -1, num_vars)
            y = arr[i+input_ts: i+input_ts + 1].reshape(-1, num_vars)

            xs.append(x)
            ys.append(y)

            i += 1
        xs, ys = np.array(xs), np.array(ys)
        yield xs, ys[:, :, :1]
        
def normalize(arr):
    min_ = np.min(arr, (0, 2, 3, 4))[:, None, None, None]
    max_ = np.max(arr, (0, 2, 3, 4))[:, None, None, None]
    return (arr - min_) / (max_ - min_)

def int_to_datetime(x):
    return datetime.datetime.fromtimestamp(x / 1e9)