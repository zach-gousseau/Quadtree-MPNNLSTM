import numpy as np

def get_n_params(model):
    """Get number of parameters in a PyTorch model"""
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp



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