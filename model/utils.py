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

def round_to_day(dt):
    return datetime.datetime(*dt.timetuple()[:3])

class GraphSamplerData:
    def __init__(self, edge_index_samples, n_ids, e_ids):
        self.edge_index_samples = edge_index_samples
        self.n_ids = n_ids
        self.e_ids = e_ids
        

class GraphSampler:
    def __init__(self, nodes, edge_index, n_samples=100, n_hops=3):
        self.n_samples = n_samples
        self.edge_index = edge_index
        self.nodes = nodes
        self.n_hops = n_hops
        
    def get_random_target_nodes(self):
        originating_nodes = self.edge_index[0] 
        terminating_nodes = self.edge_index[1] 

        # Initial node
        parent_nodes = torch.Tensor([np.random.choice(self.nodes)])

        nodes_sample = parent_nodes
        while len(nodes_sample) < self.n_samples:
            children_idx = torch.where(torch.isin(originating_nodes, parent_nodes))[0]
            parent_nodes = terminating_nodes[children_idx]
            nodes_sample = torch.unique(torch.cat((nodes_sample, parent_nodes), 0))
        return nodes_sample[:self.n_samples]
        
    
    def __iter__(self):
        while True:
            nodes_sample = self.get_random_target_nodes()
            n_ids = torch.where(torch.isin(self.nodes, nodes_sample))[0]

            terminating_nodes = self.edge_index[1] 

            edge_index_samples = []
            e_ids = []
            relevant_nodes = nodes_sample
            for _ in range(self.n_hops):
                e_id = torch.where(torch.isin(terminating_nodes, relevant_nodes))[0]  # Get indices of relevant edges

                edge_index_sample = self.edge_index[:, e_id]
                edge_index_samples.append(edge_index_sample)
                e_ids.append(e_id)

                relevant_nodes = edge_index_sample[0]  # For next iteration

            yield GraphSamplerData(edge_index_samples, n_ids, e_ids)