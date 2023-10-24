import numpy as np
import warnings
import numpy.ma as ma
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Any
import matplotlib.pyplot as plt
from collections import defaultdict
import numba as nb
from functools import lru_cache

from model.utils import minmax, add_positional_encoding

CONDITIONS = [
    'max_larger_than',
    'max_smaller_than',
    'min_larger_than',
    'min_smaller_than',
]


class Graph:
    """
    Object to store the mesh state, including the graph (pyg), pixel mapping, hidden and cell states of the LSTM.
    """
    def __init__(self, edge_index, edge_attr, **kwargs):
        self.pyg = Data(edge_index=edge_index, edge_attr=edge_attr, **kwargs)  # pytorch-geometric graph object
        self.mapping = None
        self.n_pixels_per_node = None

        self.hidden = None
        self.cell = None

def quadtree_decompose_cuda(img, padding=0, thresh=0.05, max_size=8, mask=None, transform_func=None, condition='max_larger_than'):
    """CUDA version of quadtree_decompose()"""
    assert max_size & (max_size - 1) == 0
    assert condition in CONDITIONS
    n, m = img.shape
    labels = torch.full((-(n // -max_size) * max_size, -(m // -max_size) * max_size), -1, dtype=int, device='cuda')
    shape = n_padded, m_padded = labels.shape

    img = torch.nn.functional.pad(img, (0, m_padded-m, 0, n_padded-n))
    img_for_criteria = transform_func(img) if transform_func else img

    cur_label = torch.tensor(0, dtype=torch.int32, device='cuda')

    stack = []
    for i in range(n_padded // max_size):
        for j in range(m_padded // max_size):
            stack.append((i*max_size, j*max_size, max_size))

    while stack:
        x, y, size = stack.pop()

        if x >= n or y >= m:
            continue

        l, r, t, b = x, x + size + 1, y, y + size + 1

        if size == 1:
            if mask is not None and mask[x, y]:
                continue

            labels[x, y] = cur_label
            cur_label += 1
            continue

        cell = img_for_criteria[
            max(0, l-padding): min(r+padding, shape[1]),
            max(0, t-padding): min(b+padding, shape[1])
        ]

        if condition == 'max_larger_than':
            split_cell = cell.max() > thresh
        elif condition == 'max_smaller_than':
            split_cell = cell.max() < thresh
        elif condition == 'min_larger_than':
            split_cell = cell.min() > thresh
        elif condition == 'min_smaller_than':
            split_cell = cell.min() < thresh

        overlaps_mask = mask is not None and mask[max(0, l-padding): min(r+padding, shape[1]), max(0, t-padding): min(b+padding, shape[1])].any()
        split_cell = split_cell or overlaps_mask

        if split_cell:
            new_size = size // 2
            stack.append((x, y, new_size))
            stack.append((x + new_size, y, new_size))
            stack.append((x, y + new_size, new_size))
            stack.append((x + new_size, y + new_size, new_size))
        else:
            labels[x:x+size, y:y+size] = cur_label
            cur_label += 1

    return labels[:n, :m]


def plot_contours(ax, labels):
    """Plot cell contours for a given set of pixel labels shape=(w, h)"""
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            try:
                if labels[i][j] != labels[i][j+1]:
                    ax.plot([j+0.5, j+0.5], [i-0.5, i+0.5], c='k', lw=0.5)
            except IndexError:
                pass

            try:
                if labels[i][j] != labels[i+1][j]:
                    ax.plot([j-0.5, j+0.5], [i+0.5, i+0.5], c='k', lw=0.5)
            except IndexError:
                pass

def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)

# These are just faster versions of any(), max() and min() for 2D arrays
@nb.jit(nopython=True)
def any_2d(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j]:
                return True
    return False

@nb.jit(nopython=True)
def max_2d(arr):
    max_val = arr[0, 0]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] > max_val:
                max_val = arr[i, j]
    return max_val

@nb.jit(nopython=True)
def min_2d(arr):
    min_val = arr[0, 0]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] < min_val:
                min_val = arr[i, j]
    return min_val

def quadtree_decompose(img, padding=1, thresh=0.05, max_size=8, mask=None, high_interest_region=None, transform_func=None, condition='max_larger_than'):
    """
    Perform quadtree decomposition on an image.

    This function decomposes the input image into a quadtree by dividing the image
    into four quadrants of equal size and repeating the process recursively until
    the maximum value in each quadrant is either below some threshold or contains a 
    single pixel. 
    
    Optionally, provide a mask which will ensure not cell contains any value which 
    falls within the mask. Masked pixels are assigned a label of -1.
    
    Optionally, provide a high interest region mask which will continue splitting the
    image where it overlaps the region.
    
 
    Parameters:
    img (np.ndarray): Input image with shape (height, width).
    padding (int, optional): Padding to add around each cell when checking the splitting criteria. Default is 0.
    thresh (float, optional): Threshold to use as splitting criteria. Default is 0.05.
    max_size (int, optional): Maximum grid cell size. Default is 8.
    mask (np.ndarray, optional): Boolean mask. Masked out pixels will be set as invalid
    high_interest_region (np.ndarray, optional): Boolean mask. Masked out pixels will be kept as valid.
    transform_func (optional): Function to apply to the input image used for criteria evaluation
    condition (str, optional): Condition to use along with the threshold for splitting cells

    Returns:
    np.ndarray: Array of shape (height, width) containing the labels for each pixel in the image.
        Note: A '-1' label means that the pixel is invalid (according to the provided mask)
    """
    
    assert max_size & (max_size - 1) == 0
    
    assert condition in CONDITIONS

    n, m = img.shape
    
    # Initialize label array with the base grid (maximum grid cell size) while 
    # Note that the initial label array may be larger than the original image since
    # we do not want to cut off any base grid cells
    # labels = torch.full((-(n // -max_size) * max_size, -(m // -max_size) * max_size), -1, dtype=int)
    labels = np.full((-(n // -max_size) * max_size, -(m // -max_size) * max_size), -1, dtype=int)
    shape = n_padded, m_padded = labels.shape
    
    # Pad the image to match the labels array
    img = np.pad(img, ((0, n_padded-n), (0, m_padded-m)), mode='edge')
    # img = F.pad(img.unsqueeze(0), (0, m_padded-m, 0, n_padded-n), mode='replicate').squeeze(0)
    
    # Apply transformation if desired
    img_for_criteria = transform_func(img) if transform_func else img

    cur_label = 0
    
    # Build initial stack using each of the cells in the base grid
    stack = []
    for i in range(n_padded // max_size): 
        for j in range(m_padded // max_size):
            stack.append((i*max_size, j*max_size, max_size))

    while stack:
        x, y, size = stack.pop()
        
        # Skip if within the padded zone (which we ignore)
        if x >= n or y >= m:
            continue

        l, r, t, b = x, x + size + 1, y, y + size + 1
        
        # Stop if cell is singular
        if size == 1:
            if mask is not None and mask[x, y]:
                continue

            labels[x, y] = cur_label
            cur_label += 1
            continue
        
        cell = img_for_criteria[
            max(0, l-padding): min(r+padding, shape[1]),
            max(0, t-padding): min(b+padding, shape[1])
        ]
        
        # Split if the cell meets the specified criteria
        if condition == 'max_larger_than':
            split_cell = max_2d(cell) > thresh 
        elif condition == 'max_smaller_than':
            split_cell = max_2d(cell) < thresh 
        elif condition == 'min_larger_than':
            split_cell = min_2d(cell) > thresh 
        elif condition == 'min_smaller_than':
            split_cell = min_2d(cell) < thresh

        
        # Even if it doesn't meet the criteria, split if the cell overlaps a masked area
        overlaps_mask = mask is not None and any_2d(mask[max(0, l-padding): min(r+padding, shape[1]), max(0, t-padding): min(b+padding, shape[1])])
        
        if high_interest_region is not None:
            overlaps_hir = high_interest_region is not None and any_2d(high_interest_region[max(0, l-padding): min(r+padding, shape[1]), max(0, t-padding): min(b+padding, shape[1])])
        else:
            overlaps_hir = False
        
        split_cell = split_cell or overlaps_mask or overlaps_hir
        
        # Perform splitting if criteria is met, otherwise set all pixels to the current label
        if split_cell:
            new_size = size // 2
            stack.append((x, y, new_size))
            stack.append((x + new_size, y, new_size))
            stack.append((x, y + new_size, new_size))
            stack.append((x + new_size, y + new_size, new_size))
        else:
            labels[x:x+size, y:y+size] = cur_label
            cur_label += 1
    
    return labels[:n, :m]

def get_adj(labels, xx, yy, edges_at_corners=True, use_edge_attrs=True):
    """
    Get the adjacency matrix for a given label matrix by connecting all adjacent cells (this could be more efficient).
    Basically for a label matrix that looks (e.g.) like this: 

    0, 0, 1
    2, 3, 3
    2, 3, 3

    it would set edges between (0, 1); (0, 2); (0, 3); (1, 3); (2, 3) and create the corresponding adjacency matrix:

        0  1  2  3  
        ----------
    0 | 0, 1, 1, 1
    1 | 1, 0, 0, 1
    2 | 1, 0, 0, 1
    3 | 1, 1, 1, 0

    The result is stored as a set of indices rather than an actual adjacency matrix, i.e.:
    
    0, 0, 0, 1, 2 
    1, 2, 3, 3, 3

    Parameters:
    labels (np.ndarray): Labels array of shape (w, h). The labels must contain labels 0 through N for N cells.
    xx (np.ndarray, optional): The x positions for each node of shape (1, N)
    yy (np.ndarray, optional): The y positions for each node of shape (1, N)
    edges_at_corners (bool, optional): Whether to add edges between any two cells which touch at the corners
    use_edge_attrs (bool, optional): Whether to use edge attributes of shape (c, N) for c features, or only edge weights (1, N)
    """
    w, h = labels.shape
    adj_dict = {}

    edge_sources, edge_targets, edge_attrs = [], [], []

    for i in range(w):
        for j in range(h):

            node = labels[i][j]
            
            # Skip if the current label is invalid (-1)
            if node == -1:
                continue

            if node not in adj_dict:
                adj_dict[node] = []

            neighbors = set()

            if i != 0:
                neighbors.add(labels[i-1][j])
            if i != w-1:
                neighbors.add(labels[i+1][j])
            if j != 0:
                neighbors.add(labels[i][j-1])
            if j != h-1:
                neighbors.add(labels[i][j+1])
            
            if edges_at_corners:
                if (i != 0) and (j != 0):
                    neighbors.add(labels[i-1][j-1])
                if (i != w-1) and (j != 0):
                    neighbors.add(labels[i+1][j-1])
                if (i != 0) and (j != h-1):
                    neighbors.add(labels[i-1][j+1])
                if (i != w-1) and (j != h-1):
                    neighbors.add(labels[i+1][j+1])

            # Remove self-loop if it exists
            # try:
            #     neighbors.remove(node)
            # except KeyError:
            #     pass

            # Remove links to invalid nodes (-1) if it exists
            try:
                neighbors.remove(-1)
            except KeyError:
                pass
    
            for neighbor in neighbors:
                if neighbor not in adj_dict[node]:
                    adj_dict[node].append(neighbor)
                    edge_sources.append(node)
                    edge_targets.append(neighbor)

    if use_edge_attrs:
        edge_attrs = torch.stack((
            dist_angle(edge_sources, edge_targets, xx, yy),
            dist(edge_sources, edge_targets, xx, yy)
        )).T
    else:
        edge_attrs = dist(edge_sources, edge_targets, xx, yy)
        edge_attrs = 1 - (edge_attrs / edge_attrs.max())

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    return edge_index, edge_attrs

def dist(node0, node1, xx, yy):
    """
    Distance between two nodes. xx/yy are arrays of x/y positions for each node,
    and node0, node1 are the index of the nodes of interest
    """
    return torch.sqrt((yy[node0] - yy[node1])**2 + (xx[node0] - xx[node1])**2)

def dist_angle(node0, node1, xx, yy):
    """
    Angle (bearing) from one node to another. xx/yy are arrays of x/y positions for each node,
    and node0, node1 are the index of the nodes of interest
    """
    return torch.atan2(xx[node0] - xx[node1], yy[node0] - yy[node1]) % (2*np.pi) / (2*np.pi)

def dist_xy(node0, node1, xx, yy):
    """Categorical <<distance>> which gives (0, 0.25, 0.5, 0.75, 1) if node1 is to (above, right, left, below) node0"""
    x = xx[node0] < xx[node1]
    y = yy[node0] < yy[node1]
    return (x*2 + y) / 3
    # return np.array((xx[node0] - xx[node1], yy[node0] - yy[node1]))  # Or this just gives the x and y distances as a tuplre

def get_graph_nodes(labels):
    """Nodes are just 0 to N where N is the number of nodes"""
    graph_nodes = np.arange(torch.max(labels)+1)
    
def flatten_pixelwise(img, mask):
    """Shortcut flatten() if each pixel is treated as a node"""
    if mask is not None:
        data = img[:, ~mask, :]
    else:
        data = img.reshape(img.shape[0], -1, img.shape[-1])
    return data

def flatten(img, mapping, n_pixels_per_node, mask=None):
    """
    Given an input image of dimension (n_samples, w, h, channels) and a labels array of dimension (w, h)
    which correspond to the mesh node to which each pixel in the original image belong, convert the image to 
    its mesh representation.
    img: (n_samples, w, h, c)"""
    assert len(img.shape) == 4, f'array should be 4-dimensional (n_samples, w, h, c); got {img.shape}'
    n_samples, w, h, c = img.shape

    if mapping is None:
        return flatten_pixelwise(img, mask)
    
    # (n_samples, w, h, c) -> (c, n_samples, w*h)
    img_flattened = torch.moveaxis(img, -1, 0).reshape(c, n_samples, w*h)
    
    # Compute mean values for each graph node
    while True:
        data = img_flattened @ mapping.T / n_pixels_per_node

        if data.isnan().any():
            warnings.warn('Matrix multiplication in flatten() failed, trying again.')
        else:
            break

    # (c, n_samples, w*h) -> (n_samples, w*h, c)
    data = torch.moveaxis(data, 0, -1)

    return data

def grouped_mean(arr, labels):
    """
    Given an 1-dimensional array of length N containing data and a non-negative label array of the same size,
    for each unique label in the labels array, compute the mean value of the corresponding entries
    in the data array. Invalid entries should be labelled -1, and will be excluded from the mean.
    
    e.g.
    arr = [1, 2, 3, 4, 5]
    labels = [0, 1, 1, 2, 2]
    
    should return: [1, 2.5, 4.5] for labels [0, 1, 2]
    """
    if -1 in labels:
        labels = labels + 1
        binned_data = np.bincount(labels, arr)
        bin_counts = np.bincount(labels)
        return (binned_data / bin_counts)[1:]
    else:
        binned_data = np.bincount(labels, arr)
        bin_counts = np.bincount(labels)
        return binned_data / bin_counts


def grouped_mean_along_axis_2d(arr, labels, axes):
    """Apply grouped_mean() along two axes"""
    def grouped_mean_along_axis(arr, labels):
        return np.apply_along_axis(grouped_mean, axis=axes[0], arr=arr, labels=labels)
    return np.apply_along_axis(grouped_mean_along_axis, axis=axes[1], arr=arr, labels=labels) 


def unflatten(data, mapping, image_shape, mask=None):
    """Create an image of shape (n, w, h, c) for n samples of dimensions w, h and c channels"""
    if mapping is None:
        return unflatten_pixelwise(data, mask, image_shape)
    data = torch.moveaxis(data, -1, 0)
    # img = (data @ mapping.to_dense()).reshape(*data.shape[:-1], *image_shape)
    img = (data @ mapping).reshape(*data.shape[:-1], *image_shape)
    while img.isnan().any():
        img = (data @ mapping).reshape(*data.shape[:-1], *image_shape)
        warnings.warn('Matrix multiplication in unflatten() failed, trying again.')
    return torch.moveaxis(img, 0, -1)

def unflatten_pixelwise(data, mask, image_shape):
    """unflatten() if each pixel is treated as a node"""
    _, c = data.shape
    if mask is not None:
        img = torch.full((*image_shape, c), np.nan).to(data.device)
        img[~mask, :] = data
    else:
        img = data.reshape(*image_shape, c)
    return img


def get_adj_pixelwise(labels, xx=None, yy=None, use_edge_attrs=True, resolution=0.25):
    """get_adj() if each pixel is treated as a node"""
    rows, cols = labels.shape

    # Shift the array by one row and one column in each direction
    north = np.roll(labels, shift=-1, axis=0)
    south = np.roll(labels, shift=1, axis=0)
    west = np.roll(labels, shift=-1, axis=1)
    east = np.roll(labels, shift=1, axis=1)

    # Remove the indices which have rolled around 
    north[-1] = -1
    south[0] = -1
    west[:, -1] = -1
    east[:, 0] = -1

    neighbors = torch.Tensor(np.array([
        labels.flatten().repeat(4),
        np.vstack((north, south, west, east)).reshape(4, rows*cols).T.flatten()
    ])).type(torch.int64)

    # Mask out invalid nodes
    edge_index = neighbors[:, ~torch.any(neighbors == -1, 0)]

    if use_edge_attrs:
        edge_attrs = torch.stack((
            dist_angle(edge_index[0], edge_index[1], xx, yy),
            dist(edge_index[0], edge_index[1], xx, yy)
        )).T
    else:
        edge_attrs = None#torch.ones(neighbors.shape[1])

    return edge_index, edge_attrs
    

def image_to_graph_pixelwise(img, mask=None, use_edge_attrs=True, resolution=0.25):
    """image_to_graph() if each pixel is treated as a node"""
    img0, _ = torch.max(img[..., 0], 0)  # For multi-step inputs
    image_shape = img.shape[1:3]

    if mask is not None:
        labels = ma.masked_array((~mask).flatten().cumsum() - 1, mask=mask.astype(bool)).filled(-1).reshape(img0.shape)
        graph_nodes = torch.arange(np.sum(~mask))
    else:
        labels = np.arange(np.prod(img.shape[1:3])).reshape(img.shape[1:3])
        graph_nodes = torch.arange(np.prod(img0.shape))

    data = flatten_pixelwise(img, mask)
    xx, yy = data[0, ..., -2]*image_shape[1]*resolution, data[0, ..., -1]*image_shape[0]*resolution

    cell_sizes = torch.ones((data.shape[0], len(graph_nodes))).to(data.device) * (resolution ** 2)
    data = torch.cat([data, cell_sizes.unsqueeze(-1)], -1)

    n_pixels_per_node = torch.ones((len(graph_nodes))).to(img.device)
    mapping = None

    # Distances are all the same so don't bother calculating them. Uses '1' as the distance for each edge.
    edge_index, edge_attrs = get_adj_pixelwise(labels, xx=xx, yy=yy, use_edge_attrs=use_edge_attrs)

    out = dict(
        edge_index=edge_index,
        edge_attrs=edge_attrs,
        data=data,
        graph_nodes=graph_nodes,
        mapping=mapping,
        n_pixels_per_node=n_pixels_per_node,
    )

    return out

def get_mapping_(labels):
    """Slow version of get_mapping(), not used"""
    graph_nodes = get_graph_nodes(labels)
    labels_flat = labels.flatten()
    mapping = torch.zeros((graph_nodes[-1]+1, len(labels_flat)))

    for i, n in enumerate(labels_flat):
        if n != -1:
            mapping[n][i] = 1
    
    n_pixels_per_node = torch.sum(mapping, 1)
    mapping = mapping
    return mapping, graph_nodes, n_pixels_per_node

def get_mapping(labels):
    """
    Create a NxP binary sparse matrix for N pixels and P graph nodes, where entry (n, p) 
    is assigned 1 if the pth pixel (of the flattened grid) belongs to node n.

    Basically for a label matrix that looks (e.g.) like this: 

    0, 0, 1
    2, 3, 3 -> flatten -> 0, 0, 1, 2, 3, 3, 2, 3, 3
    2, 3, 3

    the corresponding mapping would be:

    0 |   1, 1, 0, 0, 0, 0, 0, 0, 0
    1 |   0, 0, 1, 0, 0, 0, 0, 0, 0
    2 |   0, 0, 0, 1, 0, 0, 1, 0, 0
    3 |   0, 0, 0, 0, 1, 1, 0, 1, 1

    This function also returns an array of nodes (e.g. [0, 1, 2, 3] in this case) and the number of pixels per node
    (e.g. [2, 1, 2, 4] in this case).

    """
    labels_flat = labels.flatten()
    mask = (labels_flat != -1)
    row = labels_flat[mask].tolist()
    col = torch.arange(len(labels_flat))[mask]
    data = torch.ones(len(row), dtype=torch.float32)
    
    graph_nodes, n_pixels_per_node = np.unique(row, return_counts=True)
    n_pixels_per_node = torch.Tensor(n_pixels_per_node)
    
    mapping = torch.sparse_coo_tensor((row, col), data, size=(graph_nodes[-1]+1, len(labels_flat)))#.to_dense()
    return mapping, graph_nodes, n_pixels_per_node


def image_to_graph(img, thresh=0.05, max_grid_size=64, mask=None, high_interest_region=None, transform_func=None, condition='max_larger_than', use_edge_attrs=True, resolution=0.25):
    """
    Convert an image to its graph representation using quadtree decomposition.

    The steps are:
    1. Use quadtree_decompose() to decompose and assign cell labels to each pixel in the image
    2. Use get_mapping() to get a transformation matrix which maps from pixel-space to cell-space and apply it to the image (and add cell sizes)
    3. Use get_adj() to get the adjacecy matrix and edge attributes (as a set of edge indices and attributes rather than an actual adj. matrix)

    Parameters:
    img (np.ndarray): Input image with shape (batch_size, height, width, channels).
    thresh (float, optional): Threshold for quadtree decomposition. Default is 0.05.
    max_grid_size (int, optional): Cell size of the base grid which will be further divided using quadtree decomposition (acts as the maximum
        cell size in the mesh)
    mask (np.ndarray, optional): Pixels in img which should be ignored. Must be same shape as img
    high_interest_region (np.ndarray, optional): Pixels in img which are of high interest and will be split to its highest resolution. Must be same shape as img
    transform_func (optional): Function to apply to the input image used for criteria evaluation
    condition (str, optional): Condition to use along with the threshold for splitting cells
    use_edge_attrs (bool, optional): Whether to use edge attributes of shape (c, N) for c features, or only edge weights (1, N)
    resolution (float, optional): The resolution of the image (i.e. the size of a single pixel)
    ...

    Returns:
    dict: Dictionary containing the following information:
        edge_index (torch.Tensor): Edge connection tensor of shape (2, num_nodes)
        edge_attrs (torch.Tensor): Edge attributes tensor of shape (edge_dim, num_nodes)
        n_pixels_per_node (torch.Tensor):
        data (np.ndarray): Array of shape (batch_size, num_nodes, num_features+1) containing the data for each graph node. The last column is the size of the graph node.
        graph_nodes (list): List of graph nodes.
        mapping (dict): Dictionary mapping graph node labels to indices.

    TODO: Add ability to choose which channel to use in the decomposition.
    """

    assert len(img.shape) == 4, f'array should be 4-dimensional (n_samples, w, h, c); got {img.shape}'

    if torch.any(torch.isnan(img)):
        raise ValueError(f'Found NaNs in image data {torch.sum(torch.isnan(img))} / {np.prod(img.shape)}')

    if thresh == -np.inf:
        return image_to_graph_pixelwise(img, mask, use_edge_attrs=use_edge_attrs, resolution=resolution)

    img_for_decompose, _ = torch.max(img[..., 0], 0)  # For multi-step inputs
    n_samples, h, w, c = img.shape

    image_shape = img_for_decompose.shape
    img_for_decompose = img_for_decompose.cpu().detach().numpy()
    
    labels = quadtree_decompose(
        img_for_decompose,
        thresh=thresh, 
        max_size=max_grid_size,
        mask=mask,
        high_interest_region=high_interest_region,
        transform_func=transform_func,
        condition=condition
        )

    mapping, graph_nodes, n_pixels_per_node = get_mapping(labels)
    mapping = mapping.to_dense()
    # mapping, n_pixels_per_node = mapping.to(img.device).type(img.dtype), n_pixels_per_node.to(img.device).type(img.dtype)
    mapping, n_pixels_per_node = mapping.to(img.device), n_pixels_per_node.to(img.device)
    data = flatten(img, mapping, n_pixels_per_node)

    if torch.any(torch.isnan(data)):
        raise ValueError(f'Found NaNs in graph data {torch.sum(torch.isnan(data))} / {np.prod(data.shape)}')
    
    xx, yy = data[0, ..., -2]*image_shape[1]*resolution, data[0, ..., -1]*image_shape[0]*resolution
    # xx, yy = xx.detach().cpu(), yy.detach().cpu()
    
    # Get sizes for each graph node 
    cell_sizes = n_pixels_per_node  # TODO: scale by latitude

    # Pseudo-normalize and add node sizes as feature 
    cell_sizes = torch.Tensor(cell_sizes) / ((max_grid_size/2)**2)
    cell_sizes = cell_sizes.repeat((n_samples, *[1]*len(cell_sizes.shape)))

    data = torch.cat([data, cell_sizes.unsqueeze(-1)], -1)

    edge_index, edge_attrs = get_adj(labels, xx, yy, use_edge_attrs=use_edge_attrs)

    out = dict(
        edge_index=edge_index,
        edge_attrs=edge_attrs,
        data=data,
        graph_nodes=graph_nodes,
        mapping=mapping,
        n_pixels_per_node=n_pixels_per_node,
        max_cell_size=max_grid_size,
    )

    return out

def create_static_heterogeneous_graph(image_shape, max_grid_size, mask, high_interest_region=None, use_edge_attrs=True, resolution=0.25, device=None):
    """
    Create a static heterogeneous graph which uses a higher density of points near edges (of a given mask). This is done
    by using quadtree decomposition to recursively split a base grid (with cells of size max_grid_size) if they overlap the mask. 
    """
    arr = torch.zeros(size=(1, *image_shape, 1)).to(device)
    arr = add_positional_encoding(arr)
    graph_structure = image_to_graph(
        arr, thresh=np.inf, 
        max_grid_size=max_grid_size, 
        mask=mask, 
        high_interest_region=high_interest_region, 
        use_edge_attrs=use_edge_attrs, 
        resolution=resolution
        )
    del graph_structure['data']
    return graph_structure

def get_nan_nodes(mask, graph_structure):
    return np.where(~mask.flatten() @ graph_structure['mapping'].to_dense().cpu().numpy().T == 0)[0]

def replace_with_map(arr, map_):
    return np.vectorize(map_.get)(arr)

def create_static_homogeneous_graph(image_shape, max_grid_size, mask, high_interest_region=None, use_edge_attrs=True, resolution=0.25, device=None):
    """Create a static homogeneous graph of a specified resolution"""
    
    # First create a heterogeneous graph without a mask (homogeneous other than at the borders)
    graph_structure = create_static_heterogeneous_graph(image_shape, max_grid_size, None, None, use_edge_attrs, resolution, device)
    
    # Remove any nodes which entirely overlap the mask
    nodes_to_delete = get_nan_nodes(mask, graph_structure)

    edge_mask = [not (node.item() in nodes_to_delete or neighbor.item() in nodes_to_delete) for node, neighbor in graph_structure['edge_index'].T]
    graph_structure['edge_index'] = graph_structure['edge_index'][:, edge_mask]
    graph_structure['edge_attrs'] = graph_structure['edge_attrs'][edge_mask]

    node_mask = np.array(list(set(graph_structure['graph_nodes']) - set(nodes_to_delete)))
    graph_structure['graph_nodes'] = graph_structure['graph_nodes'][node_mask]
    graph_structure['mapping'] = graph_structure['mapping'][node_mask]
    graph_structure['n_pixels_per_node'] = graph_structure['n_pixels_per_node'][node_mask]
    
    # Since we removed nodes, "rename" them so they are from (0, n) for n nodes
    node_mapping = {old_node: new_node for new_node, old_node in enumerate(graph_structure['graph_nodes'])}

    graph_structure['graph_nodes'] = replace_with_map(graph_structure['graph_nodes'], node_mapping)
    graph_structure['edge_index'] = replace_with_map(graph_structure['edge_index'], node_mapping)
    
    # Remove mappings from land->cell
    for i, pixel in enumerate(mask.flatten()):
        if pixel:
            graph_structure['mapping'][:, i] = 0
            
    # Re-calculate n_pixels_per_node
    graph_structure['n_pixels_per_node'] = graph_structure['mapping'].sum(1)

    # Back to Tensors
    graph_structure['edge_index'] = torch.Tensor(graph_structure['edge_index']).type(torch.int64).to(device)
    graph_structure['edge_attrs'] = torch.Tensor(graph_structure['edge_attrs']).to(device)
    graph_structure['n_pixels_per_node'] = torch.Tensor(graph_structure['n_pixels_per_node']).to(device)
    graph_structure['mapping'] = torch.Tensor(graph_structure['mapping']).to(device)
    
    return graph_structure

"""
import torch 
A = torch.Tensor(
    [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]]
).to_sparse()
v = torch.Tensor([1, 0, 1])
v = v.to_sparse()
X = A@v

"""
