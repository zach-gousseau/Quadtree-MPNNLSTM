import numpy as np
import warnings
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from collections import defaultdict
import numba as nb

from utils import minmax

CONDITIONS = [
    'max_larger_than',
    'max_smaller_than',
    'min_larger_than',
    'min_smaller_than',
]

def plot_contours(ax, labels):
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

def quadtree_decompose(img, padding=0, thresh=0.05, max_size=8, mask=None, transform_func=None, condition='max_larger_than'):
    """
    Perform quadtree decomposition on an image.

    This function decomposes the input image into a quadtree by dividing the image
    into four quadrants of equal size and repeating the process recursively until
    the maximum value in each quadrant is either below some threshold or contains a 
    single pixel. 
    
    Optionally, provide a mask which will ensure not cell contains any value which 
    falls within the mask. Masked pixels are assigned a label of -1.
    
 
    Parameters:
    img (np.ndarray): Input image with shape (height, width).
    padding (int, optional): Padding to add around each cell when checking the splitting criteria. Default is 0.
    thresh (float, optional): Threshold to use as splitting criteria. Default is 0.05.
    max_size (int, optional): Maximum grid cell size. Default is 8.
    mask (np.ndarray, optional): Boolean mask. 
    transform_func (optional): Function to apply to the input image used for criteria evaluation

    Returns:
    np.ndarray: Array of shape (height, width) containing the labels for each pixel in the image.
        Note: A '-1' label means that the pixel is invalid (according to the provided mask)
    """
    
    assert max_size & (max_size - 1) == 0
    
    assert condition in CONDITIONS

    n, m = img.shape
    
    n_padded, m_padded = int(-(n // -max_size) * max_size), int(-(m // -max_size) * max_size)
    
    # Pad the image to match the labels array
    img = np.pad(img, ((0, n_padded - n), (0, m_padded-m)), mode='edge')
    # img = F.pad(img.unsqueeze(0), (0, m_padded-m, 0, n_padded-n), mode='replicate').squeeze(0)
    
    # Apply transformation if desired
    img_for_criteria = transform_func(img) if transform_func else img

    mapping = np.zeros(shape=(n*m, n, m))  # Largest possible mapping 
    
    # Build initial stack using each of the cells in the base grid
    stack = []
    for i in range(n_padded // max_size): 
        for j in range(m_padded // max_size):
            stack.append((i*max_size, j*max_size, max_size))

    i = 0
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

            mapping[i, x, y] = 1
            i += 1
            continue
        
        cell = img_for_criteria[
            max(0, l-padding): min(r+padding, n_padded),
            max(0, t-padding): min(b+padding, m_padded)
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
        overlaps_mask = mask is not None and any_2d(mask[max(0, l-padding): min(r+padding, n_padded), max(0, t-padding): min(b+padding, m_padded)])
        # overlaps_mask = mask is not None and torch.any(mask[max(0, l-padding): min(r+padding, shape[1]), max(0, t-padding): min(b+padding, shape[1])])
        split_cell = split_cell or (overlaps_mask)
        
        # Perform splitting if criteria is met, otherwise set all pixels to the current label
        if split_cell:
            new_size = size // 2
            stack.append((x, y, new_size))
            stack.append((x + new_size, y, new_size))
            stack.append((x, y + new_size, new_size))
            stack.append((x + new_size, y + new_size, new_size))
        else:
            mapping[i, x:x+size, y:y+size] = 1
            i += 1
    
    return torch.Tensor(mapping[:i].reshape(-1, m*n))

import threading
lock = threading.Lock()

def process_stack(stack):
    while True:
        # Pop an item from the stack
        with lock:
            if len(stack) == 0:
                break
            item = stack.pop()

        # Process the item
        quadtree_decompose(*item)

def quadtree_decompose_multithread(img, num_threads=4, **kwargs):
    # Build initial stack using each of the cells in the base grid
    stack = []
    max_size = kwargs.get('max_size', 8)
    n, m = img.shape
    for i in range(-(n // -max_size)):
        for j in range(-(m // -max_size)):
            stack.append((img, 0, 0, max_size, None, None, None, None, None, None, None, None, None, None))

    # Create and start threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=process_stack, args=(stack,))
        threads.append(t)
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

def mapping_to_labels(mapping, image_shape):
    graph_nodes = torch.arange(0, mapping.shape[0]).to(mapping.device)
    return torch.einsum('i,ijk->ijk', graph_nodes+1, mapping.reshape(-1, *image_shape)).sum(0).type(torch.int) - 1

def create_graph_structure(mapping, image_shape, xx=None, yy=None, calculate_distances=True):
    """
    Create a graph structure with undirected edge, with the distance between nodes 
    as edge attributes.

    :param torch.Tensor: mapping: Sparse matrix of mapping between grid-space and mesh-space
    :param tuple: image_shape: Shape of the original grid
    :param torch.Tensor: xx: x position of each of the nodes in the mapping object
    :param torch.Tensor: yy: y position of each of the nodes in the mapping object 
    :param bool: calculate_distances: Whether to calculate the distance between each node, or just use 1. 
    
    :return torch_geometric.data.Data: Data object containing the edge indices and edge attributes for the graph.
    """
    labels = mapping_to_labels(mapping, image_shape).cpu().numpy()

    w, h = labels.shape

    if calculate_distances:
        assert xx is not None and yy is not None, 'Provide x and y positions if distances are desired!'

    edge_sources = []
    edge_targets = []
    edge_attrs = []
    
    graph_nodes = np.arange(0, mapping.shape[0])
    for node in graph_nodes:
        idx, jdx = np.where(labels==node)
        b = max(idx.min() - 1, 0)
        t = min(idx.max() + 1, w-1)
        r = max(jdx.min() - 1, 0)
        l = min(jdx.max() + 1, h-1)

        neighbors = set(labels[b:t+1, r:l+1].flatten())
        try:
            neighbors.remove(-1)
        except KeyError:
            pass

        for neighbor in neighbors:

            if calculate_distances:
                edge_attr = dist(node, neighbor, xx, yy)
            else:
                edge_attr = 1

            edge_sources.append(node)
            edge_targets.append(neighbor)
            edge_attrs.append(edge_attr)

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_attrs = torch.tensor(edge_attrs, dtype=torch.float32)
    
    graph_structure = Data(edge_index=edge_index, edge_attr=edge_attrs)
    graph_structure.image_shape = image_shape
    return graph_structure

def dist(node0, node1, xx, yy):
    return torch.sqrt((yy[node0] - yy[node1])**2 + (xx[node0] - xx[node1])**2)

def dist_xy(node0, node1, xx, yy):
    return np.array((xx[node0] - xx[node1], yy[node0] - yy[node1]))
    

def flatten(img, mapping, n_pixels_per_node):
    """
    Given an input image of dimension (n_samples, w, h, channels) and a labels array of dimension (w, h)
    which correspond to the mesh node to which each pixel in the original image belong, convert the image to 
    its mesh representation. Note this could also be done using np.tensordot(b, a, axes=((1, 2), (1, 0)))
    img: (n_samples, w, h, c)"""
    assert len(img.shape) == 4, f'array should be 4-dimensional (n_samples, w, h, c); got {img.shape}'
    n_samples, w, h, c = img.shape
    
    # (n_samples, w, h, c) -> (c, n_samples, w*h)
    img_flattened = torch.moveaxis(img, -1, 0).reshape(c, n_samples, w*h)
    
    # Compute mean values for each graph node
    while True:
        data = img_flattened @ mapping.T.to_dense() / n_pixels_per_node

        if data.isnan().any():
            warnings.warn('Matrix multiplication in flatten() failed, trying again.')
        else:
            break

    # (c, n_samples, w*h) -> (n_samples, w*h, c)
    data = torch.moveaxis(data, 0, -1)

    return data

def unflatten(data, mapping, image_shape):
    """Create an image of shape (n, w, h, c) for n samples of dimensions w, h and c channels"""
    data = torch.moveaxis(data, -1, 0)
    img = (data @ mapping.to_dense()).reshape(*data.shape[:-1], *image_shape)
    return torch.moveaxis(img, 0, -1)
    

def image_to_graph_pixelwise(img, mask=None):
    pass


def image_to_graph(img, thresh=0.05, max_grid_size=8, mask=None, transform_func=None, condition='max_larger_than'):
    """
    Decomposes an image into a quadtree and then generates a graph representation of the image
    using quadtree decomposition. The graph nodes are the centroids of the quadtree cells and the
    edges between nodes represent the distances between the corresponding image patches.

    Parameters:
    img (np.ndarray): Input image with shape (batch_size, height, width, channels).
    num_features (int, optional): Number of features to include in the data for each graph node. Default is 4.
    thresh (float, optional): Threshold for quadtree decomposition. Default is 0.05.

    Returns:
    dict: Dictionary containing the following information:
        labels (np.ndarray): Array of shape (height, width) containing the labels for each pixel in the image.
        distances (dict): Dictionary of distances between pairs of graph nodes.
        data (np.ndarray): Array of shape (batch_size, num_nodes, num_features+1) containing the data for each graph node. The last column is the size of the graph node.
        graph_nodes (list): List of graph nodes.
        adj_dict (dict): Dictionary of adjacencies for each graph node.
        mappings (dict): Dictionary mapping graph node labels to indices.

    TODO: Add ability to choose which channel to use in the decomposition.
    """

    assert len(img.shape) == 4, f'array should be 4-dimensional (n_samples, w, h, c); got {img.shape}'

    if torch.any(torch.isnan(img)):
        raise ValueError(f'Found NaNs in image data {torch.sum(torch.isnan(img))} / {np.prod(img.shape)}')

    img_for_decompose, _ = torch.max(img[..., 0], 0)  # For multi-step inputs
    n_samples, h, w, c = img.shape

    image_shape = img_for_decompose.shape
    img_for_decompose = img_for_decompose.cpu().detach().numpy()

    if thresh == -np.inf:
        return image_to_graph_pixelwise(img, mask)
    
    mapping = quadtree_decompose(
        img_for_decompose,
        thresh=thresh, 
        max_size=max_grid_size,
        mask=mask,
        transform_func=transform_func,
        condition=condition
        )

    n_pixels_per_node = mapping.sum(1)
    graph_nodes = np.arange(0, mapping.shape[0])

    mapping, n_pixels_per_node = mapping.to(img.device), n_pixels_per_node.to(img.device)
    
    data = flatten(img, mapping, n_pixels_per_node)

    if torch.any(torch.isnan(data)):
        raise ValueError(f'Found NaNs in graph data {torch.sum(torch.isnan(data))} / {np.prod(data.shape)}')
    
    # Get sizes for each graph node (TODO: scale by latitude)
    node_sizes = n_pixels_per_node

    # Pseudo-normalize and add node sizes as feature 
    node_sizes = torch.Tensor(node_sizes) / ((max_grid_size/2)**2)
    node_sizes = node_sizes.repeat((n_samples, *[1]*len(node_sizes.shape)))

    data = torch.cat([data, node_sizes.unsqueeze(-1)], -1)

    out = dict(
        data=data,
        graph_nodes=graph_nodes,
        mapping=mapping,
        n_pixels_per_node=n_pixels_per_node,
    )

    return out