import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import minmax

def plot_contours(ax, labels):
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            try:
                if labels[i][j] != labels[i][j+1]:
                    ax.plot([j+0.5, j+0.5], [i-0.5, i+0.5], c='k', lw=1)
            except IndexError:
                pass

            try:
                if labels[i][j] != labels[i+1][j]:
                    ax.plot([j-0.5, j+0.5], [i+0.5, i+0.5], c='k', lw=1)
            except IndexError:
                pass

def create_graph_structure(graph_nodes, distances):
    """
    Create a graph structure with undirected edge, with the distance between nodes 
    as edge attributes.

    :param: graph_nodes: List of graph nodes
    :param: distances: Nested dictionary of giving the distances between nodes, e.g.
        {
            node0: {node1: 0.2, node2: 0.3, ...},
            node1: {node0: 0.2, node3: 0.1, ...},
            ...
        }
    Returns:
    torch_geometric.data.Data: Data object containing the edge indices and edge attributes for the graph.
    """
    edge_sources = []
    edge_targets = []
    edge_attrs = []
    for node in graph_nodes:
        for neighbor in distances[node]:
            edge_sources.append(node)
            edge_targets.append(neighbor)
            edge_attrs.append(distances[node][neighbor])

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_attrs = torch.tensor(edge_attrs, dtype=torch.float32)
    return Data(edge_index=edge_index, edge_attr=edge_attrs)

def create_blocks(M, N, B):
    num_rows = -(M // -B)
    num_cols = -(N // -B)
    label = np.arange(num_rows * num_cols).reshape(num_rows, num_cols)
    blocks = np.repeat(label, B, axis=0)
    blocks = np.repeat(blocks, B, axis=1)
    return blocks[:M, :N]

def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)

def quadtree_decompose(img, padding=0, thresh=0.05, max_size=8, mask=None):
    """
    Decompose an image into a quadtree.

    This function decomposes the input image into a quadtree by dividing the image
    into four quadrants of equal size and repeating the process recursively until
    each quadrant is either homogeneous (the variance within the cell is lower than
    the specified threshold) or contains a single pixel.
 
    Parameters:
    img (np.ndarray): Input image with shape (height, width).
    padding (int, optional): Padding to add around the image when checking for homogeneity. Default is 2.
    thresh (float, optional): Threshold for determining homogeneity. Default is 0.05.

    Returns:
    np.ndarray: Array of shape (height, width) containing the labels for each pixel in the image.
        Note: A '-1' label means that the pixel is invalid (according to the provided mask)
    """

    assert is_power_of_two(max_size)

    # get image dimensions
    n, m = img.shape
    
    # initialize label array
    labels = np.full((-(n // -max_size) * max_size, -(m // -max_size) * max_size), -1, dtype=int)#.astype(int)  #create_blocks(n, m, max_size)
    # labels = np.ndarray((-(n // -max_size) * max_size, -(m // -max_size) * max_size)).astype(int)  #create_blocks(n, m, max_size)

    shape = n_padded, m_padded = labels.shape

    img = np.pad(img, ((0, n_padded-n), (0, m_padded-m)), mode='edge')
    
    # counter for current label
    global cur_label
    cur_label = 0

    # function for recursive decomposition
    def decompose(x: int, y: int, size: int):
        global cur_label

        l, r, t, b = x, x+size+1, y, y+size+1

        # If the current cell is a single pixel, do not continue decomposing
        if size <= 1: 
            if x<n and y<m:
                # Check against mask if provided
                invalid = False
                if mask is not None:
                    if mask[x, y]:
                        invalid = True

                # Assign label to single pixel
                if not invalid:
                    labels[x, y] = cur_label
                    cur_label += 1
                else:
                    if not mask[x, y]:
                        labels[x, y] = cur_label
                        cur_label += 1
                    
            return
        
        img_region = img[max(0, l-padding): min(r+padding, shape[1]),
                         max(0, t-padding): min(b+padding, shape[1])]
        
        img_region = np.abs(np.abs(img_region - 0.5) - 0.5)
        is_homogeneous = (np.nanmax(img_region) < thresh) or (size==1)
        
        # Check if the cell overlaps any invalid pixels 
        if mask is not None:
            overlaps_mask = np.any(
            mask[max(0, l-padding): min(r+padding, shape[1]),
                max(0, t-padding): min(b+padding, shape[1])]
            )
        else:
            overlaps_mask = False
        
        # if the cell is homogeneous, smaller than the maximum grid size 
        # and does not overlap the mask, assign the same label to all its pixels
        if is_homogeneous and (size < max_size) and (not overlaps_mask):
            if (x<n and y<m):
                labels[x:x+size, y:y+size] = cur_label
                cur_label += 1
        else:
            # otherwise, decompose the cell into four quadrants of equal size
            new_size = size // 2
            decompose(x, y, new_size)
            decompose(x + new_size, y, new_size)
            decompose(x, y + new_size, new_size)
            decompose(x + new_size, y + new_size, new_size)
    
    # start recursive decomposition from the top left corner of the image
    for i in range(n_padded // max_size): 
        for j in range(m_padded // max_size):
            decompose(i*max_size, j*max_size, max_size)

    # return the resulting label array
    return labels[:n, :m]



def get_adj(labels, xx=None, yy=None, calculate_distances=True, edges_at_corners=False):
    """Get the adjacency matrix for a given label matrix (this could be more efficient)"""
    w, h = labels.shape
    adj_dict = {}

    if calculate_distances:
        assert xx is not None and yy is not None, 'Provide x and y positions if distances are desired!'

    for i in range(w):
        for j in range(h):

            node = labels[i][j]
            
            # Skip if the current label is invalid (-1)
            if node == -1:
                continue

            if node not in adj_dict:
                adj_dict[node] = {}

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
                    if calculate_distances:
                        adj_dict[node][neighbor] = dist(node, neighbor, xx, yy)
                    else:
                        adj_dict[node][neighbor] = 1


    return adj_dict

def dist(node0, node1, xx, yy):
    return np.sqrt((yy[node0] - yy[node1])**2 + (xx[node0] - xx[node1])**2)

def dist_xy(node0, node1, xx, yy):
    return np.array((xx[node0] - xx[node1], yy[node0] - yy[node1]))

def get_graph_nodes(labels):
    graph_nodes = np.unique(labels)

    # Remove -1 from the list of graph nodes if it exists (ie if a mask was provided)
    if -1 in graph_nodes:
        return graph_nodes[1:]
    else:
        return graph_nodes
    

def flatten(img, labels):
    """
    Given an input image of dimension (n_samples, w, h, channels) and a labels array of dimension (w, h)
    which correspond to the mesh node to which each pixel in the original image belong, convert the image to 
    its mesh representation.
    img: (n_samples, w, h, c)"""

    n_samples, w, h, num_features = img.shape
    graph_nodes = get_graph_nodes(labels)

    # Get mappings from pixel space to graph space and vice versa
    labels_flat = labels.flatten(order='C')
    map_pixel_to_graph = {i: n for i, n in enumerate(labels_flat) if n!=-1}  # One-to-one mapping
    
    # map_graph_to_pixel = {n: np.where(labels_flat == i)[0] for i, n in enumerate(graph_nodes)}  # One-to-many mapping
    map_graph_to_pixel = {n: [] for n in graph_nodes}  # One-to-many mapping  or use defaultdict(list)
    for i, n in enumerate(labels_flat):
        if n != -1:
            map_graph_to_pixel[n].append(i)
            
    # values, inverse = np.unique(labels_flat, return_inverse=True)
    # map_graph_to_pixel = {value: np.where(inverse == i)[0] for i, value in enumerate(values)}

    # Store mappings - TODO: this can be its own class
    mappings = {
        'p->n': map_pixel_to_graph,
        'n->p': map_graph_to_pixel
    }

    # Compute mean values for each graph node
    # TODO: Would be nice if we didn't have to flatten first
    img_flat = img.reshape(n_samples, w*h, num_features)#, order='C')  # Assume it flattens column-wise !

    # Slow version
    # data = torch.empty((n_samples, len(graph_nodes), num_features), dtype=img.dtype)
    # for i, n in enumerate(graph_nodes):
    #     idx = mappings['n->p'][n]
    #     data[:, i] = torch.mean(img_flat[:, idx], 1)

    data = grouped_mean_along_axis_2d(img_flat, labels_flat, axes=(0, 1))
    # data = torch.Tensor(data).type(torch.float32)
    return data, mappings

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


def unflatten(img_flat, graph_nodes, mappings, image_shape=(8, 8), nan_value=0):
    """Create an image of shape (n, w, h, c) for n samples of dimensions w, h and c channels"""
    
    # Start with an array of dimension (n, w*h, c) since spatial indexing is column-wise flattened
    img = np.full((img_flat.shape[0], np.prod(image_shape), img_flat.shape[-1]), nan_value, dtype=float)
    for n in graph_nodes:
        img[:, mappings['n->p'][n]] = np.expand_dims(img_flat[:, n], 1)

    # Reshape to (n, w, h, c)
    img = img.reshape((img_flat.shape[0], *image_shape, img_flat.shape[-1]))
    return img

def image_to_graph_pixelwise(img, mask=None):
    """TODO: implement masking"""

    img0 = np.max(img[..., 0], 0)  # For multi-step inputs

    labels = np.arange(np.prod(img0.shape)).reshape(img0.shape)
    graph_nodes = np.arange(np.prod(img0.shape))

    data = img.reshape((img.shape[0], img.shape[1]*img.shape[2], img.shape[3]))

    node_sizes = np.ones((data.shape[0], len(graph_nodes)))
    data = np.concatenate([data, np.expand_dims(node_sizes, -1)], -1)

    mappings = {}
    mappings['n->p'] = {n: [n] for n in graph_nodes}
    mappings['p->n'] = {n: n for n in graph_nodes}

    # Distances are all the same so don't bother calculating them. Uses '1' as the distance for each edge.
    distances = get_adj(labels, calculate_distances=False)

    out = dict(
        labels=labels,
        distances=distances,
        data=data,
        graph_nodes=graph_nodes,
        mappings=mappings,
    )

    return out


def image_to_graph(img, thresh=0.05, max_grid_size=8, mask=None):
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
    img0 = np.max(img[..., 0], 0)  # For multi-step inputs
    image_shape = img0.shape

    if np.any(np.isnan(img0)):
        raise ValueError('NaNs in data!!')

    if thresh == -np.inf:
        return image_to_graph_pixelwise(img, mask)

    labels = quadtree_decompose(img0, thresh=thresh, max_size=max_grid_size, mask=mask)
    graph_nodes = get_graph_nodes(labels)

    data, mappings = flatten(img, labels)

    if np.any(np.isnan(data)):
        raise ValueError('NaNs in data!!')

    xx, yy = data[0, ..., 1]*image_shape[1], data[0, ..., 2]*image_shape[0]
    
    # Get sizes for each graph node
    # node_sizes = np.array([np.sum(labels == label) for label in graph_nodes]).astype(float)
    node_sizes = np.unique(labels, return_counts=True)[1]

    # Remove first item if invalid pixels exist 
    # Note that np.unique() provides a sorted output, so this removes the count
    # of '-1' elements.
    if -1 in labels:
        node_sizes = node_sizes[1:]

    # Make sure nothing has gone wrong 
    assert len(node_sizes) == len(graph_nodes)

    # Pseudo-normalize
    node_sizes = node_sizes / ((max_grid_size/2)**2)
    node_sizes = np.tile(node_sizes, (img.shape[0], 1))
    data = np.concatenate([data, np.expand_dims(node_sizes, -1)], -1)

    distances = get_adj(labels, xx=xx, yy=yy, calculate_distances=False)

    # Normalize data to (0, 1)
    # data = data / np.max(data, 1)[:, None, :]

    out = dict(
        labels=labels,
        distances=distances,
        data=data,
        graph_nodes=graph_nodes,
        mappings=mappings,
    )

    return out