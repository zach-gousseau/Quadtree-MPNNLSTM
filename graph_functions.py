import numpy as np
import torch
from torch_geometric.data import Data

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
    edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
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

def quadtree_decompose(img, padding=2, thresh=0.05, max_size=8):
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
    """

    assert is_power_of_two(max_size)

    # get image dimensions
    n, m = img.shape
    
    # initialize label array
    labels = np.ones((-(n // -max_size) * max_size, -(m // -max_size) * max_size)).astype(int)  #create_blocks(n, m, max_size)

    shape = n_padded, m_padded = labels.shape

    img = np.pad(img, ((0, n_padded-n), (0, m_padded-m)), mode='edge')
    
    # counter for current label
    global cur_label
    cur_label = 0

    # function for recursive decomposition
    def decompose(x: int, y: int, size: int):
        global cur_label

        # check if the current cell is homogeneous (contains pixels of the same color)
        l, r, t, b = x, x+size+1, y, y+size+1
        if size <= 1 and (x<m and y<n):
            if (x<n and y<m):
                labels[x, y] = cur_label
                cur_label += 1
            return
        
        is_homogeneous = (np.nanmax(
        img[max(0, l-padding): min(r+padding, shape[1]),
            max(0, t-padding): min(b+padding, shape[1])]
        ) < thresh) or (size==1)
        
        # if the cell is homogeneous, assign the same label to all its pixels
        if is_homogeneous and size < max_size:
            # print(1, labels[x:x+size, y:y+size].shape)
            if (x<n and y<m):
                labels[x:x+size, y:y+size] = cur_label
                cur_label += 1
        else:
            # otherwise, decompose the cell into four quadrants of equal size
            new_size = size // 2
            # print(x, y, new_size)
            decompose(x, y, new_size)
            decompose(x + new_size, y, new_size)
            decompose(x, y + new_size, new_size)
            decompose(x + new_size, y + new_size, new_size)
    
    # start recursive decomposition from the top left corner of the image
    for i in range(n_padded // max_size): 
        for j in range(m_padded // max_size):
            # print(i*max_size, j*max_size)
            decompose(i*max_size, j*max_size, max_size)

    # return the resulting label array
    return labels[:n, :m]



def get_adj(labels):
    """Get the adjacency matrix for a given label matrix (this could be more efficient)"""
    w, h = labels.shape
    adj_dict = {}

    for i in range(w):
        for j in range(h):
            if labels[i][j] not in adj_dict:
                adj_dict[labels[i][j]] = []

            neighbors = set()

            if i != 0:
                neighbors.add(labels[i-1][j])
            if i != w-1:
                neighbors.add(labels[i+1][j])
            if j != 0:
                neighbors.add(labels[i][j-1])
            if j != h-1:
                neighbors.add(labels[i][j+1])

            try:
                neighbors.remove(labels[i][j])
            except KeyError:
                pass

            for neighbor in neighbors:
                if neighbor not in adj_dict[labels[i][j]]:
                    adj_dict[labels[i][j]].append(neighbor)

    return adj_dict

def dist(node0, node1, xx, yy):
    return np.sqrt((yy[node0] - yy[node1])**2 + (xx[node0] - xx[node1])**2)

def dist_xy(node0, node1, xx, yy):
    return np.array((xx[node0] - xx[node1], yy[node0] - yy[node1]))

def flatten(img, labels, num_features=4):
    graph_nodes = sorted(np.unique(labels))

    map_pixel_to_graph = {}
    p = 0
    for i in range(img.shape[-3]):
        for j in range(img.shape[-2]):
            map_pixel_to_graph[p] = labels[i, j]
            p += 1

    map_graph_to_pixel = {n: [] for n in graph_nodes}
    for p, n in map_pixel_to_graph.items():
        map_graph_to_pixel[n].append(p)

    mappings = {
        'p->n': map_pixel_to_graph,
        'n->p': map_graph_to_pixel
    }

    img_flat = img.reshape(-1, img.shape[-2]*img.shape[-3], num_features, order='C')
    data = np.ndarray((img.shape[0], len(graph_nodes), num_features))

    for i, n in enumerate(graph_nodes):
        idx = mappings['n->p'][n]
        data[:, i] = np.mean(img_flat[:, idx], 1)
    return data, graph_nodes, mappings

def unflatten(img_flat, graph_nodes, mappings, image_shape=(8, 8)):
    img = np.ndarray((img_flat.shape[0], np.prod(image_shape), img_flat.shape[-1]))
    # print(graph_nodes)
    for n in graph_nodes:
        for p in mappings['n->p'][n]:
            img[:, p] = img_flat[:, n]

    return img.reshape((img_flat.shape[0], *image_shape, img_flat.shape[-1]))

def image_to_graph(img, num_features=4, thresh=0.05, max_grid_size=8):
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

    labels = quadtree_decompose(img0, thresh=thresh, max_size=max_grid_size)

    data, graph_nodes, mappings = flatten(img, labels, num_features)

    if np.any(np.isnan(data)):
        raise ValueError('NaNs in data!')

    xx, yy = data[0, ..., 1]*image_shape[1], data[0, ..., 2]*image_shape[0]
    
    # Get sizes for each graph node
    # node_sizes = np.array([np.sum(labels == label) for label in graph_nodes]).astype(float)
    node_sizes = np.unique(labels, return_counts=True)[1]

    # Pseudo-normalize
    node_sizes = node_sizes / ((max_grid_size/2)**2)
    node_sizes = np.tile(node_sizes, (img.shape[0], 1))
    data = np.concatenate([data, np.expand_dims(node_sizes, -1)], -1)

    adj_dict = get_adj(labels)
    # print(list(np.array(graph_nodes[1:]) - np.array(graph_nodes[:-1])))

    # Get distances
    distances = {}
    for node in graph_nodes:
        distances[node] = {}
        for neighbor in adj_dict[node]:
            distances[node][neighbor] = dist(node, neighbor, xx, yy) #/ max_grid_size *8

    # Normalize data to (0, 1)
    # data = data / np.max(data, 1)[:, None, :]

    out = dict(
        labels=labels,
        distances=distances,
        data=data,
        graph_nodes=graph_nodes,
        adj_dict=adj_dict,
        mappings=mappings,
    )

    return out