from graph_functions import CONDITIONS
import numpy as np
import torch
import random
import cupyx.scipy.ndimage
import xarray as xr
import numba as nb

def dist_from_05(arr):
    return abs(abs(arr - 0.5) - 0.5)

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

from numba import cuda

@cuda.jit
def quadtree_decompose_jit(img, labels, padding, thresh, max_size, mask, transform_func, condition):
    n, m = img.shape
    
    # Initialize label array with the base grid (maximum grid cell size) while 
    # Note that the initial label array may be larger than the original image since
    # we do not want to cut off any base grid cells
    
    n_padded, m_padded = labels.shape
    
    # Pad the image to match the labels array
    img = cuda.pad(img, ((0, n_padded-n), (0, m_padded-m)), mode='edge')
    
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
            labels[x:x+size, y:y+size] = cur_label
            cur_label += 1

        # Since the padded zones are not useful, we remove them
        if n_padded != n or m_padded != m:
            labels = labels[:n, :m]
            
        return labels



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
            labels[x:x+size, y:y+size] = cur_label
            cur_label += 1
    
    return labels[:n, :m]


import cupy as cp

def quadtree_decompose_gpu(img, padding=0, thresh=0.05, max_size=8, mask=None, transform_func=None, condition='max_larger_than'):
    assert max_size & (max_size - 1) == 0
    assert condition in CONDITIONS

    n, m = img.shape
    labels = cp.full((-(n // -max_size) * max_size, -(m // -max_size) * max_size), -1, dtype=int)
    shape = n_padded, m_padded = labels.shape
    # img = cp.pad(img, ((0, n_padded-n), (0, m_padded-m)), mode='edge')
    img = cupyx.scipy.ndimage.pad(img, ((0, n_padded-n), (0, m_padded-m)), mode='edge')
    img_for_criteria = transform_func(img) if transform_func else img

    cur_label = 0
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
            split_cell = cp.max(cell) > thresh 
        elif condition == 'max_smaller_than':
            split_cell = cp.max(cell) < thresh 
        elif condition == 'min_larger_than':
            split_cell = cp.min(cell) > thresh 
        elif condition == 'min_smaller_than':
            split_cell = cp.min(cell) < thresh
        
        overlaps_mask = mask is not None and cp.any(mask[max(0, l-padding): min(r+padding, shape[1]), max(0, t-padding): min(b+padding, shape[1])])
        split_cell = split_cell or (overlaps_mask)
        
        if split_cell:
            new_size = size // 2
            stack.append((x, y, new_size))
            stack.append((x + new_size, y, new_size))
            stack.append((x, y + new_size, new_size))
            stack.append((x + new_size, y + new_size, new_size))
        else:
            labels[x:x+size, y:y+size] = cur_label
            cur_label += 1
    
    return labels[:n, :m].get()



def quadtree_decompose_cuda(img, padding=0, thresh=0.05, max_size=8, mask=None, transform_func=None, condition='max_larger_than'):
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

    return labels[:n, :m].cpu().numpy()

# from graph_functions import q

if __name__ == '__main__':

    # np.random.seed(42)
    # random.seed(42)
    # torch.manual_seed(42)

    ds = xr.open_zarr('data/era5_hb_daily.zarr')    # ln -s /home/zgoussea/scratch/era5_hb_daily.zarr data/era5_hb_daily.zarr
    mask = np.isnan(ds.siconc.isel(time=150)).values

    arr = ds.isel(time=0).siconc.values
    arr = torch.from_numpy(arr).cuda()



    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()

    for i in range(100):
        quadtree_decompose_jit(
            img=arr,
            padding=0,
            thresh=0.15,
            max_size=8,
            mask=mask,
            transform_func=dist_from_05,
            condition='max_larger_than'
            )

    pr.disable()
    stats = pstats.Stats(pr).sort_stats('time')
    stats.print_stats(10)