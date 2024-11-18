import numpy as np
from PIL import Image
import hashlib


def get_hash(tile):
    hash = hashlib.md5(tile.tobytes())
    hash = int.from_bytes(hash.digest()[:5], "big")
    return hash


def hash_grid(grid, tile_size, return_dict=False):
    hash_dict = {}
    W, H, _ = grid.shape
    hashed_grid = np.zeros((W // tile_size, H // tile_size))
    for i in range(0, W, tile_size):
        for j in range(0, H, tile_size):
            tile = grid[
                i:i+tile_size,
                j:j+tile_size,
            ]
            hash = get_hash(tile)
            hashed_grid[i//tile_size, j//tile_size] = hash
            hash_dict[hash] = tile
    if return_dict:
        return hashed_grid, hash_dict
    return hashed_grid




def label_grids(grids, hash_dict=None):
    if not isinstance(grids, list):
        grids = [grids]
    # label_grid = hashed_grid.copy()
    master_list = np.concatenate([grid.flatten() for grid in grids])
    unique_values = np.unique(master_list)
    unique_labels = list(range(len(unique_values)))
    unique_dict = dict(zip(unique_values, unique_labels))
    label_grids = []
    for grid in grids:
        copy_grid = grid.copy()
        for key, new_value in unique_dict.items():
            copy_grid[grid == key] = new_value
        label_grids.append(copy_grid)
        if hash_dict is not None:
            label_dict = {}
    return label_grids, unique_labels



# def get_proportions(grid, unique_labels):
#     size = grid.size
#     unique, counts = np.unique(grid, return_counts=True)
#     value_dict = dict(zip(unique, counts))
#     proportions = [value_dict[x]/size  if x in value_dict.keys() else 0 for x in unique_labels]

#     return np.array(proportions)
