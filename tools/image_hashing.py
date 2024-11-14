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
    hashed_grid = np.zeros((H // tile_size, W // tile_size))
    for i in range(hashed_grid.shape[0]):
        for j in range(hashed_grid.shape[1]):
            tile = grid[
                i * tile_size : tile_size * (i + 1),
                j * tile_size : tile_size * (j + 1),
            ]
            hash = get_hash(tile)
            hashed_grid[i, j] = hash
            hash_dict[hash] = tile

    return hashed_grid




def label_grids(grids):
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
    return label_grids, unique_labels

def hash_to_tile, unique_values)


# def get_proportions(grid, unique_labels):
#     size = grid.size
#     unique, counts = np.unique(grid, return_counts=True)
#     value_dict = dict(zip(unique, counts))
#     proportions = [value_dict[x]/size  if x in value_dict.keys() else 0 for x in unique_labels]

#     return np.array(proportions)
