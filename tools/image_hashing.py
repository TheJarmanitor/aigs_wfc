import numpy as np
from PIL import Image
import hashlib


def get_hash(tile):
    hash = hashlib.md5(tile.tobytes())
    hash = int.from_bytes(hash.digest()[:5], "big")
    return hash


def hash_grid(grid, tile_size):
    W, H, _ = grid.shape
    hashed_grid = np.zeros((H // tile_size, W // tile_size))
    for i in range(hashed_grid.shape[0]):
        for j in range(hashed_grid.shape[1]):
            hashed_grid[i, j] = get_hash(
                grid[
                    i * tile_size : tile_size * (i + 1),
                    j * tile_size : tile_size * (j + 1),
                ]
            )
    return hashed_grid

def label_grids(grids, return_unique_values=True):
    if not isinstance(grids, list):
        grids = [grids]
    # label_grid = hashed_grid.copy()
    master_list = np.concatenate([grid.flatten() for grid in grids])
    unique_values = np.unique(master_list)
    unique_dict = dict(zip(unique_values, list(range(len(unique_values)))))
    label_grids = []
    for grid in grids:
        copy_grid = grid.copy()
        for key, new_value in unique_dict.items():
            copy_grid[grid == key] = new_value
        label_grids.append(copy_grid)
    if return_unique_values:
        return label_grids, unique_values
    return label_grids
