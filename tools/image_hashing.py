import numpy as np
from PIL import Image
import hashlib


def get_hash(tile):
    hash = hashlib.md5(tile.tobytes())
    hash = int.from_bytes(hash.digest()[:5], "big")
    return hash


def hash_grid(grid, tile_size, return_label=True):
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
    if return_label:
        label_grid = hashed_grid.copy()
        unique_hashes = np.unique(hashed_grid)
        unique_dict = dict(zip(unique_hashes, list(range(len(unique_hashes)))))

        for key, new_value in unique_dict.items():
            label_grid[hashed_grid == key] = new_value
        return label_grid

    return hashed_grid
