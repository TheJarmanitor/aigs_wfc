# %% Libraries
from matplotlib.image import _ImageBase
from tensorneat.problem.func_fit import FuncFit
from tensorneat.pipeline import Pipeline
from tensorneat import algorithm, genome, common
from PIL import Image
from tools.image_hashing import hash_grid

import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np


# %%
class TileProperties(FuncFit):
    def __init__(self, grid, tile_size, error_method="mse") -> None:
        self.grid = grid
        self.tile_size = tile_size
        self.error_method = error_method
        self.grid_information, self.proportion_values = self._prepare_data()

    def _prepare_data(self):
        hashed_grid = hash_grid(self.grid, tile_size=self.tile_size)
        height, width, _ = self.grid.shape
        unique_values, unique_counts = np.unique(hashed_grid, return_counts=True)
        unique_proportions = unique_counts / (height * width)
        grid_information = []

        for y in range(height):
            for x in range(width):
                normalized_x = (2 * x / (width - 1)) - 1  # Normalize x coordinate
                normalized_y = (2 * y / (height - 1)) - 1  # Normalize y coordinate
                coordinates = np.array([normalized_x, normalized_y])
                one_hot = np.eye(len(unique_values))[int(hashed_grid[x, y])]

                tile_information = np.concatenate((coordinates, one_hot))
                grid_information.append(tile_information)

        return np.array(grid_information), unique_proportions


# %%

test_grid = np.array(Image.open("images/piskel_example1.png.png"))[..., :3]
tile_grid = hash_grid(test_grid, tile_size=1)
# unique_values, unique_counts = np.unique(tile_grid, return_counts=True)
# unique_counts/144
# grid_information
tile_properties = TileProperties(test_grid, tile_size=1)
tile_properties._prepare_data()

# %%

# %%
plt.imshow(test_grid)
