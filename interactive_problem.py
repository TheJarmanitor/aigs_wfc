from tensorneat.problem import BaseProblem

# %% Libraries
from matplotlib.image import _ImageBase
from tensorneat import genome, common
from interactive_NEAT import InteractiveNEAT
from PIL import Image
from tools.image_hashing import hash_grid, label_grids
from tools.visualize_cppn import visualize_output_grid

import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from jax import vmap

from tensorneat.common import State


class InteractiveGrid(BaseProblem):
    def __init__(self, grid_size) -> None:
        self.grid_size = grid_size

        self._prepare_data()

    def setup(self, state: State = State()):
        return state

    def _prepare_data(self):
        grid_info_list = []
        width, height = self.grid_size

        grid_information = []

        for y in range(width):
            for x in range(height):
                normalized_x = (2 * x / (width - 1)) - 1  # Normalize x coordinate
                normalized_y = (2 * y / (height - 1)) - 1  # Normalize y coordinate
                coordinates = np.array([normalized_x, normalized_y])
                grid_information.append(coordinates)

        grid_info_list.append(np.array(grid_information))
        self.input_data = jnp.concatenate(grid_info_list, axis=0)

    @property
    def inputs(self):
        return self.input_data

    @property
    def input_shape(self):
        return self.input_data.shape
