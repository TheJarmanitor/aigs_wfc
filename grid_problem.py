# %% Libraries
from matplotlib.image import _ImageBase
from tensorneat.problem.func_fit import FuncFit
from tensorneat.pipeline import Pipeline
from tensorneat import algorithm, genome, common
from PIL import Image
from tools.image_hashing import hash_grid, label_grids

import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from jax import vmap

from tools.visualize_labeled import visualize_labeled, network_dict

# %%
class TileProperties(FuncFit):
    def __init__(self, grids, tile_size, hash=True, error_method="mse") -> None:
        self.grids = grids if isinstance(grids, list) else [grids]
        self.tile_size = tile_size
        self.hash = hash
        self.error_method = error_method
        self._prepare_data()

    def _prepare_data(self):
        grid_info_list = []
        one_hot_list = []
        if self.hash:
            hashed_grids = [hash_grid(grid, tile_size=1) for grid in self.grids]
        else:
            hashed_grids = self.grids
        labeled_grids, unique_labels = label_grids(hashed_grids)
        for grid in labeled_grids:
            height, width = grid.shape
            _, unique_counts = np.unique(grid, return_counts=True)
            # unique_proportions = unique_counts / (height * width)
            # proportion_list.append(unique_proportions)
            grid_information = []
            one_hot_output = []

            for y in range(height):
                for x in range(width):
                    normalized_x = (2 * x / (width - 1)) - 1  # Normalize x coordinate
                    normalized_y = (2 * y / (height - 1)) - 1  # Normalize y coordinate
                    coordinates = np.array([normalized_x, normalized_y])
                    one_hot = np.eye(len(unique_labels))[int(grid[x, y])]

                    grid_information.append(coordinates)
                    one_hot_output.append(one_hot)
            grid_info_list.append(np.array(grid_information))
            one_hot_list.append(np.array(one_hot_output))

        self.input_data = jnp.concatenate(grid_info_list, axis=0)
        self.output_data = jnp.concatenate(one_hot_list, axis=0)
        self.unique_labels = unique_labels

    def evaluate(self, state, randkey, act_func, params):

        predict = vmap(act_func, in_axes=(None, None, 0))(
                    state, params, self.inputs
                )
        predict = jnp.argmax(predict, axis=1)
        predict = jnp.eye(len(self.unique_labels))[predict]
        predict_proportions = np.sum(predict, axis=0)/(predict.shape[0]*predict.shape[1])
        target_proportions = np.sum(self.targets, axis=0)/(self.output_shape[0]*self.output_shape[1])

        if self.error_method == "mse":
                loss = jnp.mean((predict_proportions - target_proportions) ** 2)

        elif self.error_method == "rmse":
            loss = jnp.sqrt(jnp.mean((predict_proportions - target_proportions) ** 2))

        elif self.error_method == "mae":
            loss = jnp.mean(jnp.abs(predict_proportions - target_proportions))

        elif self.error_method == "mape":
            loss = jnp.mean(jnp.abs((predict_proportions - target_proportions) / target_proportions))

        else:
            raise NotImplementedError

        return -loss

    @property
    def inputs(self):
        return self.input_data

    @property
    def targets(self):
            return self.output_data

    @property
    def input_shape(self):
        return self.input_data.shape

    @property
    def output_shape(self):
        return self.output_data.shape

# %%
test_grid = np.array(Image.open("images/piskel_example1.png.png"))[..., :3]
samples_dir = 'images'
images = [np.array(Image.open(os.path.join(samples_dir, file)))[..., :3] for file in os.listdir(samples_dir)
          if file.startswith(('piskel'))]

algo = algorithm.NEAT(
    pop_size=1000,  # Population size
    species_size=20,  # Size of species
    survival_threshold=0.1,  # Threshold for survival
    genome=genome.DefaultGenome(
        num_inputs=2,  # Normalized Pixel Coordinates and Number of input features (RGB values)
        num_outputs=4,  # Number of output categories (Red, Brown, Green, Blue)
        output_transform=common.ACT.sigmoid,  # Activation function for output layer
    ),
)

problem = TileProperties(test_grid, tile_size=1)

pipeline = Pipeline(
    algorithm=algo,  # The configured NEAT algorithm
    problem=problem,  # The problem instance
    generation_limit=200,  # Maximum generations to run
    fitness_target=-1e-6,  # Target fitness level
    seed=42,  # Random seed for reproducibility
)



state = pipeline.setup()
# Run the NEAT algorithm until termination
state, best = pipeline.auto_run(state)
# Display the results of the pipeline run
pipeline.show(state, best)
# %%

# %%
#network = algo.genome.network_dict(state, *best)
#algo.genome.visualize(network, save_path="images/tileproperties_network.png")


network = network_dict(algo.genome, state, *best)
visualize_labeled(algo.genome,network,["SGM"], rotate=90, save_path="network.svg", with_labels=True)
# %%

result = vmap(algo.forward,in_axes=(None,None,0))(state, algo.transform(state, best), problem.inputs)
# %%
print(result)
result = np.argmax(result, axis=1)
print(result)
result = result.reshape(test_grid.shape[:2])
print(result)

# %% get ratio of elements
unique, counts = np.unique(result, return_counts=True)
print(unique, counts)
print("----------")
target = np.argmax(problem.output_data, axis=1)
unique, counts = np.unique(target, return_counts=True)
print(unique, counts)
