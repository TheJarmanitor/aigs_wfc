# %% Libraries
from matplotlib.image import _ImageBase
from tensorneat.problem.func_fit import FuncFit
from tensorneat.pipeline import Pipeline
from tensorneat import algorithm, genome, common
from PIL import Image
from tools.image_hashing import hash_grid, label_grids
from tools.visualize_cppn import visualize_output_grid

import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from jax import vmap, jit
from jax.lax import fori_loop, while_loop

from tools.visualize_labeled import visualize_labeled, network_dict


# %%
class TileProperties(FuncFit):
    def __init__(self, grid, tile_size, grid_size=None, hash=True, error_method="mse") -> None:
        self.grid = grid
        self.tile_size = tile_size
        # self.hash = hash
        self.error_method = error_method
        self.grid_size = grid_size
        self._prepare_data()

    def _prepare_data(self):
        grid_info_list = []
        one_hot_list = []
        hashed_grid, hash_tile_dict = hash_grid(self.grid, self.tile_size, return_dict=True)

        labeled_grids, unique_labels, label_tile_dict = label_grids(hashed_grid, hash_dict=hash_tile_dict)

        grid = labeled_grids[0]
        self.label_tile_dict = label_tile_dict

        width, height = grid.shape if self.grid_size is None else self.grid_size
        #_, unique_counts = np.unique(grid, return_counts=True)
        # unique_proportions = unique_counts / (height * width)
        # proportion_list.append(unique_proportions)
        grid_information = []
        one_hot_output = []

        #inpt data
        for y in range(width):
            for x in range(height):
                normalized_x = (2 * x / (width - 1)) - 1  # Normalize x coordinate
                normalized_y = (2 * y / (height - 1)) - 1  # Normalize y coordinate
                coordinates = np.array([normalized_x, normalized_y])
                grid_information.append(coordinates)
        
        #target data
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                one_hot = np.eye(len(unique_labels))[int(grid[x, y])]
                one_hot_output.append(one_hot)

        grid_info_list.append(np.array(grid_information))
        one_hot_list.append(np.array(one_hot_output))

        self.input_data = jnp.concatenate(grid_info_list, axis=0)
        self.output_data = jnp.concatenate(one_hot_list, axis=0)
        self.unique_labels = unique_labels

        self.width = width
        self.height = height

        self.target = self.extract_metrics(self.output_data, len(unique_labels), custom_shape=grid.shape)

    def extract_metrics(self, image, colors, custom_shape=None):
        if image.shape[1] == colors:
            image = jnp.argmax(image, axis=1)
            
        height, width = self.width, self.height
        if custom_shape is not None:
            height, width = custom_shape
        
        image = image.reshape(height, width)

        if colors is list:
            colors = len(colors)

        # tiles ratio
        tiles_ratio = jnp.zeros(colors)
        def tiles_ratio_loop(i, tiles_ratio):
            tiles_ratio = tiles_ratio.at[i].set(jnp.sum(image == i))
            return tiles_ratio
        tiles_ratio = fori_loop(0, colors, tiles_ratio_loop, tiles_ratio)

#        for i in range(colors):
#            tiles_ratio[i] = jnp.sum(image == i)
        tiles_ratio = 1.5 * jnp.array(tiles_ratio) / (height * width)

        # edge ratio
        edge_ratio = jnp.zeros(colors**2)

        @jit
        def edge_ratio_loop(i, edge_ratio):
            a,b = i//colors, i%colors
            index = jnp.minimum(a*colors+b,b*colors+a)
            edge_ratio = edge_ratio.at[index].set(jnp.sum((image[:-1, :] == a) & (image[1:, :] == b)) \
                + jnp.sum((image[:, :-1] == a) & (image[:, 1:] == b)))
            return edge_ratio
        
        edge_ratio = jnp.array(fori_loop(0, colors**2, edge_ratio_loop, edge_ratio))
#        for i in range(colors):
#            for j in range(colors):
#                index = min(i*colors+j,j*colors+i)
#                edge_ratio[index] += jnp.sum((image[:-1, :] == i) & (image[1:, :] == j))
#                edge_ratio[index] += jnp.sum((image[:, :-1] == i) & (image[:, 1:] == j))
#        edge_ratio = jnp.array(edge_ratio)
        edge_ratio = jnp.sqrt(edge_ratio / ((height-1)**2 + (width-1)**2))

        symm_h = 0
        symm_v = 0

        @jit
        def symm_loop_vert(a, symm_v):
            symm_x = a // (width-1)
            i = a % (width-1)
            symm_v += jnp.sum(image[:, symm_x-i+1] == image[:, symm_x+i]) * ((i < symm_x) * (symm_x+i < width))
            return symm_v
        
        @jit
        def symm_loop_hor(a, symm_h):
            symm_y = a // (height-1)
            i = a % (height-1)
            symm_h += jnp.sum(image[symm_y-i+1, :] == image[symm_y+i, :]) * ((i < symm_y) * (symm_y+i < height))
            return symm_h
        
        symm_v = fori_loop(0, (width-1)**2, symm_loop_vert, symm_v)
        symm_h = fori_loop(0, (height-1)**2, symm_loop_hor, symm_h)


#        for symm_x in range(width-1):
#            for i in range(1,min(symm_x+1, width-(symm_x+1))+1):
#                symm_v += jnp.sum(image[:, symm_x-i+1] == image[:, symm_x+i])
#        for symm_y in range(height-1):
#            for i in range(1,min(symm_y+1, height-(symm_y+1))+1):
#                symm_h += jnp.sum(image[symm_y-i+1, :] == image[symm_y+i, :])
#
        # https://oeis.org/A002620 -> number of pairs per elements in vector of length n
        symm_h /= ((np.ceil(height/2)*np.floor(height/2))*width)
        symm_v /= ((np.ceil(width/2)*np.floor(width/2))*height)

        #diagonal symmetry
        
        symm_d = 0
        symm_dd = 0

        for i in range(0, min(width, height)):
            for j in range(0, i):
                symm_d += image[i,j] == image[j,i]

        for i in range(0, min(width, height)):
            for j in range(0, height-i-1):
                symm_dd += image[i,j] == image[width-j-1,height-i-1]

        s = min(width, height)
        symm_d /= s*(s-1)/2
        symm_dd /= s*(s-1)/2
            
        symmetry_ratio = jnp.array([symm_h, symm_v, symm_d, symm_dd])

        return jnp.concatenate([tiles_ratio, edge_ratio, symmetry_ratio])
        


    def evaluate(self, state, randkey, act_func, params):

        predict = vmap(act_func, in_axes=(None, None, 0))(
                    state, params, self.inputs
                )
        predict = jnp.argmax(predict, axis=1)
        predict = jnp.eye(len(self.unique_labels))[predict]
        predict_metrics = self.extract_metrics(predict, len(self.unique_labels))
#
#        if self.error_method == "mse":
#                loss = jnp.mean((predict_metrics - self.target) ** 2)
#
#        elif self.error_method == "rmse":
#            loss = jnp.sqrt(jnp.mean((predict_metrics - self.target) ** 2))
#
#        elif self.error_method == "mae":
#            loss = jnp.mean(jnp.abs(predict_metrics - self.target))
#
#        elif self.error_method == "mape":
#            loss = jnp.mean(jnp.abs((predict_metrics - self.target) / self.target))
#
#        else:
#            raise NotImplementedError
#
        return -jnp.sqrt(jnp.mean((predict_metrics - self.target) ** 2))

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


def cppn_neat(input_grid: np.array, pop_size: int = 1000, species_size: int = 20, grid_size: tuple = None
              , survival_threshold: float = 0.1, activation_functions: list = [common.ACT.sigmoid], generation_limit: int = 200
              , fitness_target: float = -1e-6, seed: int= 42, tile_size: int = 16, show_network: bool = False, activation_labels: list = ["SGM"]
              , visualize_output_path: str = None):

    problem = TileProperties(input_grid, tile_size=tile_size, grid_size=grid_size, error_method="rmse")


    algo = algorithm.NEAT(
        pop_size=pop_size,  # Population size
        species_size=species_size,  # Size of species
        survival_threshold=survival_threshold,  # Threshold for survival
        genome=genome.DefaultGenome(
            num_inputs=2,  # Normalized Pixel Coordinates and Number of input features (RGB values)
            num_outputs=4,  # Number of output categories (Red, Brown, Green, Blue)
            #output_transform=common.ACT.sigmoid,  # Activation function for output layer
            node_gene=genome.DefaultNode(
                activation_options=activation_functions,  # Activation functions for hidden layers
            )
        ),
    )

    
    pipeline = Pipeline(
        algorithm=algo,  # The configured NEAT algorithm
        problem=problem,  # The problem instance
        generation_limit=generation_limit,  # Maximum generations to run
        fitness_target=fitness_target,  # Target fitness level
        seed=seed,  # Random seed for reproducibility
    )



    state = pipeline.setup()
    # Run the NEAT algorithm until termination
    state, best = pipeline.auto_run(state)
    # Display the results of the pipeline run
    # pipeline.show(state, best)

    if show_network:
        network = network_dict(pipeline.algorithm.genome, state, *best)
        path_to_netowrk = "network.svg" if visualize_output_path is None else visualize_output_path.replace(".png", "_network.svg")
        visualize_labeled(pipeline.algorithm.genome,network,activation_labels, rotate=90, save_path=path_to_netowrk, with_labels=True)

    algo_forward = vmap(pipeline.algorithm.forward,in_axes=(None,None,0))(state, pipeline.algorithm.transform(state, best), problem.inputs)
    result = np.argmax(algo_forward, axis=1)
    print(f"target metrics: {problem.target}")
    print(f"result metrics: {problem.extract_metrics(algo_forward, 4)}")
    # print(result)
    # result = result.reshape(test_grid.shape[:2])
    # print(result)

    # unique, counts = np.unique(result, return_counts=True)
    # print(unique, counts)
    # print("----------")
    # target = np.argmax(problem.output_data, axis=1)
    # unique, counts = np.unique(target, return_counts=True)
    # print(unique, counts)

    if visualize_output_path is not None:
        visualize_output_grid(result.reshape(grid_size), input_grid, tile_size, visualize_output_path, 1)
        new_path = visualize_output_path.replace(".png", "_upscaled.png")
        visualize_output_grid(result.reshape(grid_size), input_grid, tile_size, new_path, 10)

    return result, problem.label_tile_dict

if __name__ == "__main__":
    input_image = np.array(Image.open("images\piskel_example1.png.png"))[..., :3]

    result = cppn_neat(input_image, pop_size=500, species_size=20, survival_threshold=0.1, generation_limit=1000, fitness_target=-5e-4, seed=123, show_network=False)
    result = result.reshape(input_image.shape[:2])
    print(result)

    tp = TileProperties(input_image, tile_size=1)
    print(tp.extract_metrics(tp.output_data, len(tp.unique_labels)))
    print("--------")
    print(tp.extract_metrics(result, len(tp.unique_labels)))
