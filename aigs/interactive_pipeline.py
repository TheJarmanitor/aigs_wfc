import json
import os

import jax, jax.numpy as jnp
from jax import vmap, jit
import datetime, time
import numpy as np

from .interactive_NEAT import InteractiveNEAT
from .interactive_problem import InteractiveGrid
from tensorneat.common import State, StatefulBaseClass

from .tools.image_hashing import hash_grid, label_grids
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class InteractivePipeline(StatefulBaseClass):
    def __init__(
        self,
        algorithm: InteractiveNEAT,
        problem: InteractiveGrid,
        input_grid,
        tile_size=1,
        seed: int = 42,
        is_save: bool = False,
        save_dir=None,
    ):
        self.algorithm = algorithm
        self.problem = problem
        self.input_grid = input_grid
        self.tile_size = tile_size
        self.seed = seed
        self.pop_size = self.algorithm.pop_size

        hashed_grid, hash_dict = hash_grid(
            self.input_grid, self.tile_size, return_dict=True
        )
        _, self.unique_labels, self.label_to_tile = label_grids(hashed_grid, hash_dict)

        np.random.seed(self.seed)

        assert (
            algorithm.num_inputs == self.problem.input_shape[-1]
        ), f"algorithm input shape is {algorithm.num_inputs} but problem input shape is {self.problem.input_shape}"

    def setup(self, state=State()):
        print("initializing")
        state = state.register(randkey=jax.random.PRNGKey(self.seed))

        state = self.algorithm.setup(state)
        state = self.problem.setup(state)

        print("initializing finished")
        return state

    def generate(self, state):

        pop = self.algorithm.ask(state)

        pop_transformed = jax.vmap(self.algorithm.transform, in_axes=(None, 0))(
            state, pop
        )

        predict = vmap(self.problem.evaluate, in_axes=(None, None, 0))(
            state, self.algorithm.forward, pop_transformed
        )
        return pop, predict

    def evolve(self, state, selected_indices):
        state = self.algorithm.tell(state, selected_indices)

        return state

    def visualize_population(
        self,
        predict,
        pixel_size=1,
        save_path=None,
        file_name="output_pop",
        save_as_text=False,
    ):
        W, H = self.problem.grid_size
        population = jnp.argmax(predict, axis=2)
        if save_path is not None:
            if save_as_text:
                population_re = np.array(population.reshape(-1, W, H)).astype(np.uint8)
                for p in range(self.pop_size):
                    np.savetxt(
                        f"{save_path}/{file_name}_{p}_wfc.png.txt",
                        population_re[p],
                        fmt="%i",
                    )
            else:
                population_transform = population.reshape(-1)
                population_transform = np.array(population_transform)
                print(population_transform)
                new_grid = np.array(
                    [self.label_to_tile[x] for x in population_transform]
                )[:, :, :, :-1].reshape(-1, W, H, 3)
                img = np.zeros(
                    (self.pop_size, H * pixel_size, W * pixel_size, 3), dtype=np.uint8
                )
                for p in range(self.pop_size):
                    # for y in range(H):
                    #     for x in range(W):
                    #         img[
                    #             p,
                    #             y * pixel_size : (y + 1) * pixel_size,
                    #             x * pixel_size : (x + 1) * pixel_size,
                    #         ] = new_grid[p, y, x]
                    mpimg.imsave(f"{save_path}/{file_name}_{p}.png", new_grid[p])
