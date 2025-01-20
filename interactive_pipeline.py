import json
import os

import jax, jax.numpy as jnp
from jax import vmap
import datetime, time
import numpy as np

from interactive_NEAT import InteractiveNEAT
from interactive_problem import InteractiveGrid
from tensorneat.common import State, StatefulBaseClass

from tools.image_hashing import hash_grid, label_grids
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class InteractivePipeline(StatefulBaseClass):
    def __init__(
        self,
        algorithm: InteractiveNEAT,
        problem: InteractiveGrid,
        input_grid,
        seed: int = 42,
        is_save: bool = False,
        save_dir=None,
    ):
        self.algorithm = algorithm
        self.problem = problem
        self.input_grid = input_grid
        self.seed = seed
        self.pop_size = self.algorithm.pop_size

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

    def step(self, state):
        randkey_, randkey = jax.random.split(state.randkey)
        keys = jax.random.split(randkey_, self.pop_size)

        pop = self.algorithm.ask(state)

        pop_transformed = jax.vmap(self.algorithm.transform, in_axes=(None, 0))(
            state, pop
        )

        predict = vmap(self.problem.evaluate, in_axes=(None, None, 0))(
            state, self.algorithm.forward, pop_transformed
        )


        return state.update(randkey=randkey), predict

    def evole(self, state, selected_indices):
        state = self.algorithm.tell(state, selected_indices)

        return state


    def visualize_population(
        self, predict, tile_size=1, pixel_size=1, save_path=None
    ):
        W, H = self.problem.grid_size
        population = jnp.argmax(predict, axis=2)
        population = population.reshape(-1)
        population = np.array(population)
        hashed_grid, hash_dict = hash_grid(self.input_grid, tile_size, return_dict=True)
        _, _, label_to_tile = label_grids(hashed_grid, hash_dict)
        new_grid = np.array([label_to_tile[x] for x in population])[
            :, :, :, :-1
        ].reshape(-1, W, H, 3)
        img = np.zeros(
            (self.pop_size, H * pixel_size, W * pixel_size, 3), dtype=np.uint8
        )
        fig, axes = plt.subplots(1, self.pop_size)

        for p in range(self.pop_size):
            for y in range(H):
                for x in range(W):
                    img[
                        p,
                        y * pixel_size : (y + 1) * pixel_size,
                        x * pixel_size : (x + 1) * pixel_size,
                    ] = new_grid[p, y, x]
            axes[p].imshow(new_grid[p])
            if save_path is not None:
                mpimg.imsave(f"{save_path}/output_pop_{p}.png", new_grid[p])
