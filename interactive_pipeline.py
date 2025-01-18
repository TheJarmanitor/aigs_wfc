import json
import os

import jax, jax.numpy as jnp
import datetime, time
import numpy as np

from interactive_NEAT import InteractiveNEAT
from interactive_problem import InteractiveGrid
from tensorneat.common import State, StatefulBaseClass


class InteractivePipeline(StatefulBaseClass):
    def __init__(
        self,
        algorithm: InteractiveNEAT,
        problem: InteractiveGrid,
        seed: int = 42,
        is_save: bool = False,
        save_dir=None,
    ):
        self.algorithm = algorithm
        self.problem = problem
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

        return state, pop_transformed
