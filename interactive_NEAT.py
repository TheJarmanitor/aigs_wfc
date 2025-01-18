from typing import Callable

import jax
from jax import vmap, numpy as jnp
import numpy as np

from tensorneat.algorithm import BaseAlgorithm
from tensorneat.genome import BaseGenome
from interactive_species import InteractiveSpeciesController
from tensorneat.common import State


class InteractiveNEAT(BaseAlgorithm):
    def __init__(
        self,
        genome: BaseGenome,
        pop_size: int,
    ) -> None:
        self.genome = genome
        self.pop_size = pop_size
        self.species_controller = InteractiveSpeciesController(pop_size)

    def setup(self, state=State()):
        # setup state
        state = self.genome.setup(state)

        k1, randkey = jax.random.split(state.randkey, 2)

        # initialize the population
        initialize_keys = jax.random.split(k1, self.pop_size)
        pop_nodes, pop_conns = vmap(self.genome.initialize, in_axes=(None, 0))(
            state, initialize_keys
        )

        state = state.register(
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
            generation=jnp.float32(0),
        )

        # initialize species state
        state = self.species_controller.setup(state, pop_nodes[0], pop_conns[0])

        return state.update(randkey=randkey)

    def ask(self, state):
        return state.pop_nodes, state.pop_conns

    def tell(self, state, selected_indices):
        state = state.update(generation=state.generation + 1)

        state, winner, loser, elite_mask = self.species_controller.update_species(
            state, selected_indices
        )

        state = self._create_next_generation(state, winner, loser, elite_mask)

        state = self.species_controller.speciate(state)

        return state

    def transform(self, state, individual):
        nodes, conns = individual
        return self.genome.transform(state, nodes, conns)

    def forward(self, state, transformed, inputs):
        return self.genome.forward(state, transformed, inputs)

    @property
    def num_inputs(self):
        return self.genome.num_inputs

    @property
    def num_outputs(self):
        return self.genome.num_outputs
