from typing import Callable

import jax
from jax import vmap, numpy as jnp, jit
import numpy as np

from tensorneat.algorithm import BaseAlgorithm
from tensorneat.genome import BaseGenome
from .interactive_species import InteractiveSpeciesController
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
        print(winner, loser, elite_mask)

        state = self._create_next_generation(state, winner, loser, elite_mask)

        state = self.species_controller.speciate(state)

        return state

    def select_winners(self, selected_indices=None):
        # Example: Use CLI or GUI for selection
        if selected_indices is None:
            selected_indices = input(
                "Enter the indices of the individuals you want to select as winners, separated by commas: "
            )
            selected_indices = list(map(int, selected_indices.split(",")))
        else:
            selected_indices = list(map(int,selected_indices))

        if len(selected_indices) > 5:
            selected_indices = selected_indices[-5:]

        print(f"Selected indices: {selected_indices}")
        return jnp.array(selected_indices)

    def _create_next_generation(self, state, winner, loser, elite_mask):

        min_size = min(len(winner), len(loser))
        # find next node key for mutation
        all_nodes_keys = state.pop_nodes[:, :, 0]
        max_node_key = jnp.max(
            all_nodes_keys, where=~jnp.isnan(all_nodes_keys), initial=0
        )
        next_node_key = max_node_key + 1
        new_node_keys = jnp.arange(self.pop_size) + next_node_key

        new_conn_markers = jnp.full((self.pop_size, 3), 0)

        # prepare random keys
        k1, k2, randkey = jax.random.split(state.randkey, 3)
        crossover_randkeys = jax.random.split(k1, self.pop_size)
        mutate_randkeys = jax.random.split(k2, self.pop_size)

        wpn, wpc = state.pop_nodes[winner], state.pop_conns[winner]
        lpn, lpc = state.pop_nodes[loser], state.pop_conns[loser]

        # batch crossover
        n_nodes, n_conns = vmap(
            self.genome.execute_crossover, in_axes=(None, 0, 0, 0, 0, 0)
        )(
            state, crossover_randkeys, wpn, wpc, lpn, lpc
        )  # new_nodes, new_conns

        # batch mutation
        m_n_nodes, m_n_conns = vmap(
            self.genome.execute_mutation, in_axes=(None, 0, 0, 0, 0, 0)
        )(
            state, mutate_randkeys, n_nodes, n_conns, new_node_keys, new_conn_markers
        )  # mutated_new_nodes, mutated_new_conns

        print(elite_mask)
        # elitism don't mutate
        pop_nodes = jnp.where(elite_mask[:, None, None], state.pop_nodes, m_n_nodes)
        pop_conns = jnp.where(elite_mask[:, None, None], state.pop_conns, m_n_conns)

        return state.update(
            randkey=randkey,
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
        )

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
