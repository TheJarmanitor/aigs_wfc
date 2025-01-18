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

    def select_winners(self):
        # Example: Use CLI or GUI for selection

        selected_indices = input(
            "Enter the indices of the individuals you want to select as winners, separated by commas: "
        )
        selected_indices = list(map(int, selected_indices.split(",")))

        if len(selected_indices) > 5:
            selected_indices = selected_indices[-5:]

        return jnp.array(selected_indices)

    def _create_next_generation(self, state, winner, loser, elite_mask):

        # find next node key for mutation
        all_nodes_keys = state.pop_nodes[:, :, 0]
        max_node_key = jnp.max(
            all_nodes_keys, where=~jnp.isnan(all_nodes_keys), initial=0
        )
        next_node_key = max_node_key + 1
        new_node_keys = jnp.arange(self.pop_size) + next_node_key

        # find next conn historical markers for mutation if needed
        if "historical_marker" in self.genome.conn_gene.fixed_attrs:
            all_conns_markers = vmap(
                self.genome.conn_gene.get_historical_marker, in_axes=(None, 0)
            )(state, state.pop_conns)

            max_conn_markers = jnp.max(
                all_conns_markers, where=~jnp.isnan(all_conns_markers), initial=0
            )
            next_conn_markers = max_conn_markers + 1
            new_conn_markers = (
                jnp.arange(self.pop_size * 3).reshape(self.pop_size, 3)
                + next_conn_markers
            )
        else:
            # no need to generate new conn historical markers
            # use 0
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

        # elitism don't mutate
        pop_nodes = jnp.where(elite_mask[:, None, None], n_nodes, m_n_nodes)
        pop_conns = jnp.where(elite_mask[:, None, None], n_conns, m_n_conns)

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
