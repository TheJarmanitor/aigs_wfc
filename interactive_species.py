from tensorneat.algorithm import SpeciesController
import numpy as np
import jax
import jax.numpy as jnp


class InteractiveSpeciesController(SpeciesController):
    def __init__(self, pop_size) -> None:
        self.pop_size = pop_size

    def setup(self, state, first_nodes, first_conns):
        return state

    def update_species(self, state, selected_indices):

        winner = selected_indices

        loser = jnp.setdiff1d(jnp.arange(self.pop_size), selected_indices)

        elite_mask = jnp.isin(jnp.arange(self.pop_size), winner)

        return state, winner, loser, elite_mask

    def speciate(self, state, distance_function=None):
        return state
