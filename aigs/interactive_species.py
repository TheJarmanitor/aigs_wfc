import jax
import jax.numpy as jnp
from jax import vmap

from tensorneat.common import (
    State,
    StatefulBaseClass,
)


class InteractiveSpeciesController(StatefulBaseClass):
    def __init__(self, pop_size) -> None:
        self.pop_size = pop_size

    def setup(self, state, first_nodes, first_conns):
        return state

    def update_species(self, state, selected_indices):

        k1, k2 = jax.random.split(state.randkey)
        # crossover info
        winner, loser, elite_mask = self._create_crossover_pair(selected_indices, k1)

        return (
            state.update(randkey=k2),
            winner,
            loser,
            elite_mask,
        )

    def _create_crossover_pair(self, selected_indices, randkey):
        """
        Create crossover pairs using user-selected indices, following the Picbreeder pipeline.
        - The selected_indices directly determine winners.
        - Losers are the remaining individuals not in the selected_indices.
        """

        crossover_randkeys = jax.random.split(randkey, self.pop_size)
        p_idx = jnp.arange(self.pop_size)

        # Perform crossover by pairing winners with losers
        def aux_func(key, idx):
            """
            Randomly select a parent from winners (w_idx) and losers (l_idx).
            """

            fa, ma = jax.random.choice(key, idx, shape=(2,), replace=True)
            elite = idx  # Mark winners as elite (can be replaced with better logic if needed)
            return fa, ma, elite

        # Perform pairing for crossover
        fas, mas, elites = vmap(aux_func, in_axes=(0, None))(
            crossover_randkeys, selected_indices
        )

        # Assign the part1 and part2 (potential parents) to winners and losers
        is_part1_win = jnp.isin(fas, selected_indices)
        part1 = jnp.where(is_part1_win, fas, mas)
        part2 = jnp.where(~is_part1_win, fas, mas)

        # Determine the elite mask (individuals in selected_indices are elites)
        elite_mask = jnp.isin(jnp.arange(self.pop_size), elites)

        return part1, part2, elite_mask

    def speciate(self, state, distance_function=None):
        return state
