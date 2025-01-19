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

        # Winners are the user-selected indices
        winner = selected_indices

        # Losers are all individuals not selected
        loser = jnp.setdiff1d(jnp.arange(self.pop_size), selected_indices)

        # Pad the winners and losers to match the population size (if necessary)
        def pad_array(array, target_size):
            pad_length = target_size - len(array)
            if pad_length > 0:
                pad_values = jnp.full(
                    (pad_length,), array[0]
                )  # Repeat the first element for padding
                return jnp.concatenate([array, pad_values])
            return array

        # Determine the target size (maximum of winner and loser arrays)
        target_size = max(len(winner), len(loser))
        winner = pad_array(winner, target_size)
        loser = pad_array(loser, target_size)

        # Prepare random keys for crossover
        crossover_randkeys = jax.random.split(randkey, target_size)

        # Perform crossover by pairing winners with losers
        def aux_func(key, w_idx, l_idx):
            """
            Randomly select a parent from winners (w_idx) and losers (l_idx).
            """
            fa = jax.random.choice(key, jnp.array([w_idx, l_idx]))
            ma = jax.random.choice(key, jnp.array([w_idx, l_idx]))
            elite = w_idx  # Mark winners as elite (can be replaced with better logic if needed)
            return fa, ma, elite

        # Perform pairing for crossover
        fas, mas, elites = vmap(aux_func)(crossover_randkeys, winner, loser)

        # Assign the part1 and part2 (potential parents) to winners and losers
        is_part1_win = jnp.isin(fas, selected_indices)
        part1 = jnp.where(is_part1_win, fas, mas)
        part2 = jnp.where(~is_part1_win, fas, mas)

        # Determine the elite mask (individuals in selected_indices are elites)
        elite_mask = jnp.isin(jnp.arange(self.pop_size), selected_indices)

        return part1, part2, elite_mask

    def speciate(self, state, distance_function=None):
        return state
