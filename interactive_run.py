# %%
from interactive_NEAT import InteractiveNEAT
from interactive_problem import InteractiveGrid
from interactive_pipeline import InteractivePipeline

from tensorneat import genome, common

import jax
from jax import vmap, numpy as jnp, jit
import numpy as np
import matplotlib.pyplot as plt

# %%

test_genome = genome.DefaultGenome(
    num_inputs=2,
    num_outputs=4,
    node_gene=genome.DefaultNode(
        activation_options=[common.ACT.sigmoid, common.ACT.tanh, common.ACT.sin]
    ),
)

algo = InteractiveNEAT(
    pop_size=9,
    genome=test_genome,
)

problem = InteractiveGrid(grid_size=(16, 16))
grid = plt.imread("images/cppn_inputs/piskel_example1.png")

pipeline = InteractivePipeline(algorithm=algo, problem=problem, input_grid=grid)
# %%
state = pipeline.setup()

# %%
while True:
    state, population = pipeline.step(state)
    pipeline.visualize_population(population, save_path="outputs/population/")
    selected_indices = algo.select_winners()
    state = pipeline.evole(state, selected_indices)
