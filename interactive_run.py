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
print(test_genome.input_idx)

# %%
state = pipeline.setup()
state, population = pipeline.step(state)
# i guess its just expects the data in a weird layer of abstraction of vmaps and batches

#pop_transformed = vmap(algo.transform, in_axes=(None, 0))(state, population)
#
#predict = vmap(problem.evaluate, in_axes=(None, None, 0))(
#    state, algo.forward, pop_transformed
#)
#
#labeled_predicts = jnp.argmax(predict, axis=2)
#print(labeled_predicts.shape)
## %%
#
#
#pipeline.visualize_population(labeled_predicts, grid, tile_size=1)
