# %%
from interactive_NEAT import InteractiveNEAT
from interactive_problem import InteractiveGrid
from interactive_pipeline import InteractivePipeline

from tensorneat import genome, common

import jax
from jax import vmap, numpy as jnp
import numpy as np

# %%

test_genome = genome.DefaultGenome(
    num_inputs=2,
    num_outputs=4,
    node_gene=genome.DefaultNode(
        activation_options=[common.ACT.sigmoid, common.ACT.tanh, common.ACT.sin]
    ),
)

algo = InteractiveNEAT(
    pop_size=1,
    genome=test_genome,
)

problem = InteractiveGrid(grid_size=(16, 16))

pipeline = InteractivePipeline(
    algorithm=algo,
    problem=problem,
)
# %%
print(test_genome.input_idx)

# %%

state = pipeline.setup()
state, population = pipeline.step(state)
print(population[1])
# predict = algo.forward(state, population, problem.inputs[0])
# predict = vmap(algo.forward, in_axes=(None, None, 0))(state, population, problem.inputs)
