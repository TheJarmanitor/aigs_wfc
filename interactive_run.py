# %%
from interactive_NEAT import InteractiveNEAT
from interactive_problem import InteractiveGrid
from interactive_pipeline import InteractivePipeline

from tensorneat import genome, common

import jax
from jax import vmap, numpy as jnp, jit
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
# i guess its just expects the data in a weird layer of abstraction of vmaps and batches
def evaluate(state, act_func, population):
    predict = vmap(act_func, in_axes=(None, None, 0))(state, population, problem.inputs)
    return predict
predict = vmap(evaluate,in_axes=(None,None,0))(state, algo.forward, population)
print("Ran without error")

    
