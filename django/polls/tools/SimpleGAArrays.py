import jax.numpy as jnp
from jax import random, jit, vmap

def GenerateNewPopulation(parents, population_size, mutation_rate, mutation_power, rng=None, keep_parents=True, treshold_func=None):
    if rng is None:
        rng = random.PRNGKey(0)

    population = []
    if keep_parents:
        population.extend(parents)

    while len(population) < population_size:
        rng, key, key2 = random.split(rng,3)
        parent = random.choice(key, jnp.array(parents))
        print(parent)
        offspring = mutateFromParent(parent, mutation_rate, mutation_power, key2)
        print(offspring)
        if treshold_func is not None and not treshold_func(offspring,population):
            continue
        population.append(offspring)
    return jnp.array(population)
    

@jit
def _mutateElement(x, mutation_rate, mutation_power, rng):
    key1, key2 = random.split(rng)
    mutation_mask = random.uniform(key1, shape=x.shape) < mutation_rate
    mutation_values = random.normal(key2, shape=x.shape) * mutation_power
    return jnp.where(mutation_mask, x + mutation_values, x)

@jit
def mutateFromParent(parent, mutation_rate, mutation_power, rng):
    offspring = parent.flatten()
    rngs = random.split(rng, offspring.shape[0])
    offspring = vmap(_mutateElement,(0,None,None,0),0)(offspring, mutation_rate, mutation_power, rngs)
    return offspring.reshape(parent.shape)




