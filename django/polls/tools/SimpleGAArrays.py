import jax.numpy as jnp
from jax import random, jit, vmap

def GenerateNewPopulation(parents, population_size, mutation_rate, mutation_power, rng=None, keep_parents=True, treshold_func=None):
    if rng is None:
        rng = random.PRNGKey(0)

    population = []
    if keep_parents:
        population.extend(parents)

    while len(population) < population_size:
        rng, key, key2, key3 = random.split(rng,4)
        if len(parents) == 1 or random.uniform(key3) < 0.5:
            parent = random.choice(key, jnp.array(parents))
            offspring = mutateFromParent(parent, mutation_rate, mutation_power, key2)
        else:
            parent1, parent2 = random.choice(key, jnp.array(parents), shape=(2,), replace=False)
            offspring = crossover(parent1, parent2, key2)
        #TODO: hash
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

@jit
def crossover(parent1, parent2, rng):
    keys = random.split(rng, 10)
    a,b = 5. * (random.uniform(keys[0]) - 0.5), 2. * (random.uniform(keys[1]) - 0.5)
    def getval(x,y):
        return y < a*x + b
    getvaljitted = jit(getval)
    height, width, channels = parent1.shape

    x_coords = jnp.linspace(-1, 1, width)
    y_coords = jnp.linspace(-1, 1, height)

    X, Y = jnp.meshgrid(x_coords, y_coords)

    mask = getvaljitted(X, Y)
    mask = jnp.expand_dims(mask, axis=-1).repeat(channels, axis=-1)

    return jnp.where(mask, parent1, parent2)





