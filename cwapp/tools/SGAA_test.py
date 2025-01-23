from .SimpleGAArrays import GenerateNewPopulation
import jax.numpy as jnp
import numpy as np
import matplotlib.image as mpimg
import time
from jax import random
from math import sin, cos

def differentTest(offspring,others):
    am_off = jnp.argmax(offspring,2)
    for other in others:
        if jnp.all(am_off == jnp.argmax(other,2)):
            return False
    return True

def UNUSED_init_population():
    parents = jnp.zeros((9,32,32,3)).tolist()
    for i in range(32):
        for j in range(32):
            x = i/32.
            y = j/32.
            parents[0][i][j] = [x,y,1.2-(x+y)]
            parents[1][i][j] = [x,y,(x+y)-0.5]
            parents[2][i][j] = [y,x,y*x]
            parents[3][i][j] = [y*y,x*x,0.5]
            parents[4][i][j] = [x+x*y,y+y*x,0.3]
            parents[5][i][j] = [sin(x*10),cos(y*10),x+y-0.3]
            parents[6][i][j] = [x*x-y*y,x*y,0.5]
            parents[7][i][j] = [x+y-x*y,y-x,x-y]
            parents[8][i][j] = [x-y,y-x,0.]
    return [jnp.array(parent) for parent in parents]
    
    

    


def test_GenerateNewPopulation():
    rng = random.PRNGKey(0)
    parent = jnp.zeros((32,32,3)).tolist()
    for i in range(32):
        for j in range(32):
            parent[i][j] = [i/32.,j/32,1.2-(i+j)/32.]
    parents = [jnp.array(parent)]
    population_size = 9
    mutation_rate = 0.1
    mutation_power = 0.5
    keep_parents = True
    treshold_func = differentTest

    population = GenerateNewPopulation(parents, population_size, mutation_rate, mutation_power, rng, keep_parents, treshold_func)
    assert population.shape == (population_size,32,32,3)
    paths = [ imgFromArray(population[i],f"img_{i}.png") for i in range(population_size) ]
    return paths



def imgFromArray(arr, path):
    arr = jnp.argmax(arr,2)
    RGB = [
    [ 40, 229,  34], #land
    [ 24,  28, 214], #water
    [ 85,  10,  10], #mountains
    [ 211, 26,  26]  #city
]
    img = np.zeros((arr.shape[0],arr.shape[1],3))
    for i in range(3):
        img[arr == i] = RGB[i]
    mpimg.imsave(path,img)
    return path

if __name__ == "__main__":
    stopwatch = time.time()
    paths = test_GenerateNewPopulation()
    print("Time elapsed: ", time.time() - stopwatch)