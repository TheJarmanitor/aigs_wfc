# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:34:37 2024

@authors: Frank, Oleg & Jarl
"""
#%% Import libs
from jax import random, nn, lax, jit, value_and_grad, vmap
from jaxtyping import Array
import jax.numpy as jnp


#%% Initialization of CPPN
def init_cppn_params(input_dim: int, hidden_dims: list, 
                     output_dim: int, rng: int, scale: float = 0.01):
    """
    Parameters
    ----------
    input_dim : int
        Input dimension used for the input layer.
    hidden_dims : list
        List of integeres assigning the number of hidden nodes in each of the hidden layers.
    output_dim : int
        Output dimension used for the output layer.
    rng : int
        Random number generator.
    scale : float, optional
        Scaling of weights. The default is 0.01.

    Returns
    -------
    params : list
        Returns a list of initial weights and biases for the CPPN.
    """
    
    #Initialize MLP
    params = []
    
    #Split random key generator
    rng, layer_rng = random.split(rng)
    
    #Determine first prev_dim as input dim
    prev_dim = input_dim
    
    #Create random weights and biases for the number of hidden layers
    for hidden_dim in hidden_dims:
        w = random.normal(layer_rng, (prev_dim, hidden_dim)) * scale
        b = jnp.zeros(hidden_dim)
        params.append((w, b))
        prev_dim = hidden_dim
        
    #Output layer weights and biases
    w = random.normal(layer_rng, (prev_dim, output_dim))
    b = jnp.zeros(output_dim)
    params.append((w, b))
    return params

#%% Crossover & Mutation Functions
def crossover(rng: int, parents: Array, 
              minval: float = 0.0, maxval: float = 1.0) -> Array:
    """
    Parameters
    ----------
    rng : int
        Random Number Generator.
    parents : Array
        Array of parents input.
    minval : float
        Minimum value of uniform distribution. The default is 0.0.
    maxval : float, optional
        Maximum value of uniform distribution. The default is 1.0.

    Returns
    -------
    Array
        Crossover of parents.
    """
    
    #store shape of parents
    size, dim = parents.shape
    
    #random number generator split
    rng, subkey = random.split(rng)
    
    # Create random indices to choose pairs of parents for crossover
    idx1 = random.randint(subkey, (size,), 0, size)
    idx2 = random.randint(subkey, (size,), 0, size)
    
    # Blend the parents with some random weight
    alpha = random.uniform(subkey, (size, dim), minval=minval, maxval=maxval)
    offspring = alpha * parents[idx1] + (1 - alpha) * parents[idx2]
    return offspring

def mutation(rng: int, pop: Array, std: float = 0.1) -> Array:
    """
    Parameters
    ----------
    rng : int
        A random number generator.
    pop : Array
        The population that will be mutated (genome).
    std : float, optional
        The standard devation to alter the population. The default is 0.1.

    Returns
    -------
    Array
        Mutated population by Gaussian Noise.
    """
    #random number generator split
    rng, subkey = random.split(rng)
    
    # Add Gaussian noise to mutate population
    mutation = std * random.normal(subkey, pop.shape)
    mutated_population = pop + mutation
    return mutated_population

#%% Dropout, activation and apply cppn functions

def dropout_fn(rng: int, obs: Array, dropout: float) -> Array:
    """
    Parameters
    ----------
    rng : int
        Random Number Generator
    obs : Array
        Input array of observations.
    dropout : float
        Dropout probability 
    Returns
    -------
    Array
        Returns an array where some weights and biases that may be removed by some probability.
    """
    #assert if dropout rate
    assert 0 <= dropout <= 1, "dropout must be between 0 and 1"
    
    #generate random key
    key = random.split(rng)[1]
    
    # The probability of keeping a unit
    keep_prob = 1 - dropout  
    
    # Generate the dropout mask
    mask = random.bernoulli(key, keep_prob, shape=obs.shape)  
    
    return jnp.where(mask, obs / keep_prob, 0)  # Scale the kept units and zero out dropped ones

def activation_fn(key, obs):
    """
        Function to determine the different activation functions, perhaps chosen by some probability?
    """
    return NotImplementedError

def apply_cppn_fn(params: list, obs: Array, rng: int, 
                  train: bool = True, dropout_rate: float = 0.5) -> Array:
    """
    Parameters
    ----------
    params : list
        A list of parameters, weights and biases.
    obs : Array
        Input array, observations.
    rng : int
        Random key generator
    train : Bolean, optional
        For training sessions crossovers, mutations, dropouts are applied. The default is True.
    dropout_rate : float, optional
        The dropout probability of weights and biases. The default is 0.5.
    Returns
    -------
    Array
        Returns the output layer of the CPPN
    """
    
    def apply_layer(rng, params, obs):
        x = obs
        for i, (w, b) in enumerate(params):
            x = jnp.dot(x, w) + b
            if i < len(params)-1:
                x = activation_fn(x) #determines activation function (not yet implemented)
                if train: #do crossovers, mutations and dropouts while training
                    rng = random.split(rng)[0]  # Split RNG 
                    x = crossover(rng, x) #crossover of parents
                    x = mutation(rng, x)  #mutation of population
                    x = dropout_fn(rng, x, dropout=dropout_rate) #dropout of connections (weights and biases)
            else:
                x = nn.softmax(x) #softmax output layer for probabilities
        return x
                
    rngs = random.split(rng, len(obs))  #create random rngs equal to the length of observations
    output = vmap(apply_layer, in_axes=[0,None,0])(rngs, params, obs) #vmap to apply for all samples
    return output