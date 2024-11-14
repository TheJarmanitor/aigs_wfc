# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 06:48:52 2024

@author: nuffz
"""

#%% Import Libs
import numpy as np
import time
import pickle
import os
import jax.numpy as jnp
import matplotlib.image as mpimg
import grid_problem
from sys import argv
from jax import random
from PIL import Image
from sys import argv
from typing import List
from tools import image_hashing, rule_split, visualize_labeled, visualize_wfc, wfc

start = time.time()

#%% Settings & input for cppn neat

#input for cppn
input_grid = np.array(Image.open("images/piskel_example1.png.png"))[..., :3] 

#setting for cppn
pop_size = 1000
species_size = 20
survival_threshold = 0.1 
generation_limit = 200
fitness_target = -1e-6
seed = 42
show_network = False

#%% Settings for rule split wfc

path_to_input_image = ".\images\dragonwarr_island.png"
tile_size = 16
output_name = "dragon"

#%% Settings for wfc

#rules for tiles
rules_islands_beaches = [
    #a -> a,b
    [
        {0,1},
        {0,1},
        {0,1},
        {0,1},
    ],
    #b -> a,c
    [
        {0,2},
        {0,2},
        {0,2},
        {0,2},
    ],
    #c -> b,c
    [
        {2,2},
        {2,1},
        {2,1},
        {2,1},
    ]
]

rules_mountain = [
    #a
    [
        {0,1}, #up -> a,b
        {0,1}, #right -> a,b
        {0}, #down -> a
        {0,1}, #left -> a,b
    ],
    #b -> a,c
    [
        {2}, #up -> c
        {0,2}, #right -> a,c
        {0}, #down -> a
        {0,2}, #left -> a,c
    ],
    #c -> b,c
    [
        {2}, #up -> c
        {2,1}, #right -> b,c
        {2,1}, #down -> b,c
        {2,1}, #left -> b,c
    ]
]

#bundle for local weights
bundle=[
    [8],
    [1, 2, 3],
    [6, 17],
    [9]
]

local_weights = wfc.local_weight(bundle)

#rules folder
path_folder = "dragon"

#output size pixels
size = 64

#%% Settings for visualize wfc

path_folder = "dragon"
input_file = "output.txt"
output_file = "output.png"
SHOW_NUKES = True
#%% Execute cppn neat

result_cppn_neat = grid_problem.cppn_neat(input_grid = input_grid, pop_size = pop_size, species_size= species_size
                                , survival_threshold=survival_threshold, generation_limit = generation_limit
                                , fitness_target = fitness_target, seed = seed, show_network = show_network)

shape = input_grid.shape[0], input_grid.shape[1]

#%% Execute rule split

img = Image.open(path_to_input_image)
img = img.convert("RGB")
img = np.array(img)
tile_size = int(tile_size)
rules = rule_split.RuleSet([list(map(lambda x: rule_split.Color(x[0], x[1], x[2]), row)) for row in img], tile_size)
print(f"Created {rules.id_counter} tiles")
# pinrt id map
for row in rules.image_id:
    print(row)
# print rules
for t in rules.tiles:
    print(f"Tile {t.id}")
    for i, r in enumerate(t.rules):
        print(f"  {i}: {r}")
name = output_name
rules.output_to_folder_rules(name)

#%% Execute wfc

path_to_file=f"outputs/{path_folder}/rules.pkl"
rules = pickle.load(open(path_to_file, "rb"))

wfc.wfc([*range(len(rules))], rules, size, size, weights = local_weights, path_to_output=f"outputs/{path_folder}/output.txt", layout_map = result_cppn_neat.reshape(shape))

#%% Execute visualize wfc

visualize_wfc.visualize_wfc(path_folder = path_folder, input_file = input_file, output_file = output_file, SHOW_NUKES = SHOW_NUKES)

print(f"Running time: {time.time()-start} seconds")