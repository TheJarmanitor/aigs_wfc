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
from tensorneat import algorithm, genome, common
from tools import image_hashing, rule_split, visualize_labeled, visualize_wfc, wfc



#%% Settings & input for cppn neat

#setting for cppn
pop_size = 600
species_size = 20
survival_threshold = 0.05
generation_limit = 200
fitness_target = -1e-4
init_seed = 42
tile_size_cppn = 1
show_network = True
cppn_output_grid_size = (16,16)
input_cppn_image = ".\\images\\cppn_inputs\\piskel_example1.png"

#%% Settings for tileset/ruleset
#size of tile in pixels in the tileset input
tile_size = 16
output_name = "dragon_warrior"
path_to_input_image = ".\\images\\tileset_inputs\\dragon_warrior\\dragonwarr_island.png"
#output_name = "pokemon"
#path_to_input_image = ".\\images\\tileset_inputs\\pokemon\\pokemon_route_103.png"


#%% Settings for wfc
from tools.prepared_bundles import bundle_dragon_warr, bundle_pokemon_103, bundle_pokemon_110, bundle_pokemon_114, bundle_pokemon_123
bundle = bundle_dragon_warr

# bundle weights
#   note: setting the default_weight same as one of the ratios will result in wfc not using the cppn output
default_weight = 1.0
ratios = [1]

# wfc output size cell x cell
size = 40

#%% Settings for visualize wfc
input_file = "output.txt"
SHOW_NUKES = False

# custom activation functions
def gaussian_(z): 
    return 1 / (jnp.std(z) * jnp.sqrt(2*jnp.pi)) * jnp.exp(-(z-jnp.mean(z))**2 / (2 * jnp.var(z)))

# a dict of activation functions (key: value) -> (label: function)
activation_functions_dict = {
    "SGM": common.ACT.sigmoid, 
    "TNH": common.ACT.tanh, 
    "SIN": common.ACT.sin, 
    "GSS": gaussian_,
    "RLU": common.ACT.relu
}





# preparations
bundle_colors_in_cppn = [
    [ 40, 229,  34], #land
    [ 24,  28, 214], #water
    [ 85,  10,  10], #mountains
    [ 211, 26,  26]  #city
]
os.makedirs(f"outputs/{output_name}", exist_ok=True)
activation_labels = list(activation_functions_dict.keys())
activation_functions = [activation_functions_dict[activation_labels[i]] for i in range(len(activation_labels))]

# main loop
for x in range(10):
    for ratio in ratios:
        input_grid = np.array(Image.open(input_cppn_image))[..., :3] 
        start = time.time()
        #%% Execute cppn neat

        seed = ratio * 1000 + init_seed + x
        output_file = f"output_{output_name}_w{ratio}_{x}.png"
        bundle_weight = ratio

        result_cppn_neat, label_tile_dict = grid_problem.cppn_neat(input_grid = input_grid, pop_size = pop_size, species_size= species_size
                                        , survival_threshold=survival_threshold, activation_functions = activation_functions, generation_limit = generation_limit if ratio != default_weight else 1
                                        , fitness_target = fitness_target, seed = seed, tile_size = tile_size_cppn
                                        , show_network = show_network, activation_labels= activation_labels, grid_size=cppn_output_grid_size
                                        , visualize_output_path = f"outputs/{output_name}/cppn_{output_file}"
                                        )

        print(f"Result: \n{result_cppn_neat.reshape(cppn_output_grid_size)}")
        print(f"Label: \n{label_tile_dict}")

        # change bundle indexing so it fits label_tile_dict
        correct_bundle_indices = []
        for bundle_type in range(len(bundle)):
            color = bundle_colors_in_cppn[bundle_type]
            for key, value in label_tile_dict.items():
                if (color == value).all():
                    correct_bundle_indices.append(key)
                    break
        if len(correct_bundle_indices) != len(bundle):
            print("Error: bundle not found in label_tile_dict")
            exit(1)
        print(f"Correct bundle indices: {correct_bundle_indices}")
        bundle = [bundle[correct_bundle_indices.index(i)] for i in range(len(bundle))]
        print(f"Correct bundle: {bundle}")

        #%% Execute rule split

        img = Image.open(path_to_input_image)
        img = img.convert("RGB")
        img = np.array(img)
        tile_size = int(tile_size)
        rules = rule_split.RuleSet([list(map(lambda x: rule_split.Color(x[0], x[1], x[2]), row)) for row in img], tile_size)
        print(f"Created {rules.id_counter} tiles")
        # print id map
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

        path_to_file=f"outputs/{output_name}/rules.pkl"
        rules = pickle.load(open(path_to_file, "rb"))

        local_weights = wfc.local_weight(bundle, default_weight=default_weight,prob_magnitude=bundle_weight, tile_count=len(rules))

        wfc.wfc([*range(len(rules))], rules, size, size,weights=local_weights, path_to_output=f"outputs/{output_name}/output.txt", layout_map = result_cppn_neat.reshape(cppn_output_grid_size), seed=seed)

        #%% Execute visualize wfc

        visualize_wfc.visualize_wfc(path_folder = output_name, input_file = input_file, output_file = output_file, SHOW_NUKES = SHOW_NUKES)

        print(f"Running time: {time.time()-start} seconds")
