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

for ratio in [1]:
    for inp_img in range(1,2): #input images example 1
        input_grid = np.array(Image.open(f"images/piskel_example{inp_img}.png.png"))[..., :3] 
        for x in range(1,2):
            start = time.time()

            #%% Settings & input for cppn neat

            #input for cppn

            #input_grid = np.array(Image.open("images/dragonwarr_island.png"))[..., :3] 

            # samples_dir = 'images'
            # images = [Image.open(os.path.join(samples_dir, file)) for file in os.listdir(samples_dir) 
            #           if file.startswith(('piskel_'))]

            #setting for cppn
            pop_size = 1000
            species_size = 20
            survival_threshold = 0.1 
            generation_limit = 100
            fitness_target = -1e-3
            seed = ratio*100 + x + 1
            tile_size_cppn = 1
            show_network = True

            #%% Settings for rule split wfc

            path_to_input_image = ".\images\dragonwarr_island.png"
            tile_size = 16
            output_name = "dragon"

            #%% Settings for wfc

            #bundle for local weights
            bundle=[
                [0,10],        #land
                [3],  #water
                [6, 8, 17],    #mountains
                [9]         #city
            ]

            #bundle = [
            #    [9, 11, 22]                    #mountain
            #    ,[5,7,15,23]                      #land
            #    ,[0,1,3,16,30]        #water
            #    ,[10, 13, 29]               #city
            #    ]

            default_weight = 1.0
            bundle_weight = ratio

            #rules folder
            path_folder = "dragon"

            #output size pixels
            size = 64

            #%% Settings for visualize wfc

            path_folder = "dragon"
            input_file = "output.txt"
            output_file = f"output_random{ratio}_{inp_img}_{x}.png"
            SHOW_NUKES = False
            
            def gaussian_(z): 
                return 1 / (jnp.std(z) * jnp.sqrt(2*jnp.pi)) * jnp.exp(-(z-jnp.mean(z))**2 / (2 * jnp.var(z)))

            activation_functions = [common.ACT.sigmoid, common.ACT.tanh, common.ACT.sin, gaussian_] #
            #%% Execute cppn neat

            result_cppn_neat = grid_problem.cppn_neat(input_grid = input_grid, pop_size = pop_size, species_size= species_size
                                            , survival_threshold=survival_threshold, activation_functions = activation_functions, generation_limit = generation_limit
                                            , fitness_target = fitness_target, seed = seed, tile_size = tile_size_cppn, show_network = show_network)

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

            local_weights = wfc.local_weight(bundle, default_weight=default_weight,prob_magnitude=bundle_weight, tile_count=len(rules))

            wfc.wfc([*range(len(rules))], rules, size, size,weights=None, path_to_output=f"outputs/{path_folder}/output.txt", layout_map = result_cppn_neat.reshape(shape), seed=seed)

            #%% Execute visualize wfc

            visualize_wfc.visualize_wfc(path_folder = path_folder, input_file = input_file, output_file = output_file, SHOW_NUKES = SHOW_NUKES)

            print(f"Running time: {time.time()-start} seconds")
