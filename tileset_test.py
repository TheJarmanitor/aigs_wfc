# %%
from tools.image_hashing import hash_grid, label_grids
from grid_problem import cppn_neat
from tools.visualize_cppn import get_output_grid

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# %%

input_grid = np.array(Image.open("images/piskel_example1.png.png"))[..., :3]
tile_size=1
print(input_grid.shape)
plt.imshow(input_grid)
# %%
pop_size = 1000
species_size = 20
survival_threshold = 0.1
generation_limit = 200
fitness_target = -1e-6
seed = 42
show_network = False


result_cppn_neat = cppn_neat(input_grid = input_grid, pop_size = pop_size, species_size= species_size
                                , survival_threshold=survival_threshold, generation_limit = generation_limit
                                , fitness_target = fitness_target, seed = seed, show_network = show_network)
# %%


cppn_grid = get_output_grid(result_cppn_neat, input_grid, tile_size)
plt.imshow(cppn_grid)
