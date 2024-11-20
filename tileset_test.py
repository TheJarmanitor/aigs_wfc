# %%
from tools.image_hashing import hash_grid, label_grids
from grid_problem import cppn_neat

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# %%

input_grid = np.array(Image.open("images/piskel_example1.png.png"))[..., :3]
tile_size=1
print(input_grid.shape)
plt.imshow(input_grid)
# %%
hashed_grid, hash_dict = hash_grid(input_grid, tile_size=tile_size, return_dict=True)

labeled_grid, unique_labels, label_to_tile = label_grids(hashed_grid, hash_dict=hash_dict)

fig, axes = plt.subplots(2,2)

for ax, (label, tile) in zip(axes.flatten(), label_to_tile.items()):
    ax.imshow(tile)
    ax.set_title(label)
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
# sort_idx = np.argsort(label_to_tile.keys())
# print(sort_idx)
# idx = np.searchsorted(label_to_tile.keys(),result_cppn_neat,sorter = sort_idx)
# new_grid = np.asarray(label_to_tile.values())[sort_idx][idx].reshape(input_grid.shape[0], input_grid.shape[1])

new_grid = np.array([label_to_tile[x] for x in np.array(result_cppn_neat)]).reshape(input_grid.shape[0], input_grid.shape[1], 3)

plt.imshow(new_grid)
