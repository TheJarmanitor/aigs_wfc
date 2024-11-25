from tools.image_hashing import hash_grid, label_grids
import numpy as np


def get_output_grid(cppn_output, input_grid, tile_size):
   W, H, _ = input_grid.shape
   cppn_output = np.array(cppn_output)
   hashed_grid, hash_dict = hash_grid(input_grid, tile_size, return_dict=True)
   _, _, label_to_tile = label_grids(hashed_grid, hash_dict)

   new_grid = np.array([label_to_tile[x] for x in cppn_output]).reshape(W, H, 3)

   return new_grid
