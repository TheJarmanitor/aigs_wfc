from tools.image_hashing import hash_grid, label_grids
import numpy as np
import matplotlib.image as mpimg

# Visualize the output of a CPPN as a png image with colored pixels
def visualize_output_grid(cppn_output, input_grid, tile_size, path, pixel_size=1):
   W, H = cppn_output.shape
   cppn_output = cppn_output.reshape(-1)
   cppn_output = np.array(cppn_output)
   hashed_grid, hash_dict = hash_grid(input_grid, tile_size, return_dict=True)
   _, _, label_to_tile = label_grids(hashed_grid, hash_dict)
   print(label_to_tile)

   new_grid = np.array([label_to_tile[x] for x in cppn_output]).reshape(W, H, 3)

   img = np.zeros((H * pixel_size, W * pixel_size, 3), dtype=np.uint8)
   for y in range(H):
      for x in range(W):
         img[y * pixel_size:(y + 1) * pixel_size, x * pixel_size:(x + 1) * pixel_size] = new_grid[y, x]
   mpimg.imsave(path, img)
