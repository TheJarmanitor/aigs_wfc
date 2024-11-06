import numpy as np
import sys
import matplotlib.image as mpimg

path_folder = "test" if len(sys.argv) < 2 else sys.argv[1]
input_file = "output.txt" if len(sys.argv) < 3 else sys.argv[2]
output_file = "output.png" if len(sys.argv) < 4 else sys.argv[3]

path_to_rule_folder = f"outputs/{path_folder}/"
path_to_wfc_output = f"outputs/{path_folder}/{input_file}"

# read output to variable
with open(path_to_wfc_output, "r") as f:
    output = f.read()

#split input
output = output.split("\n")
output = [line.split(" ") for line in output if line != ""]

width = len(output[0])
height = len(output)
print(width, height)

# load tiles
tiles = []
i = 0
while True:
    try:
        tile = mpimg.imread(f"{path_to_rule_folder}im{i}.png")
        tiles.append(tile)
        i += 1
    except FileNotFoundError:
        break

if len(tiles) == 0:
    print("No tiles found")
    exit(1)

tile_width = len(tiles[0])
tile_height = len(tiles[0][0])

# create image
img = np.zeros((height*tile_height, width*tile_width, 3), dtype=np.uint8)
for y, line in enumerate(output):
    for x, tile_id in enumerate(line):
        img[y*tile_height:(y+1)*tile_height, x*tile_width:(x+1)*tile_width] = tiles[int(tile_id)]*255

mpimg.imsave(f"outputs/{path_folder}/{output_file}", img)