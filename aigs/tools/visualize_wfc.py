import numpy as np
import sys
import matplotlib.image as mpimg

# Render output of WFC as an image
def visualize_wfc(path_folder = "test", input_file = "output.txt", output_file = "output.png", SHOW_NUKES = True):
    
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
    
    if SHOW_NUKES:
        # load nukes
        img = np.zeros((height, width, 3), dtype=np.uint8)
        nukes = np.zeros((height, width), dtype=np.uint8)
        with open(f"outputs/{path_folder}/{output_file[:-4]}_nukelog.txt", "r") as f:
            for y, line in enumerate(f):
                for x, nuke in enumerate(line.split(" ")):
                    img[y, x] = np.array([255, 0, 0]) * int(nuke)/5.0
                    nukes[y, x] = int(nuke)
        mpimg.imsave(f"outputs/{path_folder}/{output_file[:-4]}_nukelog.png", img)
    
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
    
    # add red overlay for nukes on the output image
    if SHOW_NUKES:
        for x in range(width*tile_width):
            for y in range(height*tile_height):
                t = nukes[y//tile_height, x//tile_width]/5.0
                t = 1 if t > 1 else t
                img[y, x] = img[y, x] * (1-t) + np.array([255, 0, 0]) * t
    
    mpimg.imsave(f"outputs/{path_folder}/{output_file}", img)