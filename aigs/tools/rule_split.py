# Creates a WFC rule set from an input image

from typing import List
import numpy as np
from PIL import Image
from sys import argv
import os
import pickle

# enum for directions
class Direction:
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3
    def opposite(self):
        return Direction((self.value + 2) % 4)

class Color:
    def __init__(self, r: int, g: int, b: int):
        self.r = r
        self.g = g
        self.b = b
        self.hash = r << 16 | g << 8 | b
    def __eq__(self, other: 'Color'):
        return self.hash == other.hash

class RuleSet:
    class Tile:
        def __init__(self, tile: List[Color], id):
            self.id = id
            self.tile = tile
            self.rules = [set() for _ in range(4)]
        def same_as(self, tile) -> bool:
            return all(a == b for a, b in zip(self.tile, tile))
    
    def __init__(self, image: List[List[Color]], tile_size: int):
        self.tile_size = tile_size
        self.tiles = []
        self.id_counter = 0
        if len(image) % tile_size != 0 or len(image[0]) % tile_size != 0:
            raise ValueError("Image dimensions must be divisible by tile size")
        self.image_id = [[-1 for _ in range(len(image[0]) // tile_size)] for _ in range(len(image) // tile_size)]
        
        # split input image into tiles and id them by uniqueness
        for y in range(0, len(image), tile_size):
            for x in range(0, len(image[0]), tile_size):
                self.add_tile(image, x, y)

        # add horizontal rules
        for y in range(len(self.image_id)):
            for x in range(len(self.image_id[0]) - 1):
                self.tiles[self.image_id[y][x]].rules[Direction.RIGHT].add(self.image_id[y][x + 1])
                self.tiles[self.image_id[y][x + 1]].rules[Direction.LEFT].add(self.image_id[y][x])

        # add vertical rules
        for y in range(len(self.image_id) - 1):
            for x in range(len(self.image_id[0])):
                self.tiles[self.image_id[y + 1][x]].rules[Direction.UP].add(self.image_id[y][x])
                self.tiles[self.image_id[y][x]].rules[Direction.DOWN].add(self.image_id[y + 1][x])
        
    
    def add_tile(self, image, x, y):
        tile = [image[y + dy][x + dx] for dy in range(self.tile_size) for dx in range(self.tile_size)]
        for i, t in enumerate(self.tiles):
            if t.same_as(tile):
                self.image_id[y//self.tile_size][x//self.tile_size] = i
                return
        self.tiles.append(self.Tile(tile, self.id_counter))
        self.image_id[y // self.tile_size][x // self.tile_size] = self.id_counter
        self.id_counter += 1
    

    # this is generic wfcrules format
    def output_to_folder_xml(self, name: str):
        # get /outputs/ folder
        output_dir = os.path.join(os.getcwd(), "outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # create folder for this rule set
        tileset_dir = os.path.join(output_dir, name)
        if not os.path.exists(tileset_dir):
            os.makedirs(tileset_dir)
        # save tiles as images
        for t in self.tiles:
            img = Image.new("RGB", (self.tile_size, self.tile_size))
            img.putdata([(c.r, c.g, c.b) for c in t.tile])
            img.save(os.path.join(tileset_dir, f"im{t.id}.png"))
        
        # save rules as xml
        with open(os.path.join(output_dir, f"{name}.xml"), "w") as f:
            f.write("<set>\n")
            f.write(f"  <tiles>\n")
            for t in self.tiles:
                f.write(f"    <tile name=\"im{t.id}\" symmetry=\"F\"/>\n")
            f.write("  </tiles>\n")
            f.write("  <neighbors>\n")
            for t in self.tiles:
                for i in range(4):
                    for r in t.rules[i]:
                        f.write(f"    <neighbor left=\"im{t.id} {i}\" right=\"im{r} {i}\"/>\n")
            f.write("  </neighbors>\n")
            f.write("</set>\n")
    
    # this is our custom format
    def output_to_folder_rules(self, name: str, output_dir = None):
        # get /outputs/ folder
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # create folder for this rule set
        tileset_dir = os.path.join(output_dir, name)
        if not os.path.exists(tileset_dir):
            os.makedirs(tileset_dir)
        # save tiles as images
        for t in self.tiles:
            img = Image.new("RGB", (self.tile_size, self.tile_size))
            img.putdata([(c.r, c.g, c.b) for c in t.tile])
            img.save(os.path.join(tileset_dir, f"im{t.id}.png"))
        
        rules = []
        for t in self.tiles:
            rules.append([])
            for i in range(4):
                rules[-1].append(set(t.rules[(i+3)%4]))
        pickle.dump(rules, open(os.path.join(tileset_dir, "rules.pkl"), "wb"))


        


if __name__ == "__main__":
    if(len(argv) < 4):
        print("Usage: python rule_split.py <input_image> <tile_size> <otuput_name>")
        exit(1)
    img = Image.open(argv[1])
    img = img.convert("RGB")
    img = np.array(img)
    tile_size = int(argv[2])
    rules = RuleSet([list(map(lambda x: Color(x[0], x[1], x[2]), row)) for row in img], tile_size)
    print(f"Created {rules.id_counter} tiles")
    # #pinrt id map
    # for row in rules.image_id:
      #  print(row)
    # #print rules
    # for t in rules.tiles:
    #    print(f"Tile {t.id}")
    #    for i, r in enumerate(t.rules):
    #        print(f"  {i}: {r}")
    name = argv[3]
    rules.output_to_folder_rules(name)

        
        
        