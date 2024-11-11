import numpy as np
import time
import pickle
import jax.numpy as jnp
from sys import argv
from jax import random

def wfc(tiles, rules, width, height, fixed_tiles=[], weights=None, path_to_output=None, layout_map = None):
    '''
    Wave Function Collapse algorithm
    :param tiles: list of tile names
    :param rules: list of rules for each tile, each rule is a list of possible tiles for each direction
    :param width: width of the grid
    :param height: height of the grid
    :param fixed_tiles: list of fixed tiles, each tile is a tuple of (x, y, tile)
    :param weights: list of weights for each tile
    :param path_to_output: path to save the output to (if None, output is printed)
    '''
    # 0 -> up, 1 -> right, 2 -> down, 3 -> left
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    if weights is None:
        weights = [1 for _ in range(len(tiles))]
    # count the number of times this position was CENTER of a nuke
    nuke_count = [[0 for _ in range(width)] for _ in range(height)]
    # count the number of times this position was nuked (even if not the source of nuke)
    nuked_times = [[0 for _ in range(width)] for _ in range(height)]
    nuke_treshold = 5

    def collapse_lowest_entropy_heuristic(superposition, fixed):
        '''
        Finds the position with the least amount of possible tiles to collapse
        '''
        min = float("inf")
        min_x = -1
        min_y = -1
        for x in range(width):
            for y in range(height):
                if fixed[y][x] == -1 and len(superposition[y][x]) < min:
                    min = len(superposition[y][x])
                    min_x = x
                    min_y = y
        return min_x, min_y

    def collapse_manhattan_heuristic(superposition, fixed):
        '''
        Find the position with the smallest manhattan distance to collapsed tiles and lowest entropy
        '''
        #TODO: instead of calculating the map every time, update it during collapsing instead
        manhattan_map = [[-1 for _ in range(width)] for _ in range(height)]
        queue = []
        for x in range(width):
            for y in range(height):
                if fixed[y][x] != -1:
                    manhattan_map[y][x] = 0
                    queue.append((x, y))
        while len(queue) > 0:
            x, y = queue.pop(0)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if manhattan_map[ny][nx] == -1:
                    manhattan_map[ny][nx] = manhattan_map[y][x] + 1
                    queue.append((nx, ny))
        min = float("inf")
        min_x = -1
        min_y = -1
        for x in range(width):
            for y in range(height):
                if fixed[y][x] != -1:
                    continue
                if len(superposition[y][x]) < min:
                    min = len(superposition[y][x])
                    min_x = x
                    min_y = y
                elif len(superposition[y][x]) == min and manhattan_map[y][x] < manhattan_map[min_y][min_x]:
                    min_x = x
                    min_y = y
        return min_x, min_y

    
    def choose_random_weighted_tile(tiles, weights = weigths):
        '''
        Choose a random tile from a list of tiles, with a probability based on the weights
        :param tiles: list of tiles to choose from
        '''
        total = sum(weights[tile] for tile in tiles)
        r = np.random.uniform(0, total)
        for tile in tiles:
            r -= weights[tile]
            if r <= 0:
                return tile
        return tiles[-1]

    def choose_local_weighted_tile(possible_tiles, local_weights, rng):
        '''
        Choose a tile based on the local weights/probabilities. 
        A greater probability of tiles increases the chance of the specific tile being chosen.
        '''
        # Filter weights for only the eligible tiles, need to specify all tiles
        rng, key = random.split(rng)
        
        #filtered_tiles = [tile for tile in all_tiles if tile in possible_tiles]
        filtered_weights = [local_weights[tile] for tile in possible_tiles]
        
        weights_sum = sum(filtered_weights)
        normalized_weight = [w / weights_sum for w in filtered_weights]
    
        return random.choice(key, jnp.array(possible_tiles), p=jnp.array(normalized_weight)).item(), rng
    
    def sample_layout(x,y) -> int:
        return layout_map[y][x]
        

    def collapse(x, y, superposition, fixed, rng, local_weights: list = None):
        '''
        Try to collapse the superposition at a given position to a single tile
        :param x: x position to collapse
        :param y: y position to collapse
        :return: False if no possible tile results in solvable state, True otherwise
        '''
        #TODO add local weights
        possible_tiles  = list(superposition[y][x])
        while len(possible_tiles) > 0:
            if local_weights:
                color = sample_layout(x, y)
                tile, rng = choose_local_weighted_tile(possible_tiles, local_weights[color], rng) 
            else:
                tile = choose_random_weighted_tile(possible_tiles)
            superposition[y][x] = {tile}
            fixed[y][x] = tile
            if not propagate(x, y, superposition, fixed):
                fixed[y][x] = -1
                possible_tiles.remove(tile)
            else:
                return True
        return False
    
    def propagate(x, y, superposition, fixed):
        '''
        Propagate the contraints as a wave across the grid, beginning at the given position
        :param x: x position to start the propagation
        :param y: y position to start the propagation
        :return: False if tile with no possible tiles was found, True otherwise
        '''
        unique_queue = set()
        removed = []
        unique_queue.add((x,y))
        while len(unique_queue) > 0:
            (x,y) = unique_queue.pop()
            change = False
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if fixed[ny][nx] != -1:
                    continue
                for tile in list(superposition[ny][nx]):
                    if not is_valid_tile(nx, ny, tile, superposition):
                        superposition[ny][nx].remove(tile)
                        removed.append((nx,ny,tile))
                        change = True
                        if len(superposition[ny][nx]) == 0:
                            #return to original state
                            for x,y,tile in removed:
                                superposition[y][x].add(tile)
                            return False
            if change:
                for nx, ny in directions:
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    unique_queue.add((nx,ny))
        return True
    
    def propagate_extra(xys, superposition, fixed):
        '''
        Propagate the contraints as a wave across the grid, BUT also checks for new possible tiles to add to superpositions
        :param xys: list of x,y positions to check in the propagation
        :return: False if tile with no possible tiles was found, True otherwise
        '''
        unique_queue = set()
        for x,y in xys:
            unique_queue.add((x,y))
        while len(unique_queue) > 0:
            (x,y) = unique_queue.pop()
            change = False
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if fixed[ny][nx] != -1:
                    continue
                new_superpositions = set()
                for tile in tiles:
                    if is_valid_tile(nx, ny, tile, superposition):
                        new_superpositions.add(tile)
                if len(new_superpositions) == 0:
                    return False
                if new_superpositions != superposition[ny][nx]:
                    superposition[ny][nx] = new_superpositions
                    change = True
            if change:
                for nx, ny in directions:
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    unique_queue.add((nx,ny))
        return True

    def is_valid_tile(x, y, tile, superposition):
        '''
        Checks if a tile is valid at a given position, by having atleast one valid neighbour for each direction
        :param x: x position of the tile
        :param y: y position of the tile
        :param tile: the tile type to check
        :return: True if the tile is valid, False otherwise
        '''
        for i,(dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if all(a not in superposition[ny][nx] for a in rules[tile][i]):
                return False
        return True
    
    def nuke(x, y, superposition, fixed):
        '''
        Remove all possible tiles from a given position within a given radius
        :param x: x position to nuke
        :param y: y position to nuke
        :param radius: radius to nuke
        '''
        radius = nuke_count[y][x] + 1
        nuke_count[y][x] = radius
        print(f"Nuking {x},{y} with radius {radius}")
        if radius >= nuke_treshold:
            return False
        nuked_tiles = []
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                superposition[ny][nx] = set(range(len(tiles)))
                fixed[ny][nx] = -1
                nuked_tiles.append((nx,ny))
                nuked_times[ny][nx] += 1
        if not propagate_extra(nuked_tiles,superposition,fixed):
            return nuke(x, y, superposition, fixed)
        return True
    

    def run(superposition, fixed):
        '''
        runs the algorithm until completion or failure
    
        :return: True if the algorithm was successful, False otherwise
        '''
        rng = random.PRNGKey(0)
        while True:
            #x, y = collapse_manhattan_heuristic(superposition, fixed)
            x, y = collapse_lowest_entropy_heuristic(superposition, fixed)
            if x == -1:
                return True
            rng, key = random.split(rng)
            if not collapse(x, y, superposition, fixed, key, weights):
                if not nuke(x, y, superposition, fixed):
                    return False
    
    def print_tiles(fixed):
        for row in fixed[::-1]:
            print(" ".join(str(tiles[cell]) for cell in row))
    
    def save_to_file(fixed,path):
        with open(path, "w") as f:
            for row in fixed[::-1]:
                f.write(" ".join(str(tiles[cell]) for cell in row) + "\n")
        with open(path.replace(".txt","_nukelog.txt"), "w") as f:
            for row in nuked_times[::-1]:
                f.write(" ".join(str(cell) for cell in row) + "\n")

    
    while True:
        superposition = [[set(range(len(tiles))) for _ in range(width)] for _ in range(height)]
        # -1 ->isnt fixed, 0+ -> fixed to that tile
        fixed = [[-1 for _ in range(width)] for _ in range(height)]


        for x, y, tile in fixed_tiles:
            fixed[y][x] = tile
            superposition[y][x] = {tile}
            if not propagate(x, y, superposition, fixed):
                print("Invalid fixed tiles")
                return
            
        if run(superposition, fixed):
            break
        #count percentage of fixed tiles
        fixed_count = sum(1 for row in fixed for cell in row if cell != -1)
        print(f"Failed ({100*fixed_count/(width*height):.3f}% collapsed), retrying")
    if path_to_output is not None:
        save_to_file(fixed,path_to_output)
    else:
        print_tiles(fixed)


def local_weight(bundle: list, prob_magnitude: float = 10.0, default_weight: float = 0.01) -> list:
    # Initialize weight map with default values
    weight_map = [[default_weight for _ in range(18)] for _ in range(4)]
    
    # Update weights for tiles in the bundle
    for tile in range(18):
        for input_color in range(4):
            if tile in bundle[input_color]:
                weight_map[input_color][tile] = prob_magnitude
    
    # Normalize each row in the weight map
    weight_map_norm = []
    for row in weight_map:
        row_sum = sum(row)
        normalized_row = [w / row_sum for w in row]
        weight_map_norm.append(normalized_row)
    
    return weight_map_norm




if __name__ == "__main__":
                  
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

    #time testing
    
    path_folder = "test"
    if len(argv) > 1:
        path_folder = argv[1]

    size = 64 if len(argv) < 3 else int(argv[2])
    path_to_file=f"outputs/{path_folder}/rules.pkl"
    rules = pickle.load(open(path_to_file, "rb"))

    #print rules
    for i, rule in enumerate(rules):
        print(f"Tile {i}")
        for j, r in enumerate(rule):
            print(f"  {j}: {r}")
            
    layout_map = [[0 for _ in range(size)] for _ in range(size)]
    
    for x in range(size):
        for y in range(size):
            if x > y:
                layout_map[y][x] = 1
            if abs(x - y) < 2:
                layout_map[y][x] = 2
            if abs(size-x) < 10 and abs(size-y) < 10:
                layout_map[y][x] = 3
            
    
    #local weights definition (change this when application is up and running)
    bundle=[
        [8],
        [1, 2, 3],
        [6, 17],
        [9]
    ]
    
    local_weights = local_weight(bundle)
    
    start = time.time()
    wfc([*range(len(rules))], rules, size, size, weights = local_weights, path_to_output=f"outputs/{path_folder}/output.txt", layout_map = layout_map)
    print(f"Size {size} took {time.time()-start} seconds")
    
    
