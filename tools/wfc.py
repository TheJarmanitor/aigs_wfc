import numpy as np
import time


def wfc(tiles, rules, width, height, fixed_tiles=None, weigths=None):
    '''
    Wave Function Collapse algorithm
    :param tiles: list of tile names
    :param rules: list of rules for each tile, each rule is a list of possible tiles for each direction
    :param width: width of the grid
    :param height: height of the grid
    :param fixed_tiles: list of fixed tiles, each tile is a tuple of (x, y, tile)
    :param weigths: list of weigths for each tile
    '''

    # 0 -> up, 1 -> right, 2 -> down, 3 -> left
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    if weigths is None:
        weigths = [1 for _ in range(len(tiles))]

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

    
    def choose_random_weighted_tile(tiles):
        '''
        Choose a random tile from a list of tiles, with a probability based on the weigths
        :param tiles: list of tiles to choose from
        '''
        total = sum(weigths[tile] for tile in tiles)
        r = np.random.uniform(0, total)
        for tile in tiles:
            r -= weigths[tile]
            if r <= 0:
                return tile
        return tiles[-1]



    def collapse(x, y, superposition, fixed):
        '''
        Try to collapse the superposition at a given position to a single tile
        :param x: x position to collapse
        :param y: y position to collapse
        :return: False if no possible tile results in solvable state, True otherwise
        '''
        #TODO add local weights
        possible_tiles  = list(superposition[y][x])
        while len(possible_tiles) > 0:
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
    

    def run(superposition, fixed):
        '''
        runs the algorithm until completion or failure
    
        :return: True if the algorithm was successful, False otherwise
        '''
        while True:
            #x, y = collapse_manhattan_heuristic(superposition, fixed)
            x, y = collapse_lowest_entropy_heuristic(superposition, fixed)
            if x == -1:
                return True
            if not collapse(x, y, superposition, fixed):
                return False
    
    def print_tiles(fixed):
        for row in fixed[::-1]:
            print("".join(tiles[cell] for cell in row))
    
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
        print("Failed, retrying")
    print_tiles(fixed)






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
    size = 80
    for rules in [rules_mountain, rules_islands_beaches]:
        start = time.time()
        wfc(["a", "b", "c"], rules, size, size, [(25,25,1)], [2,1,2])
        print(f"Size {size} took {time.time()-start} seconds")
