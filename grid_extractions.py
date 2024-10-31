# %%
import jax.numpy as jnp
import numpy
import matplotlib.pyplot as plt
import cv2
from collections import Counter

# %%

test_grid = cv2.imread("test_grid.png")
test_grid = cv2.cvtColor(test_grid, cv2.COLOR_BGR2RGB)
test_grid = cv2.resize(test_grid, (576, 576))
plt.imshow(test_grid)
# %%
print(test_grid.shape)


# %%
def get_tiles(image, tile_size):
    H, W, _ = image.shape

    tiles = [
        image[x : x + tile_size, y : y + tile_size, :]
        for x in range(0, W, tile_size)
        for y in range(0, H, tile_size)
    ]
    return tiles


def get_hash_tiles(tiles):
    def pHash(cv_image):
        imgg = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h = cv2.img_hash.pHash(imgg)  # 8-byte hash
        pH = int.from_bytes(h.tobytes(), byteorder="big", signed=False)
        return pH

    hashed_tiles = [pHash(tile) for tile in tiles]
    return hashed_tiles


def get_unique_tile_hashes(tiles):
    hashed_tiles = get_hash_tiles(tiles)

    return list(dict.fromkeys(hashed_tiles))


def get_id_tile_dict(tiles):
    unique_hashes = get_unique_tile_hashes(tiles)

    hash_int_dict = dict([(unique_hashes[i], i) for i in range(len(unique_hashes))])

    hash_tile_dict = dict(zip(get_hash_tiles(tiles), tiles))

    int_tile_dict = dict([(hash_int_dict[i], hash_tile_dict[i]) for i in unique_hashes])

    return int_tile_dict


def get_id_list(tiles):
    hashed_tiles = get_hash_tiles(tiles)
    unique_hashes = get_unique_tile_hashes(tiles)

    hash_int_dict = dict([(unique_hashes[i], i) for i in range(len(unique_hashes))])

    int_tiles = [hash_int_dict[hash] for hash in hashed_tiles]

    return int_tiles


# %%
tiles = get_tiles(test_grid, 576)
plt.imshow(tiles[1])
