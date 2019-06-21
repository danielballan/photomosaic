import os
import numpy as np
import photomosaic as pm
from skimage.io import imsave
from skimage.data import chelsea
from skimage import img_as_float


here = os.path.dirname(__file__)
POOL_PATH = '/tmp/photomosaic-docs-pool/pool.json'
pool = pm.import_pool(os.path.join(POOL_PATH))

image = img_as_float(chelsea())
converted_img = pm.perceptual(image)
scaled_img = pm.rescale_commensurate(converted_img, grid_dims=(30, 30),
                                     depth=0)
tiles = pm.partition(scaled_img, grid_dims=(30, 30), depth=0)
tile_colors = [np.mean(scaled_img[tile].reshape(-1, 3), 0)
               for tile in tiles]
match = pm.simple_matcher(pool)
matches = [match(tc) for tc in tile_colors]
canvas = np.ones_like(scaled_img)  # white canvas
mos = pm.draw_mosaic(canvas, tiles, matches)

imsave(os.path.join(here, '..', '_static', 'generated_images',
                    'no-palette-adjustment.png'), mos)

adapted_img = pm.adapt_to_pool(converted_img, pool)
imsave(os.path.join(here, '..', '_static', 'generated_images',
                    'adapted-chelsea.png'), pm.rgb(adapted_img))
