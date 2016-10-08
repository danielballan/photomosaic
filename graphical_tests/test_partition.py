"""
Draw a solid circle off center in a blank image.
Use the same array as both the image and the mask.
The tiles should subdivide along the curved edge to trace out a smooth circle.
"""
from skimage import draw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import photomosaic as pm

img = np.zeros((1000, 1000))
rr, cc = draw.circle(300, 500, 150)
img[rr, cc] = 1
tiles = pm.partition(img, (10, 10), mask=img.astype(bool), depth=3)
plt.imshow(pm.draw_tile_layout(img, tiles, color=0.5))
plt.savefig('test-partition.png')
