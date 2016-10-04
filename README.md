photomosaic
===========

Assemble thumbnail-sized images from a large collection into a mosaic which,
viewed at a distance, gives the impression of one large photo.

Example
-------

```python
import photomosaic as pm

pm.set_options(colorspace='sRGB1')  # effectively turning off fancy colorspace
stuff for now
pm.set_options(imread={'plugin': 'matplotlib'})  # the default reader fails on
GIFs for some reason

# Load a sample image
from skimage import data
img = data.chelsea()

# Generate dummy images covering the color gamut to use as a pool.
pm.generate_tile_pool('pool')

# Build the pool (analyze the dummy images).
pool = pm.make_pool('pool/*.gif')

# Create the mosiac.
mos = pm.basic_mosaic(img, pool, (15, 15))

import matplotlib.pyplot as plt
plt.imshow(mos)
```

Related Project
---------------
[John Louis Del Rosario](https://github.com/john2x) also has a [photomosaic project](https://github.com/john2x/photomosaic) in Python. I studied his code while I began writing my own, and there are similarities. However, my algorithm for characterizing tiles and finding matches, which is accomplished mostly through SQL queries, is substatially different.
