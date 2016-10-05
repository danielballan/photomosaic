photomosaic
===========

Assemble thumbnail-sized images from a large collection into a mosaic which,
viewed at a distance, gives the impression of one large photo.

Example
-------

```python
import photomosaic as pm

# Load a sample image
from skimage import data
img = data.chelsea()  # cat picture!

# Generate dummy images covering the color gamut to use as a pool.
pm.generate_tile_pool('pool')

# Build the pool (analyze the dummy images).
pool = pm.make_pool('pool/*.gif')

# Create a mosiac with 15x15 tiles.
mos = pm.basic_mosaic(img, pool, (15, 15))

# (Optional) Plot the mosaic using matplotlib.
import matplotlib.pyplot as plt
plt.imshow(mos)

# Save it.
from skimage import imsave
imsave(mos, 'mosaic.png')
```

Related Project
---------------
[John Louis Del Rosario](https://github.com/john2x) also has a [photomosaic project](https://github.com/john2x/photomosaic) in Python. I studied his code while I began writing my own, and there are similarities. However, my algorithm for characterizing tiles and finding matches, which is accomplished mostly through SQL queries, is substatially different.
