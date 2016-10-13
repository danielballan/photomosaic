import os
import photomosaic as pm


here = os.path.dirname(__file__)
POOL_PATH = '/tmp/photomosaic-docs-pool/pool.json'
pool = pm.import_pool(os.path.join(POOL_PATH))

# Load a sample image
from skimage import data
img = data.chelsea()  # cat picture!
# Create a mosiac with 15x15 tiles.
mos = pm.basic_mosaic(img, pool, (30, 30), depth=1)
from skimage.io import imsave
imsave(os.path.join(here, '..', '_static', 'generated_images', 'basic-depth1.png'), mos)
