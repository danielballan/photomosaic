import os
import photomosaic as pm
from skimage import data
from skimage.io import imsave


here = os.path.dirname(__file__)
POOL_PATH = '/tmp/photomosaic-docs-pool/pool.json'
pool = pm.import_pool(os.path.join(POOL_PATH))

# Load a sample image
img = data.chelsea()  # cat picture!
# Create a mosiac with 15x15 tiles.
mos = pm.basic_mosaic(img, pool, (30, 30))
imsave(os.path.join(here, '..', '_static', 'generated_images', 'basic.png'), mos)
