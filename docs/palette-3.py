from skimage.data import chelsea
from skimage import img_as_float
import photomosaic as pm
image = img_as_float(chelsea())
converted_img = pm.perceptual(image)
POOL_PATH = '/tmp/photomosaic-docs-pool/pool.json'
pool = pm.import_pool(POOL_PATH)
adjusted_img = pm.adapt_to_pool(converted_img, pool)
pm.plot_palette(pm.color_palette(adjusted_img))