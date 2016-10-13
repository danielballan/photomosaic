import matplotlib.pyplot as plt
import photomosaic as pm
POOL_PATH = '/tmp/photomosaic-docs-pool/pool.json'
pool = pm.import_pool(POOL_PATH)
from skimage.data import chelsea
from skimage import img_as_float
image = img_as_float(chelsea())
converted_img = pm.perceptual(image)
adapted_img = pm.adjust_to_palette(converted_img, pool)
scaled_img = pm.rescale_commensurate(adapted_img, grid_dims=(30, 30), depth=1)
tiles = pm.partition(scaled_img, grid_dims=(30, 30), depth=1)
annotated_img = pm.draw_tile_layout(pm.rgb(scaled_img), tiles)
plt.imshow(annotated_img)