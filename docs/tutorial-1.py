import matplotlib.pyplot as plt
import photomosaic as pm
POOL_PATH = '/tmp/photomosaic-docs-pool/pool.json'
pool = pm.import_pool(POOL_PATH)
from skimage.data import chelsea
from skimage import img_as_float
image = img_as_float(chelsea())
converted_img = pm.perceptual(image)
adapted_img = pm.adapt_to_pool(converted_img, pool)
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 10))
ax1.imshow(pm.rgb(converted_img))
ax1.set_title("Before ('converted_img')")
ax2.imshow(pm.rgb(adapted_img))
ax2.set_title("After ('adapted_img')")