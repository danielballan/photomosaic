get_ipython().magic('matplotlib qt')
import matplotlib.pyplot as plt
import numpy as np
import photomosaic as pm
from skimage.io import imread, imsave
from skimage import img_as_float
pool = pm.import_pool('wedding_photos.pool')

pm.plot_palette(pm.color_palette(list(pool.values())))
plt.suptitle('wedding photos pool palette PER')

image = img_as_float(imread('/Users/dallan/Desktop/wedding-pictures/Ceremony/320SA_KFA_2628.jpg'))
pimg = pm.perceptual(image)

pm.plot_palette(pm.color_palette(image))
plt.suptitle('target photo palette RGB')

pm.plot_palette(pm.color_palette(pimg))
plt.suptitle('target photo palette PER')

pm.plot_palette(pm.color_palette(pm.rgb(list(pool.values()))))
plt.suptitle('wedding photos pool palette RGB')

adapted = pm.adapt_to_pool(pimg, pool)
adapted_rgb = pm.rgb(adapted)
plt.figure(); plt.imshow(adapted_rgb)
plt.suptitle('adapted target photo (adapted PER; displayed RGB)')

pm.plot_palette(pm.color_palette(adapted))
plt.suptitle('adapted target PER')

pm.plot_palette(pm.color_palette(pm.rgb(adapted)))
plt.suptitle('adapted target RGB')
