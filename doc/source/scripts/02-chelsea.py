import os
from skimage.data import chelsea
from skimage.io import imsave


here = os.path.dirname(__file__)
img = chelsea()
imsave(os.path.join(here, '..', '_static', 'generated_images', 'chelsea.png'),
       img)
