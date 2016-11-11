import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.data import chelsea
import photomosaic as pm
image = img_as_float(chelsea())
converted_img = pm.perceptual(image)
pm.plot_palette(pm.color_palette(converted_img))
plt.suptitle('Color Palette of Original Target Image')