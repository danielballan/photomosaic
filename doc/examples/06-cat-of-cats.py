import photomosaic as pm
import matplotlib.pyplot as plt


# Build mosaic.
pool = pm.import_pool('~/pools/cats/pool.json')
mosaic = pm.basic_mosaic(img, pool, (30, 30), depth=4)
plt.plot(mosaic)
plt.show()
