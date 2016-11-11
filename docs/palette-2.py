import matplotlib.pyplot as plt
import photomosaic as pm
POOL_PATH = '/tmp/photomosaic-docs-pool/pool.json'
pool = pm.import_pool(POOL_PATH)
pm.plot_palette(pm.color_palette(list(pool.values())))
plt.suptitle('Color Palette of Tile Pool')