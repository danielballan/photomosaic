POOL_PATH = '/tmp/photomosaic-docs-pool/pool.json'
import photomosaic as pm
pool = pm.import_pool(POOL_PATH)
pm.plot_palette(pm.color_palette(list(pool.values())))