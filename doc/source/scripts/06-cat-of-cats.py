import os
import photomosaic as pm
import photomosaic.flickr
import matplotlib.pyplot as plt


# For these published examples we use os.environ to keep our API key private.
# Just set your own Flickr API key here.
FLICKR_API_KEY = os.environ['FLICKR_API_KEY']

# Get a pool of cat photos from Flickr.
pm.set_options(flickr_api_key=FLICKR_API_KEY)
photomosaic.flickr.from_search('cats', 'cats/', 1000)
pool = pm.make_pool('cats/*.jpg')
pm.export_pool(pool, 'cats/pool.json')  # save color analysis for future reuse

# Build mosaic.
mosaic = pm.basic_mosaic(img, pool, (30, 30), depth=4)
plt.plot(mosaic)
plt.show()
