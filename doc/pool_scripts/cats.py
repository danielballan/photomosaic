import os
import photomosaic.flickr
import photomosaic as pm


if not os.path.isfile('~/pools/cats/pool.json'):
    FLICKR_API_KEY = os.environ['FLICKR_API_KEY']
    pm.set_options(flickr_api_key=FLICKR_API_KEY)

    photomosaic.flickr.from_search('cats', '~/pools/cats/')
    pool = pm.make_pool('~/pools/cats/*.jpg')
    pm.export_pool(pool, '~/pools/cats/pool.json')  # save color analysis for future reuse
