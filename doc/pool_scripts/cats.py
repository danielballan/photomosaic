import os
import photomosaic.flickr
import photomosaic as pm


class MissingAPIKey(Exception):
    ...


if not os.path.isfile(os.path.expanduser('~/pools/cats/pool.json')):
    try:
        FLICKR_API_KEY = os.environ['FLICKR_API_KEY']
    except KeyError:
        raise MissingAPIKey(
            "This script requires the environment variable FLICKR_API_KEY "
            "to run. It will be not be available on pull requests from "
            "other forks. See "
            "https://docs.travis-ci.com/user/pull-requests/#pull-requests-and-security-restrictions"
            ) from None
    pm.set_options(flickr_api_key=FLICKR_API_KEY)

    photomosaic.flickr.from_search('cats', '~/pools/cats/')
    pool = pm.make_pool('~/pools/cats/*.jpg')
    pm.export_pool(pool, '~/pools/cats/pool.json')  # save color analysis for future reuse
else:
    print("Pool was found in ~pools/cats/. No action needed.")
