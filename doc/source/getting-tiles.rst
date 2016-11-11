Obtaining Images for Tiles
==========================

The best mosaics come from using a large pool of tile images with a wide
variety of colors. Typically, about 300 images is the minimum for good results.
And of course, it's more fun if the pool images are thematically related to the
target image: George Washington drawn from presidential protraits, Yoda drawn
from frames from Star Wars, etc.

The photomosaic package includes convenience functions for downloading image
collections from public websites. They use the websites' official public API to
"politely" scrape images and collect attribution information. For each image
file, a text file (JSON) with attribution information is also saved. It is
possible to filter queries based on their copyright license. If you plan to
share or sell your mosiac, pay due attention to the copyright license on the
images you download.

At this time, we only support Flickr, but more will be added in the future.

Flickr
------

Apply for a Flickr API key
`here <https://www.flickr.com/services/api/keys/apply/>`_. You will get an API
key and a secret key. For this application we don't need the secret key.

.. code-block:: python

    import photomosaic as pm
    import photomosaic.flickr as pmf

    # Provide your API key.
    pm.set_options(flickr_api_key='YOUR_API_KEY_HERE')

    # Download the first 300 results from a search for 'cats cars'.
    pm.from_search('cats cars', 'some_directory', 300)

    # Download a specific album.
    pmf.from_url('https://www.flickr.com/photos/anapaunkovic/sets/72157646150658189',
                 'some_directory')

.. autofunction:: photomosaic.flickr.from_search

.. autofunction:: photomosaic.flickr.from_url
