Tutorial
========

Basic Example
-------------

To begin, you need an image that you want to mosaic-ify. You can load it like
so:

.. code-block:: python

    from skimage.io import imread
    img = imread('filepath')

Alternatively, you can use one of the built-in example images provided by
scikit-image library. We'll go with the cat picture, ``chelsea``.

.. code-block:: python

    # Load a sample image
    from skimage import data
    img = data.chelsea()  # cat picture!

For the record, this is Chelsea.

.. figure:: _static/generated_images/chelsea.png

Next, you need large collection of images to fill in the tiles in the mosaic.
If you don't have a large collection of images handy, you can generate a
collection of solid-color squares. Real photos are more interesting, but
solid-color squares are nice for experimentation.

.. code-block:: python

    import photomosaic as pm

    # Generate a collection of solid-color square images.
    pm.generate_tile_pool('pool/')

    # Analyze the collection (the "pool") of images.
    pool = pm.make_pool('pool/*.png')

Now we have everything we need to make a mosaic. Specify the target image
(``img``), the pool of available tile images (``pool``), and the number of
tiles to divide the image into.

.. code-block:: python

    # Create a mosiac with 30x30 tiles.
    mos = pm.basic_mosaic(img, pool, (30, 30))

Now ``mos`` is our mosaic. We can save it

.. code-block:: python

    from skimage.io import imsave
    imsave('mosaic.png', mos)

or plot it using matplotlib.

.. code-block:: python

    import matplotlib.pyplot as plt
    plt.imshow(mos)

.. figure:: _static/generated_images/basic.png

To make a more detailed mosaic, subdivide tiles in important regions. The
optional ``depth`` parameter selectively splits tiles into quadrants if they
contain a certain amount of contrast.

.. code-block:: python

    mos = pm.basic_mosaic(img, pool, (30, 30), depth=1)

.. figure:: _static/generated_images/basic-depth1.png

More fine-grained control over tile partitioning is shown in the next section.

Detailed Example
----------------

Paritioning Tiles
+++++++++++++++++

Adaptive Color Palette
++++++++++++++++++++++

Matching
++++++++

Drawing
+++++++

Misplacing Tiles
++++++++++++++++
