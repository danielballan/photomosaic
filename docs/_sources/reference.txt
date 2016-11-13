Reference
=========

.. currentmodule:: photomosaic

Candidate Image Pool
--------------------

.. autofunction:: make_pool
.. autofunction:: import_pool
.. autofunction:: export_pool
.. autofunction:: rainbow_of_squares

Color Conversion
----------------

.. autofunction:: perceptual
.. autofunction:: rgb
.. autofunction:: set_options

Utilities
---------

.. autofunction:: standardize_image
.. autofunction:: rescale_commensurate
.. autofunction:: crop_to_fit
.. autofunction:: sample_pixels

Color Palettes
--------------

.. autofunction:: adapt_to_pool
.. autofunction:: color_palette
.. autofunction:: palette_map
.. autofunction:: hist_map
.. autofunction:: plot_palette

Tiles
-----

.. autofunction:: partition
.. autofunction:: translate
.. autofunction:: pad
.. autofunction:: scatter

Matching
--------

.. autofunction:: simple_matcher
.. autofunction:: simple_matcher_unique

Drawing
-------

.. autofunction:: basic_mosaic
.. autofunction:: draw_mosaic
.. autofunction:: draw_tile_layout

Color Characterization
----------------------

This seems to perform worse than simply using :func:`numpy.mean`.

.. autofunction:: dominant_color
