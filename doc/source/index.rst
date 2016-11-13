photomosaic
===========

Tile images from a large collection into a mosaic which, viewed at a distance,
gives the impression of one large photo.

This implementation has several distinguishing features:

* Mix tiles of different sizes and place them in irreigular or overlapping
  positions --- not confined to the usual rectangular grid.
* Parition an image into tiles adaptively, subdividing regions of high detail
  into smaller tiles.
* Include a mask, and tile paritioning can roughly trace its edge, giving the
  appearance of puzzle pieces scattered on a table in some shape.
* To maximize contrast and reduce tile repetition, adapt the color palette of
  the target image to the palette of colors available in the tile pool.
* Obtain large collections of images to use as tiles using the official
  Flickr API. Easily collect attribution info and filter results by license.
* Employ perceptually-accurate color matching via the
  `colorspacious <colorspacious.readthedocs.io>`_ package.

Development
-----------

The code is factored with maximum flexibility in mind and uses only standard
data structures: numpy arrays, Python slice objects, and other built-in Python
types. It is easy to experiment with custom strategies for paritioning tiles,
characterizing colors, choosing matches, or drawing the final result.

You are cheerfully invited to contribute to this project
`on GitHub <https://github.com/danielballan/photomosaic>`_. In particular,
these topics might be interesting:

* Alternative matching algorithms that minimize tile repetition
* Alternative tile-partitioning algorithms
* Additional image-havesting tools (continuing to give due attention to
  licenses, attribution)

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   tutorial
   getting-tiles
   palette
   reference
   release-notes
