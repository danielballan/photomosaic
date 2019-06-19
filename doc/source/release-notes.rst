Release Notes
=============

0.3.0 (11 Nov. 2016)
--------------------

Enhancements
++++++++++++

* Implement histogram mapping to perform tiling in an adaptive color palette.
* Add ability to subdivide tiles based on local contrast.
* Add utilities to padding, translating, and randomly scattering tiles relative
  to their true position.
* Add a tool for downloading images from Flickr for use as tile pools.
* Add convenience functions for import/exporting tile pools, plotting palettes,
  and converting between color spaces.
* Add alternative implementation of ``make_pool`` that runs faster using dask.
  (More use of dask is planned.)

API Changes
+++++++++++

* Renamed ``generate_tile_pool`` to ``rainbow_of_squares``; there may be
  different tile pool generators in future.
* Change ``SimpleMatcher`` from a class to a closure, ``simple_matcher``.

Other
+++++

* Change license to 3-clause BSD (with signoff from both contributors).
* Publish sphinx documentation.
* Run automated unit tests on Travis-CI.

0.2.2 (5 Oct. 2016)
-------------------

This tag marks a nearly-total rewrite of the project. (Tags v0.2.0 and v0.2.1
have the same code but some build problems with the package.)

0.1.0 (2012)
------------

This tag is maintained for use with legacy scripts, but it is not recommended
for new users.
