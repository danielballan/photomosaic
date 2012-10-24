photomosaic
=========

Assemble thumbnail-sized images from a large collection into a mosaic which, viewed at a distance, gives the impression of one large photo.

One-Step Usage
--------------

In two lines:

    import photomosaic
    photomosaic.simple('folder-of-many-images/', 'original.jpg', (30, 30), 'mosaic.jpg')

where (30, 30) is the number of tiles along each dimension.

Basic Usage
-----------

Alternatively, you can run the process one step at a time. This gives access to more options. 

    import photomosaic as pm
    
    pm.pool('folder-of-many-images/', 'imagepool.db')
    orig_img = pm.open('original.jpg')
    img = pm.tune(orig_img, 'imagepool.db') # Adjust colors levels to what's availabe in the pool.
    tiles = pm.partition(img, (10, 10))
    pm.analyze(tiles)
    mos = pm.mosaic(tiles, 'imagepool.db')
    mos = pm.untune(mosaic, orig_img) # Transform the color palette back.
    mos.save('mosaic.jpg')

Remarks on each step:


* Generating an image pool is by far the longest step (about 45 minutes for 10,000 images) but it only has to be done once, and images can be added later without redoing the whole thing. Just run it again on a new folder or on the same folder with new images; it will skip duplicates.

* If the color scheme of your target image is not well represented in your potential tiles, shading and detail are lost. ``tune()`` ameliorates this problem by adjusting the levels of your target image to match the palette of colors available in the image pool. It's optional. The best solution is to have an image pool with all the necessary colors well represented. 

* Partitioning the image into tiles takes no time at all.

* Analyzing 900 (30x30) tiles takes about 20 seconds.

* Generating a 30x30 mosaic takes about 30 seconds. Different styles and settings are available. See Advanced Usage below.

Dependences
-----------

* numpy
* scipy
* PIL (that is, the Image module)
* sqlite3

Related Project
---------------
[John Louis Del Rosario](https://github.com/john2x) also has a [photomosaic project](https://github.com/john2x/photomosaic) in Python. I studied his code while I began writing my own, and there are similarities. However, my algorithm for characterizing tiles and finding matches, which is accomplished mostly through SQL queries, is substatially different.

Advanced Usage
--------------

A traditional photomosaic is a regular array of tiles. For a different effect, the size of the tiles can be varied. Small tiles are best used in regions of high contrast. Start with big tiles, such as 10x10. Use the ``depth`` keyword to control how many times a tile can decide to subdivide into quarters, based on the contrast within it.

    tiles = pm.partition(img, (10, 10), depth=4)

``depth`` puts a limit on how far tile-splitting can go, but it does not control how many tiles will decide to split. ``hdr`` for "high dynamic range" sets that contrast level beyond which tiles will subdivide.

    tiles = pm.partition(img, (10, 10), depth=4, hdr=80) # many tiles
    tiles = pm.partition(img, (10, 10), depth=4, hdr=200) # or fewer tiles

For a looser, scattered effect (imitating some works by [this artist](http://www.flickr.com/photos/tsevis/collections/)) tiles can be individually shrunk in place, leaving a margin that reveals the background. If the background is white, the overall effect is to lighten that tile.

Thus, shrinking is applied to all tiles that are darker than their targets, and it is applied in proportion to that descrepancy.

    img = mosaic(tiles, 'imagepool.db', vary_size=True)

By default, ``mosaic()`` begins by choosing the best match for each tile. You can experiment with different settings and rebuild the mosaic without 
