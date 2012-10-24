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
    mosaic = pm.photomosaic(tiles, 'imagepool.db')
    mosaic = pm.untune(mosaic, orig_img) # Transform the color palette back.
    mosaic.save('mosaic.jpg')

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

A traditional photomosaic is a regular array of tiles. For a looser, scattered effect (imitating some works by [this artist](http://www.flickr.com/photos/tsevis/collections/)) tiles can be individually shrunk in place, leaving a margin that reveals the background. If the background is white, the overall effect is to lighten that tile.

Thus, shrinking is applied to all tiles that are darker than their targets, and it is applied in proportion to that descrepancy.

    ...
    img = photomosaic(tiles, 'imagepool.db', vary_size=True)

By default, a shrunken tile is placed in the center of its space, leaving even margins. For artistic effect, they can be randomly nudged off center.

    ...
    img = photomosaic(tiles, 'imagepool.db', vary_size=True,
                      random_margins=True)


