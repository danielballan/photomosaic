photomosaic
=========

Assemble thumbnail-sized images from a large collection into a mosaic which, viewed at a distance, gives the impression of one large photo.

One-Step Usage
--------------

In two lines:

    import photomosaic
    photomosaic.simple('folder-of-many-images/', 'original.jpg', 20, 'mosaic.jpg')

where 20 is the size of a tile in pixels.

Basic Usage
-----------

Alternatively, you can run the process one step at a time. This gives access to more options. 

    import photomosaic as pm
    
    pm.pool('folder-of-many-images/', 'imagepool.db')
    img = pm.open('original.jpg')
    img = pm.tune(img) # Adjust colors levels to what's availabe in the pool.
    tiles = pm.partition(img, 20)
    # 20 is the tile size in px. For rectangular tiles, use tuple like (40, 30).
    pm.analyze(tiles, 'imagepool.db')
    mosaic = pm.photomosaic(tiles, 'imagepool.db')
    mosaic.save('mosaic.jpg')

The most time-consuming step is ``analyze()``, which compares every tile in the target image to every image in the pool. The final step, ``photomosaic()``, which identifies the closest matches and generates the actual mosaic, is relatively speedy. Once ``analyze()`` is done, it is convenient to run ``photomosaic()`` several times while experimenting with different settings.

For more on said settings, see Advanced Usage below.

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


