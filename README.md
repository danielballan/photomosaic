photomosaic
=========

Assemble thumbnail-sized images from a large collection into a mosaic which, viewed at a distance, gives the impression of one large photo.

Usage:
-----

    import photomosaic as pm
    
    pool('folder-of-many-images/', 'imagepool.db')
    tiles = target('original.jpg', 20, 'imagepool.db')
    # 20 = tile size in px. For rectangular tiles, use tuple: (40, 30).
    img = photomosaic(tiles, 'imagepool.db)
    img.save('mosaic.jpg')


Dependences:
-----------

* numpy
* scipy
* PIL (that is, the Image module)
* sqlite3

Related Project:
---------------
[John Louis Del Rosario](https://github.com/john2x) also has a [photomosaic project](https://github.com/john2x/photomosaic) in Python. I studied his code while I began writing my own, and there are similarities. However, my algorithm for characterizing tiles and finding matches, which is accomplished mostly through SQL queries, is substatially different.
