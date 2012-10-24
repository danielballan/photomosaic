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

### Multiscale tiles

A traditional photomosaic is a regular array of tiles. For a different effect, the size of the tiles can be varied. Small tiles are best used in regions of high contrast. Start with big tiles, such as 10x10. Use the ``depth`` keyword to control how many times a tile can decide to subdivide into quarters, based on the contrast within it.

    tiles = pm.partition(img, (10, 10), depth=4)

``depth`` puts a limit on how far tile-splitting can go, but it does not control how many tiles will decide to split. ``hdr`` for "high dynamic range" sets that contrast level beyond which tiles will subdivide.

    tiles = pm.partition(img, (10, 10), depth=4, hdr=80) # many tiles
    tiles = pm.partition(img, (10, 10), depth=4, hdr=200) # or fewer tiles

Logs displayed by ``tiles()`` tell you how many tiles have been made, in total, after each generation. 2000-6000 is a reasonable range to aim for. You can go higher if you're willing to wait for ``mosaic()`` to run for more than 10 minutes.

### Photomosaics with curved edges (masked images)

#### Simple cut-outs 

Create a black-and-white image the same size and your target image. White areas will be kept, and black areas will be masked. Open the image and pass it to ``partition``. Color images work too; they will just be converted.

    mask_img = pm.open('mask.jpg')
    tiles = pm.partition(img, (10, 10), mask=mask_img)

Tiles that fall wholly in the black area of the mask will be left blank. Of course, small tiles are better for tracing a curved edge. When you invoke multiscale tiling along with the mask, small tiles will fill in along the edge. Any tiles that straddle the edge of the mask, containing some white and some non-white, are forced to subdivide -- down to the limit set by depth, as above.

    tiles = pm.partition(img, (10, 10), mask=mask_img, depth=4)

To examine the effect before you proceed with ``analyze()`` and ``mosaic()``, you can easily assemble the original tiles.

    pm.assemble_tiles(tiles)

#### Masked image with "debris"

If the mask image contains grey, these areas can be filled with tiles probabilitically, creating the halo of debris, something like a partially completed puzzle.

    mask_img = pm.open('mask-with-some-grey-in-it.jpg')
    tiles = pm.partition(img, (10, 10), depth=4, mask=mask_img, debris=True)

To my eye, the effect is better with small tiles. You can limit debris to N-children of the original 10x10 tiles. Use the kwarg ``min_debris_depth``, which defaults to 1.

    tiles = pm.partition(img, (10, 10), depth=4, mask=mask_img, debris=True, min_debris_depth=2)

Again, examine the effect before proceeding. 

    pm.assemble_tiles(tiles)

*Important*: If you use tune/untune, you should provide untune with the mask, or the solid background will seriously distort the palette. The call is:

    mos = pm.untune(mos, orig_img, mask)

### Tile matching and repetition

This is how tile images are chosen:
* Images are rated by their closeness to the target tile. Technical details: "Closeness" is Euclidean distance in Lab color space, which is a good proxy to perceived color difference. The closeness is measured separately in the four quadrants of the tile, and then averaged.
* Ratings are adjusted randomly by up to 2.3, which is the "just-noticeable difference" in color. Thus, if one match is much better than the others, it will be chosen, but if ther many good candidates, one is taken at random.
* An image's rating is downgraded in proportion to the number of times it has already been used. By default, its rating is downgrade one 2.3 times the number of usages.

To adjust the random component, use ``tolerance``, which sets the maximum random ratings bump in units of JND (the just-noticeable difference).

To specifically suppress repetition, you can increase the penalty for reuse. The parameter is ``usage_penalty``, and again its default value is 1, in the units of JND.

Example:

    pm.mosiac(tiles, tolerance=0.5, usage_penalty=3)

P.S. If you use multiscale tiles, the smaller tiles can repeat with impunity. The usage limit only applied to original tiles than their immediate children. There is a keyword argument for this as well, ``usage_impunity=2``, but unless you have a giant image pool, I wouldn't change it. 
