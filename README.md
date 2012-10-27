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
    pm.analyze(tiles) # Find color each tile.
    pm.matchmaker(tiles, 'imagepool.db') # Choose matching images and load them.
    mos = pm.mosaic(tiles)
    mos = pm.untune(mosaic, img, orig_img) # Optionally, transform the color palette back.
    mos.save('mosaic.jpg')

Remarks on each step:


* Generating an image pool is by far the longest step (about 45 minutes for 10,000 images) but it only has to be done once, and images can be added later without redoing the whole thing. Just run it again on a new folder or on the same folder with new images; it will skip duplicates.

* If the color scheme of your target image is not well represented in your potential tiles, shading and detail are lost. ``tune()`` ameliorates this problem by adjusting the levels of your target image to match the palette of colors available in the image pool. It's optional. The best solution is to have an image pool with all the necessary colors well represented. 

* Partitioning the image into tiles takes no time at all.

* Analyzing 900 (30x30) tiles takes about 20 seconds.

* Choosing the matching images for a 30x30 mosaic takes about 30 seconds. Once this is done, you can generating the mosaic very quickly, so it's easy to experiment with styles and settings. See Advanced Usage below.

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

A traditional photomosaic is a regular array of tiles. For a different effect, 
 
    tiles = pm.partition(img, (10, 10), depth=4)
 
    tiles = pm.partition(img, (10, 10), depth=4)

Tiles with high contrast will split. ``depth`` limits how small tile-splitting can go, but it does not control how many tiles will decide to split. ``hdr`` for "high dynamic range" sets a contrast threshold, a maximum allowed difference between the brightest and darkest pixel in any channel. (This is crude measure of contrast, but it's fast to compute.)

    tiles = pm.partition(img, (10, 10), depth=4, hdr=80) # many tiles
    tiles = pm.partition(img, (10, 10), depth=4, hdr=200) # or fewer tiles

Logs displayed by ``tiles()`` tell you how many tiles have been made, in total, after each generation. 2000-6000 is a reasonable range to aim for. You can go higher if you're willing to wait for ``analyze`` and ``matchmaker`` to run for more than 10 minutes together.

### Photomosaics with curved edges (masked images)

#### Simple cut-outs 

Create a black-and-white image the same size and your target image. White areas will be kept, and black areas will be masked. Open the image and pass it to ``partition``. Color images work too; they will just be converted.

    mask_img = pm.open('mask.jpg')
    tiles = pm.partition(img, (10, 10), mask=mask_img)

Tiles that fall wholly in the black area of the mask will be left blank. Of course, a curved edge is better traced by tiles that are small. When you invoke multiscale tiling along with the mask, small tiles will fill in along the edge. Any tiles that straddle the edge of the mask, containing some white and some non-white, are forced to subdivide. They subdivide down to the limit set by ``depth``, as explained above, so, for a smoother edge, increase depth.

    tiles = pm.partition(img, (10, 10), mask=mask_img, depth=4)

To examine the effect of the blank tiles before you proceed with ``analyze()`` and ``mosaic()``, assemble them.

    pm.assemble_tiles(tiles)

#### Masked image with "debris"

If the mask image contains grey, these areas can be filled with tiles probabilitically, creating the halo of debris that looks like a partly completed jigsaw puzzle.

    mask_img = pm.open('mask-with-some-grey-in-it.jpg')
    tiles = pm.partition(img, (10, 10), depth=4, mask=mask_img, debris=True)

To my eye, the effect is better with small tiles. You can limit debris to N-children of the original 10x10 tiles. Use the kwarg ``min_debris_depth``, which defaults to 1.

    tiles = pm.partition(img, (10, 10), depth=4, mask=mask_img, debris=True, min_debris_depth=2)

Again, examine the effect before proceeding.

    pm.assemble_tiles(tiles)

*Important*: If you use tune/untune, you should provide each with the mask, or the solid background will seriously distort the palette. The calls are:

    img = pm.tune(orig_img, img, mask)
    mos = pm.untune(mos, img, orig_img, mask)

Sometimes, fully untuing looks too harsh. You can hedge by setting ``amount`` to some value less than 1.

    mos = pm.untune(mos, img, orig_img, mask, amount=0.4)

### Tile matching and repetition

This is how tile images are chosen:
* Images are rated by their closeness to the target tile. Technical details: "Closeness" is Euclidean distance in Lab color space, which is a good proxy to perceived color difference. The distance is measured separately for each of the four quadrants of the tile and then averaged.
* Ratings are adjusted randomly by up to 2.3, which is the "just-noticeable difference" in color determined by [experiments](https://lirias.kuleuven.be/bitstream/123456789/71963/1/509.pdf). Thus, if one match is much better than the others, it will be chosen, but if there are many good candidates, one is taken at random.
* An image's rating is downgraded in proportion to the number of times it has already been used.

To adjust the random component, use ``tolerance``, which sets the maximum random ratings bump in units of JND, just-noticeable difference.

To specifically suppress repetition, increase the penalty for reuse. The parameter is ``usage_penalty``. Its default value is 1, in the units of JND.


### Scattered tiles

For a looser, even more scattered effect (imitating some works by [this artist](http://www.flickr.com/photos/tsevis/collections/)) you can tweak and size and location of the tiles.

#### Pad images to optimize lightness 

If a tile is shrunk in place, it reveals the background color behind it. Suppose our mosaic has a white background. Matches that are a little darker than the original image can be shrunk to affect lightness, crudely.

The ``pad`` feature shrinks images in proportion to this lightness discrepancy. For a white background (the default) set pad to a positive number around 1.

    mos = mosiac(tiles, pad=1)

To set the background to black and pad images that are too bright:

    mos = mosaic(tiles, background=(0, 0, 0), pad=-1)

To make padding more dramatic, set ``pad`` with a higher absolute value.

#### Scatter image placement

To place image randomly within a window of their location, turn on scattering a specify a margin in pixels.

    mos = mosaic(tiles, scatter=True, margin=10)

Tiles are "shuffled" before they are placed into the image, so the overlapping is nicely disordered, not top-to-bottom or left-to-right.

#### Pad & Scatter

Finally, if you turn on ``scatter`` and ``pad`` but leave ``margin`` to its default value of 0, each padded tile will be placed randomly off-center within its own padding, but it will not leave that box to overlap with other tiles. Unpadded tiles will not shift at all.
