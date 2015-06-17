# Copyright 2012 Daniel B. Allan
# dallan@pha.jhu.edu, daniel.b.allan@gmail.com
# http://pha.jhu.edu/~dallan
# http://www.danallan.com
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses>.

from __future__ import division
import os
import logging
import time
import random
import numpy as np
#from scipy import interpolate
import Image
import ImageFilter
import color_spaces as cs
from progress_bar import progress_bar
from sql_image_pool import SqlImagePool
from tile import Tile
from image_functions import *

# Configure logger.
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

def simple(image_dir, target_filename, dimensions, output_file):
    "A convenient wrapper for producing a traditional photomosaic."
    pool = SqlImagePool('temp.db')
    pool.add_directory(image_dir)
    orig_img = open(target_filename)
    
    img = tune(orig_img, pool, quiet=True)
    tiles = partition(img, dimensions)
    analyze(tiles)
    matchmaker(tiles, pool)
    mos = mosaic(tiles)
    mos = untune(mos, img, orig_img)
    logger.info('Saving mosaic to %s', output_file)
    mos.save(output_file)
    pool.close()

def untune(mos, img, orig_img, mask=None, amount=1):
    if mask:
        m = crop_to_fit(mask, img.size)
        orig_palette = compute_palette(img_histogram(orig_img, m))
        img_palette = compute_palette(img_histogram(img, m))
    else:
        orig_palette = compute_palette(img_histogram(orig_img))
        img_palette = compute_palette(img_histogram(img))
    return Image.blend(mos, adjust_levels(mos, img_palette, orig_palette),
                          amount)

def tune(target_img, pool, mask=None, quiet=True):
    """Adjust the levels of the image to match the colors available in the
    the pool. Return the adjusted image. Optionally plot some histograms."""
    if len(pool)==0:
        return target_img
    pool_hist = pool.pool_histogram()
    pool_palette = compute_palette(pool_hist)
    if mask:
        m = crop_to_fit(mask, target_img.size)
        target_palette = compute_palette(img_histogram(target_img, m))
    else:
        target_palette = compute_palette(img_histogram(target_img))
    adjusted_img = adjust_levels(target_img, target_palette, pool_palette)
    if not quiet:
        # Use the Image.histogram() method to examine the target image
        # before and after the alteration.
        keys = 'red', 'green', 'blue'
        values = [channel.histogram() for channel in target_img.split()]
        totals = map(sum, values)
        norm = [map(lambda x: 256*x/totals[i], val) \
                for i, val in enumerate(values)]
        orig_hist = dict(zip(keys, norm)) 
        values = [channel.histogram() for channel in adjusted_img.split()]
        totals = map(sum, values)
        norm = [map(lambda x: 256*x/totals[i], val) \
                for i, val in enumerate(values)]
        adjusted_hist = dict(zip(keys, norm)) 
        plot_histograms(pool_hist, title='Images in the pool')
        plot_histograms(orig_hist, title='Unaltered target image')
        plot_histograms(adjusted_hist, title='Adjusted target image')
    return adjusted_img

def partition(img, dimensions, mask=None, depth=0, hdr=80,
              debris=False, min_debris_depth=1, base_width=None):
    "Partition the target image into a list of Tile objects."
    if isinstance(dimensions, int):
        dimensions = dimensions, dimensions
    if base_width is not None:
        cwidth = img.size[0] / dimensions[0]
        width = base_width * dimensions[0]
        factor = base_width / cwidth
        height = int(img.size[1] * factor)
        print img.size, dimensions, width, height
        img = crop_to_fit(img, (width, height))
    # img.size must have dimensions*2**depth as a factor.
    factor = dimensions[0]*2**depth, dimensions[1]*2**depth
    new_size = tuple([int(factor[i]*np.ceil(img.size[i]/factor[i])) \
                      for i in [0, 1]])
    logger.info("Resizing image to %s, a round number for partitioning. "
                "If necessary, I will crop to fit.",
                new_size)
    img = crop_to_fit(img, new_size)
    if mask:
        mask = crop_to_fit(mask, new_size)
        if not debris:
            mask = mask.convert("1") # no gray
    width = img.size[0] // dimensions[0] 
    height = img.size[1] // dimensions[1]
    tiles = []
    for y in range(dimensions[1]):
        for x in range(dimensions[0]):
            tile_img = img.crop((x*width, y*height,
                                (x + 1)*width, (y + 1)*height))
            if mask:
                mask_img = mask.crop((x*width, y*height,
                                     (x + 1)*width, (y + 1)*height))
            else:
                mask_img = None
            tile = Tile(tile_img, x, y, mask=mask_img)
            tiles.append(tile)
    for g in xrange(depth):
        old_tiles = tiles
        tiles = []
        for tile in old_tiles:
            if tile.dynamic_range() > hdr or tile.straddles_mask_edge():
                # Keep children; discard parent.
                tiles += tile.procreate()
            else:
                # Keep tile -- no children.
                tiles.append(tile)
        logging.info("There are %d tiles in generation %d",
                     len(tiles), g)
    # Now that all tiles have been made and subdivided, decide which are blank.
    [tile.determine_blankness(min_debris_depth) for tile in tiles]
    logger.info("%d tiles are set to be blank",
                len([1 for tile in tiles if tile.blank]))
    return tiles

def analyze(tiles):
    """Determine dominant colors of target tiles, and save that information
    in the Tile object."""
    pbar = progress_bar(len(tiles), "Analyzing images")
    for tile in tiles:
        analyze_one(tile)
        pbar.next()

def analyze_one(tile):
    """"Determine dominant colors of target tiles, and save that information
    in the Tile object."""
    if tile.blank:
        return
    regions = split_quadrants(tile)
    tile.rgb = map(dominant_color, regions) 
    tile.lab = map(cs.rgb2lab, tile.rgb)

def crop_to_fit(img, tile_size):
    "Return a copy of img cropped to precisely fill the dimesions tile_size."
    img_w, img_h = img.size
    tile_w, tile_h = tile_size
    img_aspect = img_w/img_h
    tile_aspect = tile_w/tile_h
    if img_aspect > tile_aspect:
        # It's too wide.
        crop_h = img_h
        crop_w = int(round(crop_h*tile_aspect))
        x_offset = int((img_w - crop_w)/2)
        y_offset = 0
    else:
        # It's too tall.
        crop_w = img_w
        crop_h = int(round(crop_w/tile_aspect))
        x_offset = 0
        y_offset = int((img_h - crop_h)/2)
    img = img.crop((x_offset,
                    y_offset,
                    x_offset + crop_w,
                    y_offset + crop_h))
    img = img.resize((tile_w, tile_h), Image.ANTIALIAS)
    return img

def shrink_by_lightness(pad, tile_size, dL):
    """The greater the greater the lightness discrepancy dL
    the smaller the tile will shrunk."""
    sgn = lambda x: (x > 0) - (x < 0)
    if sgn(pad)*dL < 0:
        return tile_size
    MAX_dL = 100 # the largest possible distance in Lab space
    MIN = 0.5 # not so close small that it's a speck
    MAX = 0.95 # not so close to unity that is looks accidental
    scaling = MAX - (MAX - MIN)*(-pad*dL)/MAX_dL
    shrunk_size = [int(scaling*dim) for dim in tile_size]
    return shrunk_size

def tile_position(tile, size, scatter=False, margin=0):
    """Return the x, y position of the tile in the mosaic, according for
    possible margins and optional random nudges for a 'scattered' look.""" 
    # Sum position of original ancestor tile, relative position of this tile's
    # container, and any margins that this tile has.
    ancestor_pos = [tile.x*tile.ancestor_size[0], tile.y*tile.ancestor_size[1]]
    if tile.depth == 0:
        rel_pos = [[0, 0]]
    else:
        x_size, y_size = tile.ancestor_size
        rel_pos = [[x*x_size//2**(gen + 1), y*y_size//2**(gen + 1)] \
                           for gen, (x, y) in enumerate(tile.ancestry)]
        
    if tile.size == size:
        padding = [0, 0]
    else:
        padding = map(lambda (x, y): (x - y)//2, zip(*([size, tile.size])))
    if scatter:
        padding = [random.randint(0, 1 + margin), random.randint(0, 1 + margin)]
    pos = tuple(map(sum, zip(*([ancestor_pos] + rel_pos + [padding]))))
    return pos


def matchmaker(tiles, pool, tolerance=1, usage_penalty=1, usage_impunity=2):
    """Assign each tile a new image, and open that image in the Tile object."""
    if len(pool)==0:
        logger.error('No images in pool to match!')
        exit(-1)
    pool.reset_usage()
    pbar = progress_bar(len(tiles), "Choosing and loading matching images")
    for tile in tiles:
        if tile.blank:
            pbar.next()
            continue
        tile.match = pool.choose_match(tile.lab, tolerance,
            usage_penalty if tile.depth < usage_impunity else 0)
        pbar.next()

def mosaic(tiles, pad=False, scatter=False, margin=0, scaled_margin=False,
           background=(255, 255, 255)):
    """Return the mosaic image.""" 
    # Infer dimensions so they don't have to be passed in the function call.
    dimensions = map(max, zip(*[(1 + tile.x, 1 + tile.y) for tile in tiles]))
    mosaic_size = map(lambda (x, y): x*y,
                         zip(*[tiles[0].ancestor_size, dimensions]))
    mos = Image.new('RGB', mosaic_size, background)
    pbar = progress_bar(len(tiles), "Scaling and placing tiles")
    random.shuffle(tiles)
    for tile in tiles:
        if tile.blank:
            pbar.next()
            continue
        if pad:
            size = shrink_by_lightness(pad, tile.size, tile.match['dL'])
            if margin == 0:
                margin = min(tile.size[0] - size[0], tile.size[1] - size[1])
        else:
            size = tile.size
        if scaled_margin:
            pos = tile_position(tile, size, scatter, margin//(1 + tile.depth))
        else:
            pos = tile_position(tile, size, scatter, margin)
        mos.paste(crop_to_fit(tile.match_img, size), pos)
        pbar.next()
    return mos

def assemble_tiles(tiles, margin=1):
    """This is not used to build the final mosaic. It's a handy function for
    assembling new tiles (without blanks) to see how partitioning looks."""
    # Infer dimensions so they don't have to be passed in the function call.
    dimensions = map(max, zip(*[(1 + tile.x, 1 + tile.y) for tile in tiles]))
    mosaic_size = map(lambda (x, y): x*y,
                         zip(*[tiles[0].ancestor_size, dimensions]))
    background = (255, 255, 255)
    mos = Image.new('RGB', mosaic_size, background)
    for tile in tiles:
        if tile.blank:
            continue
        shrunk = tile.size[0]-2*int(margin), tile.size[1]-2*int(margin)
        pos = tile_position(tile, shrunk, False, 0)
        mos.paste(tile.resize(shrunk), pos)
    return mos

def color_hex(rgb):
    "Convert [r, g, b] to a HEX value with a leading # character."
    return '#' + ''.join(chr(c) for c in rgb).encode('hex')

def testing():
    pm.simple('images/samples', 'images/samples/dan-allan.jpg', (10,10), 'output.jpg')
