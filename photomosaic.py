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
import scipy
import scipy.misc
from scipy.cluster import vq
from scipy import interpolate
import Image
import ImageFilter
import color_spaces as cs
from progress_bar import progress_bar
from sql_image_pool import SqlImagePool
from tile import Tile

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

def split_regions(img, split_dim):
    """Split an image into subregions.
    Use split_dim=2 or (2,2) or (2,3) etc.
    Return a flat list of images."""
    if isinstance(split_dim, int):
        rows = columns = split_dim
    else:
        columns, rows = split_dim
    r_size = img.size[0] // columns, img.size[1] // rows
    # regions = [[None for c in range(columns)] for r in range(rows)]
    regions = columns*rows*[None]
    for y in range(rows):
        for x in range(columns):
            region = img.crop((x*r_size[0], 
                             y*r_size[1],
                             (x + 1)*r_size[0], 
                             (y + 1)*r_size[1]))
            # regions[y][x] = region ## for nested output
            regions[y*columns + x] = region
    return regions
    
def split_quadrants(img):
    """Convenience function: calls split_regions(img, 2). Returns
    a flat 4-element list: top-left, top-right, bottom-left, bottom-right."""
    if img.size[0] & 1 or img.size[1] & 1:
        logger.debug("I am quartering an image with odd dimensions.")
    return split_regions(img, 2)

def dominant_color(img, clusters=5, size=50):
    """Group the colors in an image into like clusters, and return
    the central value of the largest cluster -- the dominant color."""
    assert img.mode == 'RGB', 'RGB images only!'
    img.thumbnail((size, size))
    imgarr = scipy.misc.fromimage(img)
    imgarr = imgarr.reshape(scipy.product(imgarr.shape[:2]), imgarr.shape[2])
    colors, dist = vq.kmeans(imgarr, clusters)
    vecs, dist = vq.vq(imgarr, colors)
    counts, bins = scipy.histogram(vecs, len(colors))
    dominant_color = colors[counts.argmax()]
    return map(int, dominant_color) # Avoid returning np.uint8 type.

def open(target_filename):
    "Just a wrapper for Image.open from PIL"
    try:
        return Image.open(target_filename)
    except IOError:
        logger.warning("Cannot open %s as an image.", target_filename)
        return

def plot_histograms(hist, title=''):
    "Plot an RGB histogram given as a dictionary with channel keys."
    import matplotlib.pyplot as plt
    fig, (red, green, blue) = plt.subplots(3, sharex=True, sharey=True)
    domain = range(0, 256)
    red.fill_between(domain, hist['red'],
                     facecolor='red')
    green.fill_between(domain, 0, hist['green'],
                       facecolor='green')
    blue.fill_between(domain, 0, hist['blue'],
                      facecolor='blue')
    red.set_xlim(0,256)
    red.set_ylim(ymin=0)
    red.set_title(title)
    fig.show()

def img_histogram(img, mask=None):
    keys = 'red', 'green', 'blue'
    channels = dict(zip(keys, img.split()))
    hist= {}
    for ch in keys:
        if mask:
            h = channels[ch].histogram(mask.convert("1"))
        else:
            h = channels[ch].histogram()
        normalized_h = [256./sum(h)*v for v in h]
        hist[ch] = normalized_h
    return hist

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

def compute_palette(hist):
    """A palette maps a channel into the space of available colors, gleaned
    from a histogram of those colors."""
    # Integrate a histogram and round down.
    palette = {}
    for ch in ['red', 'green', 'blue']:
        integrals = np.cumsum(hist[ch])
        blocky_integrals = np.floor(integrals + 0.01).astype(int)
        bars = np.ediff1d(blocky_integrals,to_begin=blocky_integrals[0])
        p = [[color]*freq for color, freq in enumerate(bars.tolist())]
        p = [c for sublist in p for c in sublist]
        assert len(p) == 256, "Palette should have 256 entries."
        palette[ch] = p
    return palette

def adjust_levels(target_img, from_palette, to_palette):
    """Transform the colors of an image to match the color palette of
    another image."""
    keys = 'red', 'green', 'blue'
    channels = dict(zip(keys, target_img.split()))
    f, g = from_palette, to_palette # compact notation
    func = {} # function to transform color at each pixel
    for ch in keys:
        def j(x):
           while True:
               try:
                   inv_f = f[ch].index(x)
                   break
               except ValueError:
                   if x < 255:
                       x += 1
                       continue 
                   else:
                       inv_f = 255
                       break
           return to_palette[ch][inv_f]
        func[ch] = j 
    adjusted_channels = [Image.eval(channels[ch], func[ch]) for ch in keys]
    return Image.merge('RGB', adjusted_channels)

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
