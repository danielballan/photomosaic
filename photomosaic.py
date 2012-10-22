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
import sqlite3
import color_spaces as cs
from directory_walker import DirectoryWalker
from memo import memo

# Configure logger.
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

def simple(image_dir, target_filename, dimensions, output_file):
    pool(image_dir, 'temp.db')
    img = open(target_filename)
    img = tune(img, 'temp.db', quiet=True)
    tiles = partition(img, dimensions)
    analyze(tiles, 'temp.db')
    mos = photomosaic(tiles, 'temp.db')
    logger.info('Saving mosaic to %s', output_file)
    mos.save(output_file)

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
        logger.warning("I am quartering an image with odd dimensions.")
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

def connect(db_path):
    "Connect to, and if need be create, a sqlite database at db_path."
    try:
        db = sqlite3.connect(db_path)
    except IOError:
        print 'Cannot connect to SQLite database at %s' % db_path
        return
    db.row_factory = sqlite3.Row # Rows are dictionaries.
    return db

def create_tables(db):
    c = db.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS Images
                 (image_id INTEGER PRIMARY KEY,
                  usages INTEGER,
                  w INTEGER,
                  h INTEGER,
                  filename TEXT UNIQUE)""")
    c.execute("""CREATE TABLE IF NOT EXISTS Colors
                 (color_id INTEGER PRIMARY KEY,
                  image_id INTEGER,
                  region INTEGER,
                  L REAL,
                  a REAL,
                  b REAL,
                  red INTEGER,
                  green INTEGER,
                  blue INTEGER)""")
    c.close()
    db.commit()

def insert(filename, w, h, rgb, lab, db):
    """Insert image info in the Images table. Insert the dominant
    color of each of its regions in the Colors table."""
    c = db.cursor()
    try:
        c.execute("""INSERT INTO Images (usages, w, h, filename)
                     VALUES (?, ?, ?, ?)""",
                  (0, w, h, filename))
        image_id = c.lastrowid
        for region in xrange(len(rgb)):
            L, a, b = lab[region]
            red, green, blue= rgb[region]
            c.execute("""INSERT INTO Colors (image_id, region, 
                         L, a, b, red, green, blue) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                         (image_id, region, 
                         L, a, b, red, green, blue))
    except sqlite3.IntegrityError:
        print "Image %s is already in the table. Skipping it." % filename
    finally:
        c.close()
    
def pool(image_dir, db_name):
    """Analyze all the images in image_dir, and store the results in
    a sqlite database at db_name."""
    db = connect(db_name)
    try:
        create_tables(db)
        walker = DirectoryWalker(image_dir)
        for filename in walker:
            try:
                img = Image.open(filename)
            except IOError:
                print 'Cannot open %s as an image. Skipping it.' % filename
                continue
            if img.mode != 'RGB':
                print 'RGB images only. Skipping %s.' % filename
                continue
            w, h = img.size
            regions = split_quadrants(img)
            rgb = map(dominant_color, regions) 
            lab = map(cs.rgb2lab, rgb)
            # Really, a proper avg in Lab space would be best.
            insert(filename, w, h, rgb, lab, db)
        db.commit()
    finally:
        db.close()

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

def open(target_filename):
    "Just a wrapper for Image.open from PIL"
    try:
        return Image.open(target_filename)
    except IOError:
        print "Cannot open %s as an image." % target_filename
        return

def tune(target_img, db_name, dial=1, quiet=False):
    """Adjsut the levels of the image to match the colors available in the
    th pool. Return the adjusted image. Optionally plot some histograms."""
    db = connect(db_name)
    try:
        pool_hist = pool_histogram(db)
    finally:
        db.close()
    palette = compute_palette(pool_hist)
    adjusted_img = adjust_levels(target_img, palette, dial)
    if not quiet:
        # Use the Image.histogram() method to examine the target image
        # before and after the alteration.
        keys = 'red', 'green', 'blue'
        values = [channel.histogram() for channel in target_img.split()]
        totals = map(sum, values)
        norm = [map(lambda x: x/totals[i], val) for i, val in enumerate(values)]
        orig_hist = dict(zip(keys, norm)) 
        values = [channel.histogram() for channel in adjusted_img.split()]
        totals = map(sum, values)
        norm = [map(lambda x: x/totals[i], val) for i, val in enumerate(values)]
        adjusted_hist = dict(zip(keys, norm)) 
        plot_histograms(pool_hist, title='Images in the pool')
        plot_histograms(orig_hist, title='Unaltered target image')
        plot_histograms(adjusted_hist, title='Adjusted target image')
    return adjusted_img

def pool_histogram(db):
    """Generate a histogram of the images in the pool.
    Return a dictionary of the channels: L, a, b, red, green blue.
    Each dict entry contains a list of the frequencies correspond to the
    domain 0 - 255.""" 
    hist = {}
    c = db.cursor()
    try: 
        for ch in ['red', 'green', 'blue']:
            c.execute("""SELECT ROUND({ch}) as {ch}_, count(*)
                         FROM Colors 
                         GROUP BY ROUND({ch}_)""".format(ch=ch))
            values, counts = zip(*c.fetchall())
            # Normalize the histogram, and fill in 0 for missing entries.
            full_domain = range(0,256)
            N = sum(counts)
            all_counts = [1./N*counts[values.index(i)] if i in values else 0 \
                          for i in full_domain]
            hist[ch] = all_counts
    finally:
        c.close()
    return hist

def compute_palette(hist):
    """A palette maps a channel into the space of available colors, gleaned
    from a histogram of those colors."""
    # Integrate a histogram and round down.
    palette = {}
    for ch in ['red', 'green', 'blue']:
        integrals = np.cumsum(hist[ch])
        blocky_integrals = np.floor(256*integrals + 0.01).astype(int)
        p = []
        for i in range(256):
            p.append(np.where(blocky_integrals >= i)[0][0])
        palette[ch] = p
    return palette

def adjust_levels(target_img, palette, dial=1):
    "Transform the colors in the target_img according to palette."
    if dial < 0 or dial > 1:
        logger.error("dial must be between 0 and 1")
    keys = 'red', 'green', 'blue'
    p_func = dict(zip(keys, [lambda x: dial*palette[ch][x] for ch in keys]))
    channels = dict(zip(keys, target_img.split()))
    adjusted_channels = [Image.eval(channels[ch], p_func[ch]) for ch in keys]
    return Image.merge('RGB', adjusted_channels)

class Tile(object):
    """Tile wraps the Image class, so all methods that apply to images (show,
    save, crop, size, ...) apply to Tiles. Tiles also store contextual
    information that is used to reassembled them in the end."""
    def __init__(self, img, x, y, ancestry=None):
        self._img = img
        self.x = x
        self.y = y
        self.container_size = self._img.size
        self.ancestry = ancestry 

    def __getattr__(self, key):
        if key == '_img':
            raise AttributeError()
        return getattr(self._img, key)

    def substitute_img(self, img):
        self._img = img
        self.container_size = self._img.size

    def pos(self):
        return self.x, self.y 

    @property
    def tile_id(self):
        return self._tile_id

    @tile_id.setter
    def tile_id(self, value):
        self._tile_id = value

    @property
    def match(self):
        return self._match

    @match.setter
    def match(self, value):
        self._match = value # sqlite Row object


def partition(img, dimensions):
    "Partition the target image into a list of Tile objects."
    if isinstance(dimensions, int):
        dimensions = dimensions, dimensions
    width = img.size[0] // dimensions[0] 
    height = img.size[1] // dimensions[1]
    tiles = []
    for y in range(dimensions[1]):
        for x in range(dimensions[0]):
            tile_img = img.crop((x*width, 
                                y*height,
                                (x + 1)*width, 
                                (y + 1)*height))
            tiles.append(Tile(tile_img, x, y))
    return tiles

def create_target_table(db):
    c = db.cursor()
    try:
        c.execute("DROP TABLE IF EXISTS Target")
        c.execute("""CREATE TABLE Target
                     (tile_id INTEGER,
                      region INTEGER,
                      L REAL,
                      a REAL,
                      b REAL,
                      red INTEGER,
                      green INTEGER,
                      blue INTEGER,
                      PRIMARY KEY (tile_id, region))""")
    finally:
        c.close()
        db.commit()

def insert_target_tile(rgb, lab, db):
    """Insert the dominant color of each a tile's regions
    in the Target table. Identify each tile by x, y."""
    c = db.cursor()
    try:
        c.execute("""SELECT IFNULL(MAX(tile_id) + 1, 0) FROM Target""")
        tile_id, = c.fetchone()
        for region in xrange(len(rgb)):
            red, green, blue = rgb[region]
            L, a, b = lab[region]
            c.execute("""INSERT INTO Target (tile_id, region, 
                         L, a, b, red, green, blue)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                         (tile_id, region,
                         L, a, b, red, green, blue))
    finally:
        c.close()
    return tile_id
    
def analyze(tiles, db_name):
    """Determine dominant colors of target tiles, and insert them into
    the Target table of the db."""
    db = connect(db_name)
    try:
        create_target_table(db)
        pbar = progress_bar(len(tiles), "Analyzing images")
        for tile in tiles:
            regions = split_quadrants(tile)
            rgb = map(dominant_color, regions) 
            lab = map(cs.rgb2lab, rgb)
            tile_id = insert_target_tile(rgb, lab, db)
            # tile_id is a number assigned by the db
            tile.tile_id = tile_id
            pbar.next()
        logger.info("Performing big join (no progress bar)")
        join(db)
        db.commit()
        logger.info("Complete.")
    finally:
        db.close()

def join(db):
    """Compare every target tile to every image by joining
    the Colors table to the Target table. Specifically, compute their
    Euclidean color distance E."""
    c = db.cursor()
    try:
        c.execute("DROP TABLE IF EXISTS BigJoin")
        start_time = time.clock()
        # Technically, compute E^2, not E, because sqlite does not support
        # sqrt(). Also compute the difference in lightness, dL.
        c.execute("""CREATE TABLE BigJoin AS
                     SELECT
                     tile_id, image_id, 
                     avg((c.L - t.L)*(c.L - t.L)
                     + (c.a - t.a)*(c.a - t.a)
                     + (c.b - t.b)*(c.b - t.b)) as Esq,
                     avg(c.L - t.L) as dL
                     FROM Colors c
                     JOIN Target t USING (region)
                     GROUP BY tile_id, image_id""")
        print "Join completed in {}".format(time.clock() - start_time)
    finally:
        c.close()
    db.commit()

def choose_match(tile_id, db, randomize=False):
    """Average perceived color difference E and lightness difference dL
    over the regions of each possible match. Rank them in E, and take
    the best image for each target tile. Allow duplicates."""
    if not randomize:
        query = """SELECT 
                   image_id,
                   Esq,
                   dL,
                   filename
                   FROM BigJoin
                   JOIN Images using (image_id)
                   WHERE tile_id=? 
                   ORDER BY Esq ASC
                   LIMIT 1"""
    elif not isinstance(randomize, int):
        logger.error("randomzie must be an integer or False")
        return
    else:
        query = """SELECT * FROM (SELECT 
                   image_id,
                   Esq,
                   dL,
                   filename
                   FROM BigJoin
                   JOIN Images using (image_id)
                   WHERE tile_id=?
                   ORDER BY Esq ASC
                   LIMIT {N}) ORDER BY RANDOM() LIMIT 1""".format(N=randomize)
    c = db.cursor()
    try:
        c.execute(query, (tile_id,))
        match = c.fetchone()
    finally:
        c.close()
    return match

def crop_to_fit(img, tile_size):
    "Return a copy of img cropped to precisely fill the dimesions tile_size."
    img_w, img_h = img.size
    tile_w, tile_h = tile_size
    img_aspect = int(round(img_w/img_h))
    tile_aspect = int(round(tile_w/tile_h))
    if img_aspect > tile_aspect:
        # It's too wide.
        crop_h = img_h
        crop_w = crop_h*tile_aspect
        x_offset = (img_w - crop_w) // 2
        y_offset = 0
    else:
        # It's too tall.
        crop_w = img_w
        crop_h = crop_w // tile_aspect
        x_offset = 0
        y_offset = (img_h - crop_h) // 2
    img = img.crop((x_offset,
                    y_offset,
                    x_offset + crop_w,
                    y_offset + crop_h))
    img = img.resize((tile_w, tile_h), Image.ANTIALIAS)
    return img

def shrink_to_brighten(img, tile_size, dL):
    """Return an image smaller than a tile. Its white margins
    will effect lightness. Also, varied tile size looks nice.
    The greater the greater the lightness discrepancy dL
    the smaller the tile is shrunk."""
    MAX_dL = 100 # the largest possible distance in Lab space
    MIN = 0.5 # not so close small that it's a speck
    MAX = 0.9 # not so close to unity that is looks accidental
    assert dL < 0, "Only shrink image when tile is too dark."
    scaling = MAX - (MAX - MIN)*(-dL)/MAX_dL
    shrunk_size = [int(scaling*dim) for dim in tile_size]
    img = crop_to_fit(img, shrunk_size) 
    return img 

def tile_position(tile, random_margins=False):
    """Return the x, y position of the tile in the mosaic, according for
    possible margins and optional random nudges for a 'scattered' look.""" 
    pos = tile.x*tile.container_size[0], tile.y*tile.container_size[1]
    return pos

@memo
def prepare_tile(filename, size, dL=None):
    """This memoized function only executes once for a given set of args.
    Hence, multiple (same-sized) tiles of the same image are speedy."""
    new_img = Image.open(filename)
    if (dL is None or dL >= 0):
        # Either we are not shrinking tiles (dL = None) or
        # the match is brighter than the target. Leave it alone.
        new_img = crop_to_fit(new_img, size)
    else:
        # Match is darker than target.
        # Shrink it to leave white padding.
        new_img = shrink_to_brighten(new_img, size, dL)
    return new_img

def photomosaic(tiles, db_name, vary_size=False, randomize=5, 
                random_margins=False):
    """Take the tiles from target() and return a mosaic image."""
    db = connect(db_name)
    try:
        pbar = progress_bar(len(tiles), "Choosing matching tiles")
        for tile in tiles:
            tile.match = choose_match(tile.tile_id, db)
            pbar.next()
        pbar = progress_bar(len(tiles), "Scaling tiles")
        for tile in tiles:
            dL = tile.match['dL'] if vary_size else None
            new_img = prepare_tile(tile.match['filename'], tile.size, dL)
            tile.substitute_img(new_img)
            pbar.next()
    finally:
        db.close()
    pbar = progress_bar(len(tiles), "Building mosaic")
    background = (255, 255, 255)
    # Infer dimensions so they don't have to be passed in the function call.
    dimensions = map(max, zip(*[(1 + tile.x, 1 + tile.y) for tile in tiles]))
    mosaic_size = tiles[0].size[0]*dimensions[0], tiles[0].size[1]*dimensions[1]
    mosaic = Image.new('RGB', mosaic_size, background)
    for tile in tiles:
        pos = tile_position(tile, random_margins)
        mosaic.paste(tile, pos)
        pbar.next()
    return mosaic

def color_hex(rgb):
    "Convert [r, g, b] to a HEX value with a leading # character."
    return '#' + ''.join(chr(c) for c in rgb).encode('hex')

def progress_bar(total_steps, message='', notifications=8):
    step = 0
    logger.info('%s...', message)
    notifications = min(total_steps, notifications)
    while step < total_steps - 1:
        if step % (total_steps // notifications) == 0:
            logger.info('%s/%s', step, total_steps)
        yield
        step += 1
    logger.info('Complete.')
    yield
