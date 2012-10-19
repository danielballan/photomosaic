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

# Configure logger.
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

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
    db.row_factory = sqlite3.Row
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
            L, a, b= lab[region]
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

def partition_target(img, tile_size):
    "Partition the target image into a 2D list of Images."
    # TODO: Allow to tiles are different sizes. 
    # Merge neighbors that are similar
    # or that inhabit regions of long spatial wavelength.
    width = img.size[0] // tile_size[0]
    height = img.size[1] // tile_size[1]
    tiles = [[None for w in range(width)] for h in range(height)]
    for y in range(height):
        for x in range(width):
            tile = img.crop((x*tile_size[0], 
                             y*tile_size[1],
                             (x + 1)*tile_size[0], 
                             (y + 1)*tile_size[1]))
            tiles[y][x] = tile
    return tiles

def create_target_table(db):
    c = db.cursor()
    try:
        c.execute("DROP TABLE IF EXISTS Target")
        c.execute("""CREATE TABLE Target
                     (tile_id INTEGER PRIMARY KEY,
                      x INTEGER,
                      y INTEGER,
                      region INTEGER,
                      L REAL,
                      a REAL,
                      b REAL,
                      red INTEGER,
                      green INTEGER,
                      blue INTEGER)""")
    finally:
        c.close()
        db.commit()

def insert_target_tile(x, y, rgb, lab, db):
    """Insert the dominant color of each a tile's regions
    in the Target table. Identify each tile by x, y."""
    c = db.cursor()
    try:
        for region in xrange(len(rgb)):
            red, green, blue= rgb[region]
            L, a, b= lab[region]
            c.execute("""INSERT INTO Target (x, y, region, 
                         L, a, b, red, green, blue)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                         (x, y, region,
                         L, a, b, red, green, blue))
    finally:
        c.close()
    
def target(target_filename, tile_size, db_name):
    """Open the target image, partition it into tiles, analyze them,
    store the results in the db, and return a 2D list of the tiles."""
    try:
        target_img = Image.open(target_filename)
    except IOError:
        print "Cannot open %s as an image." % target_filename
        return 1
    if isinstance(tile_size, int):
        tile_size = tile_size, tile_size
    tiles = partition_target(target_img, tile_size)
    db = connect(db_name)
    try:
        create_target_table(db)
        print 'Analyzing target image...'
        for x, row in enumerate(tiles):
            for y, tile in enumerate(row):
                regions = split_quadrants(tile)
                rgb = map(dominant_color, regions) 
                lab = map(cs.rgb2lab, rgb)
                # Really, a proper avg in Lab space would be best.
                insert_target_tile(x, y, rgb, lab, db)
        print 'Performing big join...'
        join(db)
        db.commit()
    finally:
        db.close()
    return tiles

def histogram(db, table_name):
    """Generate a histogram of the images in the pool, histogram(db, 'Colors'),
    or the tiles in the target, histogram(db, 'Target').
    Return a dictionary of the channels: L, a, b, red, green blue.
    Each dict entry contains a pair of tuples: [(values), (counts)]."""
    if table_name not in ['Colors', 'Target']:
        logger.error("Invalid table_name.")
        return
    hist = {}
    c = db.cursor()
    try: 
        for channel in ['L', 'a', 'b', 'red', 'green', 'blue']:
            c.execute("""SELECT ROUND({channel}) as {channel}_, count(*)
                         FROM {table}
                         GROUP BY ROUND({channel}_)""".format(
                         table=table_name, channel=channel))
            values, counts = zip(*c.fetchall())
            # Normalize the histogram, and fill in 0 for missing entries.
            if channel in ['red', 'green', 'blue']:
                full_domain = range(0,256)
            elif channel in ['a', 'b']:
                full_domain = range(-100,101)
            else:
                full_domain = range(0,101)
            N = sum(counts)
            all_counts = [1./N*counts[values.index(i)] if i in values else 0 \
                          for i in full_domain]
            hist[channel] = [full_domain, all_counts]
    finally:
        c.close()
    return hist

def join(db):
    """Compare every target tile to every image by joining
    the Colors table to the Target table."""
    query = """INSERT INTO BigJoin (x, y, image_id, Esq, dL)
               SELECT
               x, y,
               image_id, 
               avg((c.L - t.L)*(c.L - t.L)
               + (c.a - t.a)*(c.a - t.a)
               + (c.b - t.b)*(c.b - t.b)) as Esq,
               avg(c.L - t.L) as dL
               FROM Colors c
               JOIN Target t USING (region)
               GROUP BY x, y, image_id"""
    c = db.cursor()
    try:
        c.execute("DROP TABLE IF EXISTS BigJoin")
        c.execute("""CREATE TABLE BigJoin
                     (id INTEGER PRIMARY KEY,
                      x INTEGER,
                      y INTEGER,
                      image_id INTEGER,
                      Esq REAL,
                      dL REAL)""")
        start_time = time.clock()
        c.execute(query)
        print "Join completed in {}".format(time.clock() - start_time)
    finally:
        c.close()
    db.commit()

def matching(x, y, db):
    """Average perceived color difference E and lightness difference dL
    over the regions of each possible match. Rank them in E, and take
    the best image for each target tile. Allow duplicates."""
    query = """SELECT 
               image_id,
               Esq,
               dL,
               filename
               FROM BigJoin
               JOIN Images using (image_id)
               WHERE x=? AND y=?
               ORDER BY Esq ASC
               LIMIT 1"""
    c = db.cursor()
    try:
        c.execute(query, (x, y))
        match = c.fetchone()
    finally:
        c.close()
    return match

def assemble_mosaic(tiles, tile_size, background=(255, 255, 255),
                    random_margins=False):
    "Build the final image."
    # Technically, tile_size could be inferred from a tile,
    # but let's not trust it in this case.
    size = len(tiles[0])*tile_size[0], len(tiles)*tile_size[1]
    mosaic = Image.new('RGB', size, background)
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            pos = tile_position(x, y, tile.size, tile_size, random_margins)
            mosaic.paste(tile, pos)
    return mosaic # suitable to be saved with imsave

def tile_position(x, y, this_size, generic_size,
                  random_margins=False):
    """Return the x, y position of the tile in the mosaic, according for
    possible margins and optional random nudges for a 'scattered' look.""" 
    if this_size == generic_size: 
        pos = x*generic_size[0], y*generic_size[1]
    else:
        margin = ((generic_size[0] - this_size[0]) // 2, 
                  (generic_size[1] - this_size[1]) // 2)
        if randomize:
            try:
                # Set left and bottom margins to a random value
                # bound by 0 and twice their original value.
                margin = [random.randrange(2*m) for m in margin]
            except ValueError:
                pass
        pos = x*generic_size[0] + margin[0], y*generic_size[1] + margin[1]
    return pos

def photomosaic(tiles, db_name, vary_size=False, random_margins=False):
    """Take the tiles from target() and return a mosaic image."""
    tile_size = tiles[0][0].size # assuming uniform
    db = connect(db_name)
    try:
        print 'Choosing matching tiles and scaling them...'
        for x, row in enumerate(tiles):
            for y, tile in enumerate(row):
                # Replace target tile with a matched tile.
                match = matching(x, y, db)
                new_tile = make_tile(match, tile_size, vary_size)
                tiles[x][y] = new_tile
    finally:
        db.close()
    print 'Building mosaic...'
    mosaic = assemble_mosaic(tiles, tile_size, random_margins)
    return mosaic

def make_tile(match, tile_size, vary_size=False):
    "Open and resize the matched image."
    raw = Image.open(match['filename'])
    if (match['dL'] >= 0 or not vary_size):
        # Match is brighter than target.
        img = crop_to_fit(raw, tile_size)
    else:
        # Match is darker than target.
        # Shrink it to leave white padding.
        img = shrink_to_brighten(raw, tile_size, match['dL'])
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

def crop_to_fit(img, tile_size):
    "Return a copy of img cropped to precisely fill the dimesions tile_size."
    img_w, img_h = img.size
    tile_w, tile_h = tile_size
    img_aspect = img_w // img_h
    tile_aspect = tile_w // tile_h
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

def print_db(db):
    "Dump the database to the screen, for debugging."
    c = db.cursor()
    c.execute("SELECT * FROM Images")
    for row in c:
        print row 
    c.execute("SELECT * FROM Colors")
    for row in c:
        print row
    c.close()
    
def color_hex(rgb):
    "Convert [r, g, b] to a HEX value with a leading # character."
    return '#' + ''.join(chr(c) for c in rgb).encode('hex')

def reset_usage(db):
    "Before using the image pool, reset image usage count to 0."
    try:
        c = db.cursor()
        c.execute("UPDATE Images SET usages=0")
        c.close()
        db.commit()
    except sqlite3.OperationalError, e:
        print e
