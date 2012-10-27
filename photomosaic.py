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
from progress_bar import progress_bar

# Configure logger.
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

def simple(image_dir, target_filename, dimensions, output_file):
    "A convenient wrapper for producing a traditional photomosaic."
    pool(image_dir, 'temp.db')
    orig_img = open(target_filename)
    img = tune(orig_img, 'temp.db', quiet=True)
    tiles = partition(img, dimensions)
    analyze(tiles)
    matchmaker(tiles, 'temp.db')
    mos = mosaic(tiles)
    mos = untune(mos, img, orig_img)
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

def connect(db_path):
    "Connect to a sqlite database at db_path. If it does not exist, create it."
    try:
        db = sqlite3.connect(db_path)
    except IOError:
        logger.error("Cannot connect to SQLite database at %s",  db_path)
        return
    db.row_factory = sqlite3.Row # Rows are dictionaries.
    return db

def create_tables(db):
    """Create Images for image meta info, Color for RGB values and LabColor
    for LAB values. RGB and LAB are used for different steps, RGB for levels
    adjustments are LAB for measure perceived color difference precisely.
    Thus Color and LabColor are organized somewhat differently."""
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
                  red INTEGER,
                  green INTEGER,
                  blue INTEGER)""")
    c.execute("""CREATE TABLE IF NOT EXISTS LabColors
                 (labcolor_id INTEGER PRIMARY KEY,
                  image_id INTEGER,
                  region INTEGER,
                  L1 REAL,
                  a1 REAL,
                  b1 REAL,
                  L2 REAL,
                  a2 REAL,
                  b2 REAL,
                  L3 REAL,
                  a3 REAL,
                  b3 REAL,
                  L4 REAL,
                  a4 REAL,
                  b4 REAL)""")
    c.close()
    db.commit()

def in_db(filename, db):
    c = db.cursor()
    try: 
        c.execute("SELECT count(*) FROM Images WHERE filename=?", (filename,))
        return c.fetchone()[0] > 0
    finally:
        c.close()
    return False

def get_size(db):
    c = db.cursor()
    try: 
        c.execute("SELECT count(*) FROM Images")
        return c.fetchone()[0] 
    finally:
        c.close()
    return 0   

def insert(filename, w, h, rgb, lab, db):
    """Insert image info in the Images table and color information in the
    Color and LabColor tables."""
    c = db.cursor()
    try:
        c.execute("""INSERT INTO Images (usages, w, h, filename)
                     VALUES (?, ?, ?, ?)""",
                  (0, w, h, filename))
        image_id = c.lastrowid
        c.executemany("""INSERT INTO Colors (image_id, region, red, green, blue)
                         VALUES (?, ?, ?, ?, ?)""",
                         [tuple([image_id, region] + list(colors)) \
                          for region, colors in enumerate(rgb)])
        c.execute("""INSERT INTO LabColors (image_id,
                     L1, a1, b1, L2, a2, b2, L3, a3, b3, L4, a4, b4)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     tuple([image_id] + [ch for reg in lab for ch in reg]))
    except sqlite3.IntegrityError:
        logger.warning("Image %s is already in the table. Skipping it.",
                       filename)
    except:
        logger.warning("Unknown problem with image %s. Skipping it.",
                       filename)
    finally:
        c.close()
    
def pool(image_dir, db_name):
    """Analyze all the images in image_dir, and store the results in
    a sqlite database at db_name."""
    db = connect(db_name)
    try:
        create_tables(db)
        walker = DirectoryWalker(image_dir)
        file_count = len(list(walker)) # stupid but needed but progress bar
        pbar = progress_bar(file_count, "Analyzing images and building db")
        walker = DirectoryWalker(image_dir)
        for filename in walker:
            if in_db(filename, db):
                logger.warning("Image %s is already in the table. Skipping it."%filename)
                pbar.next()
                continue
            try:
                img = Image.open(filename)
            except IOError:
                logger.warning("Cannot open %s as an image. Skipping it.",
                               filename)
                pbar.next()
                continue
            if img.mode != 'RGB':
                logger.warning("RGB images only. Skipping %s.", filename)
                pbar.next()
                continue
            w, h = img.size
            try:
                regions = split_quadrants(img)
                rgb = map(dominant_color, regions) 
                lab = map(cs.rgb2lab, rgb)
            except:
                logger.warning("Unknown problem analyzing %s. Skipping it.",
                               filename)
                continue
            insert(filename, w, h, rgb, lab, db)
            pbar.next()
        db.commit()
        logger.info('Collection %s built with %d images'%(db_name, get_size(db)))
    finally:
        db.close()

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

def tune(target_img, db_name, mask=None, quiet=True):
    """Adjsut the levels of the image to match the colors available in the
    th pool. Return the adjusted image. Optionally plot some histograms."""
    db = connect(db_name)
    try:
        pool_hist = pool_histogram(db)
    finally:
        db.close()
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

def pool_histogram(db):
    """Generate a histogram of the images in the pool.
    Return a dictionary of the channels red, green blue.
    Each dict entry contains a list of the frequencies correspond to the
    domain 0 - 255.""" 
    hist = {}
    c = db.cursor()
    try: 
        for ch in ['red', 'green', 'blue']:
            c.execute("""SELECT {ch}, count(*)
                         FROM Colors 
                         GROUP BY {ch}""".format(ch=ch))
            values, counts = zip(*c.fetchall())
            # Normalize the histogram to 256 for readability,
            # and fill in 0 for missing entries.
            full_domain = range(0,256)
            N = sum(counts)
            all_counts = [256./N*counts[values.index(i)] if i in values else 0 \
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

class Tile(object):
    """Tile wraps the Image class, so all methods that apply to images (show,
    save, crop, size, ...) apply to Tiles. Tiles also store contextual
    information that is used to reassembled them in the end."""
    def __init__(self, img, x, y, mask=None, ancestry=[], ancestor_size=None):
        self._img = img
        self.x = x
        self.y = y
        self._mask = mask.convert("L") if mask else None
        self._blank = None # meaning undetermined (so far)
        self._ancestry = ancestry
        self._depth = len(self._ancestry)
        if ancestor_size:
            self._ancestor_size = ancestor_size
        else:
            self._ancestor_size = self.size

    def crop(self, *args):
        if self._mask: self._mask.crop(*args)
        return self._img.crop(*args)

    def resize(self, *args):
        if self._mask: self._mask.resize(*args)
        return self._img.resize(*args)

    def __getattr__(self, key):
        if key == '_img':
            raise AttributeError()
        return getattr(self._img, key)

    def pos(self):
        return self.x, self.y 

    def avg_color(self):
        t = [0]*3
        for rgb in self._rgb:
            for i, c in enumerate(rgb):
                t[i] += c
        return [a/len(self._rgb) for a in t] 

    @property
    def ancestry(self):
        return self._ancestry

    @property
    def depth(self):
        return self._depth

    @property
    def ancestor_size(self):
        return self._ancestor_size

    @property
    def rgb(self):
        return self._rgb

    @rgb.setter
    def rgb(self, value):
        self._rgb = value

    @property
    def lab(self):
        return self._lab

    @lab.setter
    def lab(self, value):
        self._lab = value

    @property
    def match(self):
        return self._match

    @match.setter
    def match(self, value):
        self._match = value # sqlite Row object
        try:
            self._match_img = open_tile(self._match['filename'],
                (self._ancestor_size[1], self.ancestor_size[0]))
                # Reversed on purpose, for thumbnail. Largest possible size
                # we could want later.
        except IOError:
            logger.error("The filename specified in the database as "
                         "cannot be found. Check: %s", self._match['filename'])

    @property
    def match_img(self):
        return self._match_img

    @property
    def blank(self):
        return self._blank

    def determine_blankness(self, min_depth=1):
        """Decide whether this tile is blank. Where the mask is grey, tiles
        and blanked probabilitisically. The kwarg min_depth limits this
        scattered behavior to small tiles."""
        if not self._mask: # no mask
            self._blank = False
            return
        brightest_pixel = self._mask.getextrema()[1]
        if brightest_pixel == 0: # black mask 
            self._blank = True
        elif brightest_pixel == 255: # white mask
            self._blank = False
        elif self._depth < min_depth: # gray mask -- big tile
            self._blank = True
        elif 255*np.random.rand() > brightest_pixel: # small tile
            self._blank = True
        else:
            self._blank = False
        return

    def straddles_mask_edge(self):
        """A tile straddles an edge if it contains PURE white (255) and some
        nonwhite. A tile that contains varying shades of gray does not
        straddle an edge."""
        if not self._mask:
            return False
        darkest_pixel, brightest_pixel = self._mask.getextrema()
        if brightest_pixel != 255:
            return False
        if darkest_pixel == 255:
            return False
        return True
 
    def dynamic_range(self):
        """What is the dynamic range in this image? Return the
        average dynamic range over RGB channels."""
        return sum(map(lambda (x, y): y - x, self._img.getextrema()))//3

    def procreate(self):
        """Divide image into quadrants, make each into a child tile,
        and return them all in a list.""" 
        width = self._img.size[0] // 2
        height = self._img.size[1] // 2
        children = []
        for y in [0, 1]:
            for x in [0, 1]:
                tile_img = self._img.crop((x*width, y*height,
                                    (x + 1)*width, (y + 1)*height))
                if self._mask:
                    mask_img = self._mask.crop((x*width, y*height,
                                         (x + 1)*width, (y + 1)*height))
                else:
                    mask_img = None
                child = Tile(tile_img, self.x, self.y,
                             mask=mask_img,
                             ancestry=self._ancestry + [(x, y)],
                             ancestor_size=self._ancestor_size)
                children.append(child)
        return children

def partition(img, dimensions, mask=None, depth=0, hdr=80,
              debris=False, min_debris_depth=1):
    "Partition the target image into a list of Tile objects."
    if isinstance(dimensions, int):
        dimensions = dimensions, dimensions
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

def choose_match(lab, db, tolerance=1, usage_penalty=1):
    """If there is are good matches (within tolerance times the 'just noticeable
    difference'), return one at random. If not, choose the closest match
    deterministically. Return the match (as a sqlite Row dictionary) and the
    number of good matches."""
    JND = 2.3 # "just noticeable difference"
    (L1, a1, b1), (L2, a2, b2), (L3, a3, b3), (L4, a4, b4) = lab
    tokens = {'L1': L1, 'a1': a1, 'b1': b1,
              'L2': L2, 'a2': a2, 'b2': b2,
              'L3': L3, 'a3': a3, 'b3': b3,
              'L4': L4, 'a4': a4, 'b4': b4,
              'tol': tolerance*JND, 'usage_penalty': usage_penalty*JND}
    
    c = db.cursor()
    try:
        # Before we compute the exact color distance E, 
        # which is expensive and requires
        # adding 12 numbers in quadrature, the WHERE clause computes
        # a simpler upper bound on E and filters out disqualifying rows.
        # The survivors are ranked by their exact E plus a random component
        # determined by the tolerance. Thus, decisive winners are chosen
        # deterministically, but if there are many good matches, one is taken
        # at random.
        c.execute("""SELECT
                     image_id,
                     ((L1-({L1}))*(L1-({L1}))
                       + (a1-({a1}))*(a1-({a1})) 
                       + (b1-({b1}))*(b1-({b1}))
                       + (L2-({L2}))*(L2-({L2}))
                       + (a2-({a2}))*(a2-({a2})) 
                       + (b2-({b2}))*(b2-({b2}))
                       + (L3-({L3}))*(L3-({L3}))
                       + (a3-({a3}))*(a3-({a3})) 
                       + (b3-({b3}))*(b3-({b3}))
                       + (L4-({L4}))*(L4-({L4}))
                       + (a4-({a4}))*(a4-({a4}))
                       + (b4-({b4}))*(b4-({b4})))/4. as E_sq,
                     (L1-({L1}) + L2-({L2}) + L3-({L3}) + L4-({L4}))/4. as dL,
                     usages,
                     filename
                     FROM LabColors
                     JOIN Images using (image_id)
                     WHERE
                     L1-({L1}) + a1-({a1}) + b1-({b1})
                       + L2-({L2}) + a2-({a2}) + b2-({b2})
                       + L3-({L3}) + a3-({a3}) + b3-({b3})
                       + L4-({L4}) + a4-({a4}) + b4-({b4})
                     < 4*{tol}
                     ORDER BY
                     E_sq 
                     + {tol}*{tol}*RANDOM()/9223372036854775808.
                     + {usage_penalty}*{usage_penalty}*usages ASC
                     LIMIT 1""".format(**tokens))
                     # 9223372036854775808 is the range of sqlite RANDOM()
        match = c.fetchone()
        if not match:
            return choose_match(lab, db, tolerance + 1)
        c.execute("UPDATE Images SET usages=1+usages WHERE image_id=?",
                  (match['image_id'],))
    finally:
        c.close()
    logger.debug("%s", match)
    return match

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

@memo
def open_tile(filename, temp_size=(100,100)):
    """This memoized function only opens each image once."""
    im = Image.open(filename)
    im.thumbnail(temp_size) # Resize to fit within temp_size without cropping.
    return im

def matchmaker(tiles, db_name, tolerance=1, usage_penalty=1, usage_impunity=2):
    """Assign each tile a new image, and open that image in the Tile object."""
    db = connect(db_name)
    try:
        reset_usage(db)
        pbar = progress_bar(len(tiles), "Choosing and loading matching images")
        for tile in tiles:
            if tile.blank:
                pbar.next()
                continue
            tile.match = choose_match(tile.lab, db, tolerance,
                usage_penalty if tile.depth < usage_impunity else 0)
            pbar.next()
    finally:
        db.close()

def mosaic(tiles, pad=False, scatter=False, margin=0,
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
        pos = tile_position(tile, size, scatter, margin)
        mos.paste(crop_to_fit(tile.match_img, size), pos)
        pbar.next()
    return mos

def assemble_tiles(tiles):
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
        shrunk = tile.size[0]-4, tile.size[1]-4
        pos = tile_position(tile, shrunk, False, 0)
        mos.paste(tile.resize(shrunk), pos)
    return mos

def color_hex(rgb):
    "Convert [r, g, b] to a HEX value with a leading # character."
    return '#' + ''.join(chr(c) for c in rgb).encode('hex')

def reset_usage(db):
    try:
        c = db.cursor()
        c.execute("UPDATE Images SET usages=0")
    finally:
        c.close()
    return

def testing():
    pm.simple('images/samples', 'images/samples/dan-allan.jpg', (10,10), 'output.jpg')
