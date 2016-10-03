from tqdm import tqdm
from collections import OrderedDict
import glob
import warnings
import copy
import os
import logging
import time
import random
import numpy as np
import scipy
import scipy.misc
from scipy.cluster import vq
from scipy import interpolate
from PIL import Image
from PIL import ImageFilter
import sqlite3
import color_spaces as cs
from directory_walker import DirectoryWalker
from memo import memo
from progress_bar import progress_bar
from skimage import draw, img_as_float
from skimage.io import imread
from skimage.transform import resize
import colorspacious

# Configure logger.
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

def old_simple(image_dir, target_filename, dimensions, output_file):
    "A convenient wrapper for producing a traditional photomosaic."
    pool(image_dir, 'temp.db')
    orig_image = open(target_filename)
    image = tune(orig_image, 'temp.db', quiet=True)
    tiles = partition(image, dimensions)
    analyze(tiles)
    matchmaker(tiles, 'temp.db')
    mos = mosaic(tiles)
    mos = untune(mos, image, orig_image)
    logger.info('Saving mosaic to %s', output_file)
    mos.save(output_file)


def simple(image, pool):
    """
    Complete example

    Parameters
    ----------
    image : array
    pool : dict-like
        output from :func:`make_pool`; or any mapping of
        arguments for opening an image file to a vector characterizing it:
        e.g., ``{(filename,): [1, 2, 3]}``

    Returns
    -------
    mosaic : array
    """
    partition(image, grid_size=(10, 10), depth=1)
    matcher = SimpleMatcher(pool)
    tile_colors = [analyzer(image[tile]) for tile in tiles]
    matches = []
    for tile_color in tqdm(tile_colors, total=len(tile_colors)):
        matches.append(matcher.match(tile_color))
    canvas = np.ones_likes(image)  # white canvas same shape as input image
    return draw_mosaic(canvas, tiles, matches)


def draw_mosaic(image, tiles, matches):
    """
    Assemble the mosaic, the final result.

    Parameters
    ----------
    image : array
        the "canvas" on which to draw the tiles, modified in place
    tiles : list
        list of pairs of slice objects
    tile_matches : list
        for each tile in ``tiles``, a tuple of arguments for opening the
        matching image file

    Returns
    -------
    image : array
    """
    for tile, match_args in zip(tiles, tile_matches):
        raw_match_image = imread(*match_args)
        match_image = resize(raw_match_image, _tile_size(tile))
        image[tile] = match_image
    return image 


class SimpleMatcher:
    """
    This simple matcher returns the closest match.

    It maintains an internal tree representation of the pool for fast lookups.
    """
    def __init__(self, pool):
        self._pool = OrderedDict(pool)
        self._args = list(self._pool.keys())
        data = np.array([vector for vector in self._pool.values()])
        self._tree = cKDTree(data)

    def match(self, vector):
        distance, index = self._tree.query(vector, k=1)
        return self._pool[index]



def make_pool(glob_string, *, cache=None, skip_read_failures=True,
              analyzer=dominant_color):
    """
    Analyze a collection of images.

    For each file:
    1. Read image.
    2. Convert to perceptually-uniform color space "CIECAM02".
    3. Characterize the colors in the image as a vector.

    A progress bar is displayed and then hidden after completion.

    Parameters
    ----------
    glob_string : string
        a filepath with optional wildcards, like `'*.jpg'`
    pool : dict-like, optional
        dict-like data structure to hold results; if None, dict is used
    skip_read_failures: bool, optional
        If True (default), convert any exceptions that occur while reading a
        file into warnings and continue.
    analyzer : callable, optional
        Function with signature: ``f(img) -> arr`` where ``arr`` is a vector.
        The default analyzer is :func:`dominant_color`.

    Returns
    -------
    cache : dict-like
        mapping arguments for opening file to analyzer's result, e.g.:
        ``{(filename,): [1, 2, 3]}``
    """
    if pool is None:
        pool = {}
    filenames = glob.glob(glob_string)
    for filename in tqdm(filenames):
        try:
            raw_image = imread(filename)
            image = img_as_float(raw_image)  # ensure float scaled 0-1
        except Exception as err:
            if skip_read_failures:
                warnings.warn("Skipping {}; raised exception:\n    {}"
                             "".format(filename, err))
                continue
            raise
        # Assume last axis is color axis. If alpha channel exists, drop it.
        if image.shape[-1] == 4:
            image = image[:, :, :-1]
        # Convert color to perceptually-uniform color space.
        # "JCh" is a simplified "CIECAM02".
        percep = colorspacious.cspace_convert(image, "sRGB1", "JCh")
        vector = analyzer(percep)
        pool[(filename,)] = vector
    return pool 


def dominant_color(image, n_clusters=5, sample_size=1000):
    """
    Sample pixels from an image, cluster colors, and identify dominant color.

    Parameters
    ----------
    image: array
        The last axis is expected to be the color axis.
    n_clusters : int, optional
        number of clusters; default 5
    sample_size : int, optional
        number of pixels to sample; default 1000

    Returns
    -------
    dominant_color : array
    """
    image = copy.deepcopy(image)
    # 'raveled_image' is a 2D array, a 1D list of 1D color vectors
    raveled_image = image.reshape(scipy.product(image.shape[:-1]),
                                  image.shape[-1])
    np.random.shuffle(raveled_image)  # shuffles in place
    sample = raveled_image[:min(len(raveled_image), sample_size)]
    colors, dist = vq.kmeans(sample, n_clusters)
    vecs, dist = vq.vq(sample, colors)
    counts, bins = scipy.histogram(vecs, len(colors))
    return colors[counts.argmax()]


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

def image_histogram(image, mask=None):
    keys = 'red', 'green', 'blue'
    channels = dict(zip(keys, image.split()))
    hist= {}
    for ch in keys:
        if mask:
            h = channels[ch].histogram(mask.convert("1"))
        else:
            h = channels[ch].histogram()
        normalized_h = [256./sum(h)*v for v in h]
        hist[ch] = normalized_h
    return hist

def untune(mos, image, orig_image, mask=None, amount=1):
    if mask:
        m = crop_to_fit(mask, image.size)
        orig_palette = compute_palette(image_histogram(orig_image, m))
        image_palette = compute_palette(image_histogram(image, m))
    else:
        orig_palette = compute_palette(image_histogram(orig_image))
        image_palette = compute_palette(image_histogram(image))
    return Image.blend(mos, adjust_levels(mos, image_palette, orig_palette),
                          amount)

def tune(target_image, db_name, mask=None, quiet=True):
    """Adjust the levels of the image to match the colors available in the
    th pool. Return the adjusted image. Optionally plot some histograms."""
    db = connect(db_name)
    try:
        pool_hist = pool_histogram(db)
    finally:
        db.close()
    pool_palette = compute_palette(pool_hist)
    if mask:
        m = crop_to_fit(mask, target_image.size)
        target_palette = compute_palette(image_histogram(target_image, m))
    else:
        target_palette = compute_palette(image_histogram(target_image))
    adjusted_image = adjust_levels(target_image, target_palette, pool_palette)
    if not quiet:
        # Use the Image.histogram() method to examine the target image
        # before and after the alteration.
        keys = 'red', 'green', 'blue'
        values = [channel.histogram() for channel in target_image.split()]
        totals = map(sum, values)
        norm = [map(lambda x: 256*x/totals[i], val) \
                for i, val in enumerate(values)]
        orig_hist = dict(zip(keys, norm)) 
        values = [channel.histogram() for channel in adjusted_image.split()]
        totals = map(sum, values)
        norm = [map(lambda x: 256*x/totals[i], val) \
                for i, val in enumerate(values)]
        adjusted_hist = dict(zip(keys, norm)) 
        plot_histograms(pool_hist, title='Images in the pool')
        plot_histograms(orig_hist, title='Unaltered target image')
        plot_histograms(adjusted_hist, title='Adjusted target image')
    return adjusted_image

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

def adjust_levels(target_image, from_palette, to_palette):
    """Transform the colors of an image to match the color palette of
    another image."""
    keys = 'red', 'green', 'blue'
    channels = dict(zip(keys, target_image.split()))
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

def _subdivide(tile):
    "Create four tiles from the four quadrants of the input tile."
    tile_dims = [(s.stop - s.start) // 2 for s in tile]
    subtiles = []
    for y in (0, 1):
        for x in (0, 1):
            subtile = (slice(tile[0].start + y * tile_dims[0],
                             tile[0].start + (1 + y) * tile_dims[0]),
                       slice(tile[1].start + x * tile_dims[1],
                             tile[1].start + (1 + x) * tile_dims[1]))
            subtiles.append(subtile)
    return subtiles


def partition(image, grid_dims, mask=None, depth=0, hdr=80):
    """
    Parition the target image into tiles.

    Optionally, subdivide tiles that contain high contrast, creating a grid
    with tiles of varied size.

    Parameters
    ----------
    grid_dims : int or tuple
        number of (largest) tiles along each dimension
    mask : array, optional
        Tiles that straddle a mask edge will be subdivided, creating a smooth
        edge.
    depth : int, optional
        Default is 0. Maximum times a tile can be subdivided.
    hdr : float
        "High Dynamic Range" --- level of contrast that trigger subdivision

    Returns
    -------
    tiles : list
        list of pairs of slice objects
    """
    # Validate inputs.
    if isinstance(grid_dims, int):
        tile_dims, = image.ndims * (grid_dims,)
    for i in (0, 1):
        image_dim = image.shape[i]
        grid_dim = grid_dims[i]
        if image_dim % grid_dim*2**depth != 0:
            raise ValueError("Image dimensions must be evenly divisible by "
                             "dimensions of the (subdivided) grid. "
                             "Dimension {image_dim} is not "
                             "evenly divisible by {grid_dim}*2**{depth} "
                             "".format(image_dim=image_dim, grid_dim=grid_dim,
                                       depth=depth))

    # Partition into equal-sized tiles (Python slice objects).
    tile_height = image.shape[0] // grid_dims[0] 
    tile_width = image.shape[1] // grid_dims[1]
    tiles = []
    for y in range(grid_dims[0]):
        for x in range(grid_dims[1]):
            tile = (slice(y * tile_height, (1 + y) * tile_height),
                    slice(x * tile_width, (1 + x) * tile_width))
            tiles.append(tile)

    # Discard any tiles that reside fully outside the mask.
    if mask is not None:
        tiles = [tile for tile in tiles if np.any(mask[tile])]

    # If depth > 0, subdivide any tiles that straddle a mask edge or that
    # contain an image with high contrast.
    for _ in range(depth):
        new_tiles = []
        for tile in tiles:
            if (mask is not None) and np.any(mask[tile]) and np.any(~mask[tile]):
                # This tile straddles a mask edge.
                subtiles = _subdivide(tile)
                # Discard subtiles that reside fully outside the mask.
                subtiles = [tile for tile in subtiles if np.any(mask[tile])]
                new_tiles.extend(subtiles)
            elif False:  # TODO hdr check
                new_tiles.extend(_subdivide(tile))
            else:
                new_tiles.append(tile)
        tiles = new_tiles
    return tiles


def _tile_center(tile):
    "Compute (y, x) center of tile."
    return tuple((s.stop + s.start) // 2 for s in tile)


def draw_tiles(image, tiles, color=1):
    """
    Draw the tile edges on a copy of image. Make a dot at each tile center.

    Parameters
    ----------
    image : array
    tiles : list
        list of pairs of slices, as generated by :func:`partition`
    color : int or array
        value to "draw" onto ``image`` at tile boundaries

    Returns
    -------
    annotated_image : array
    """
    annotated_image = copy.deepcopy(image)
    for y, x in tiles:
        edges = ((y.start, x.start, y.stop - 1, x.start),
                 (y.stop - 1, x.start, y.stop - 1, x.stop - 1),
                 (y.stop - 1, x.stop - 1, y.start, x.stop - 1),
                 (y.start, x.stop - 1, y.start, x.start))
        for edge in edges:
            rr, cc = draw.line(*edge)
            annotated_image[rr, cc] = color  # tile edges
            annotated_image[_tile_center((y, x))] = color  # dot at tile center
    return annotated_image


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

def crop_to_fit(image, tile_size):
    "Return a copy of image cropped to precisely fill the dimesions tile_size."
    image_w, image_h = image.shape
    tile_w, tile_h = tile_size
    image_aspect = image_w/image_h
    tile_aspect = tile_w/tile_h
    if image_aspect > tile_aspect:
        # It's too wide.
        crop_h = image_h
        crop_w = int(round(crop_h*tile_aspect))
        x_offset = int((image_w - crop_w)/2)
        y_offset = 0
    else:
        # It's too tall.
        crop_w = image_w
        crop_h = int(round(crop_w/tile_aspect))
        x_offset = 0
        y_offset = int((image_h - crop_h)/2)
    image = image.crop((x_offset,
                    y_offset,
                    x_offset + crop_w,
                    y_offset + crop_h))
    image = image.resize((tile_w, tile_h), Image.ANTIALIAS)
    return image

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
        padding = map(lambda x, y: (x - y)//2, zip(*([size, tile.size])))
    if scatter:
        padding = [random.randint(0, 1 + margin), random.randint(0, 1 + margin)]
    pos = tuple(map(sum, zip(*([ancestor_pos] + rel_pos + [padding]))))
    return pos

@memo
def open_tile(filename, temp_size=(100,100)):
    """This memoized function only opens each image once."""
    im = Image.open(filename)
    im.thumbnail(temp_size, Image.ANTIALIAS) # Resize to fit within temp_size without cropping.
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

def mosaic(tiles, pad=False, scatter=False, margin=0, scaled_margin=False,
           background=(255, 255, 255)):
    """Return the mosaic image.""" 
    # Infer dimensions so they don't have to be passed in the function call.
    dimensions = map(max, zip(*[(1 + tile.x, 1 + tile.y) for tile in tiles]))
    mosaic_size = map(lambda x, y: x*y,
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
        mos.paste(crop_to_fit(tile.match_image, size), pos)
        pbar.next()
    return mos

def assemble_tiles(tiles, margin=1):
    """This is not used to build the final mosaic. It's a handy function for
    assembling new tiles (without blanks) to see how partitioning looks."""
    # Infer dimensions so they don't have to be passed in the function call.
    dimensions = map(max, zip(*[(1 + tile.x, 1 + tile.y) for tile in tiles]))
    mosaic_size = map(lambda x, y: x*y,
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


class Pool:
    def __init__(self, load_func, analyze_func, cache_path=None,
                 kdtree_class=None):
        self._load = load_func
        self._analyze = analyze_func
        if kdtree_class is None:
            from scipy.spatial import cKDTree
            kdtree_class = cKDTree
        self._kdtree_class = kdtree_class

        # self._cache maps args for loading image to vector describing it
        if cache_path is None:
            self._cache = {}
        else:
            from historydict import HistoryDict
            self._cache = HistoryDict(cache_path)

        self._keys_in_order = []
        self._data = []
        if self._cache:
            for key, val in self._cache.items():
                self._keys_in_order.append(key)
                self._data.append(val)
        self._build_tree()
        self._stale = False  # stale means KD tree is out-of-date

    def __setstate__(self, d):
        self._load = d['load_func']
        self._analyze = d['analyze_func']
        self._kdtree_class = d['kdtree_class']
        self._cache = d['cache']
        self._keys_in_order = d['keys_in_order']
        self._data = d['data']
        self._stale = d['stale']
        self._tree = d['tree']

    def __getstate__(self):
        d = {}
        d['load_func'] = self._load
        d['analyze_func'] = self._analyze
        d['kdtree_class'] = self._kdtree_class
        d['cache'] = self._cache
        d['keys_in_order'] = self._keys_in_order
        d['data'] = self._data
        d['stale'] = self._stale
        d['tree'] = self._tree
        return d

    def _build_tree(self):
        if not self._data:
            self._tree = self._kdtree_class(np.array([]).reshape(2, 0))
        else:
            self._tree = self._kdtree_class(self._data)

    def add(self, *args):
        arr = self._load_func(*args)
        self._keys_in_order.append(args)
        self._data.append(arr)
        self._cache[args] = self._analyze_func(arr)
        self._stale = True

    def query(self, x, k):
        if self._stale:
            self._build_tree()
            self._stale = False
        return self._tree.query(x, k)
    
    def get_image(self, *args):
        return self._load(*args)
