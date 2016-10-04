import glob
import warnings
import copy
import os
import logging
import time
import random
from collections import OrderedDict
from tqdm import tqdm
import colorspacious
import numpy as np
from progress_bar import progress_bar
from skimage import draw, img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import gray2rgb
from skimage.util import crop
from scipy.spatial import cKDTree
from scipy.cluster import vq


options = {'imread': {},
           'colorspace': {"name": "J'a'b'",
                          "ciecam02_space": colorspacious.CIECAM02Space.sRGB,
                          "luoetal2006_space": colorspacious.CAM02UCS}}


def set_options(imread=None, colorspace=None):
    """
    Set global options

    Parameters
    ----------
    imread : dict
        keyword arguments passed through to every call to ``imread``
        e.g., ``{'plugin': 'matplotlib'}``
    colorspace : string or dict
        colorspace used for color comparisions; see colorspacious documentation
        for details
    """
    global options
    if imread is not None:
        options['imread'].update(imread)
    if colorspace is not None:
        options['colorspace'] = colorspace


def basic_mosaic(image, pool):
    """
    Make a mosaic in one step with some basic settings.

    See documentation (or the source code of this function) for more
    powerful features and customization.

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

    Example
    -------
    Before making the mosaic, you need a collection of images to use as tiles.
    A collection of analyzed images is a "pool". Analyzing the images takes
    much more time that making the mosaic, so it is a separate step.
    >>> pool = make_pool('directory_of_images/*.jpg')

    Load an image to turn into mosaic.
    >>> from skimage.io import imread, imsave
    >>> my_image = imread('my_image.jpg')

    Make the mosaic and save it.
    >>> mosaic = basic_mosaic(my_image)
    >>> imsave('my_mosaic.jpg', mosaic)
    """
    # TO DO crop to fit image to fit grid dimensions
    image = img_as_float(image)
    percep = colorspacious.cspace_convert(image, "sRGB1",
                                          options['colorspace'])
    tiles = partition(image, grid_dims=(10, 10), depth=1)
    matcher = SimpleMatcher(pool)
    tile_colors = [dominant_color(percep[tile]) for tile in tiles]
    matches = []
    for tile_color in tqdm(tile_colors, total=len(tile_colors)):
        matches.append(matcher.match(tile_color))
    canvas = np.ones_like(image)  # white canvas same shape as input image
    return draw_mosaic(canvas, tiles, matches)


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
    raveled_image = image.reshape(np.product(image.shape[:-1]),
                                  image.shape[-1])
    np.random.shuffle(raveled_image)  # shuffles in place
    sample = raveled_image[:min(len(raveled_image), sample_size)]
    colors, dist = vq.kmeans(sample, n_clusters)
    vecs, dist = vq.vq(sample, colors)
    counts, bins = np.histogram(vecs, len(colors))
    return colors[counts.argmax()]


def make_pool(glob_string, *, pool=None, skip_read_failures=True,
              analyzer=dominant_color):
    """
    Analyze a collection of images.

    For each file:
    1. Read image.
    2. Convert to perceptually-uniform color space.
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
            raw_image = imread(filename, **options['imread'])
        except Exception as err:
            if skip_read_failures:
                warnings.warn("Skipping {}; raised exception:\n    {}"
                             "".format(filename, err))
                continue
            raise
        image = standardize_image(raw_image)
        # Convert color to perceptually-uniform color space.
        percep = colorspacious.cspace_convert(image, "sRGB1",
                                              options['colorspace'])
        vector = analyzer(percep)
        pool[(filename,)] = vector
    return pool 


def standardize_image(image):
    """
    Ensure that image is float 0-1 RGB with no alpha.

    Parameters
    ----------
    image : array

    Returns
    -------
    image : array
        may or may not be a copy of the original
    """
    image = img_as_float(image)  # ensure float scaled 0-1
    # If there is no color axis, create one.
    if image.ndim == 2:
        image = gray2rgb(image)
    # Assume last axis is color axis. If alpha channel exists, drop it.
    if image.shape[-1] == 4:
        image = image[:, :, :-1]
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
        return self._args[index]


def draw_mosaic(image, tiles, matches, scale=1):
    """
    Assemble the mosaic, the final result.

    Parameters
    ----------
    image : array
        the "canvas" on which to draw the tiles, modified in place
    tiles : list
        list of pairs of slice objects
    matches : list
        for each tile in ``tiles``, a tuple of arguments for opening the
        matching image file
    scale : int, optional
        Scale up tiles for higher resolution image; default is 1.
        Any not-integer input will be cast to int.

    Returns
    -------
    image : array
    """
    scale = int(scale)
    for tile, match_args in zip(tiles, matches):
        if scale != 1:
            tile = tuple(slice(scale * s.start, scale * s.stop)
                         for s in tile)
        raw_match_image = imread(*match_args, **options['imread'])
        match_image = standardize_image(raw_match_image)
        sized_match_image = crop_to_fit(match_image, _tile_size(tile))
        image[tile] = sized_match_image
    return image 


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


def _tile_size(tile):
    "Compute the (y, x) dimensions of tile."
    return tuple((s.stop - s.start) for s in tile)


def draw_tile_layout(image, tiles, color=1):
    """
    Draw the tile edges on a copy of image. Make a dot at each tile center.

    This is a utility for inspecting a tile layout, not a necessary step in
    the mosaic-building process.

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


def crop_to_fit(image, shape):
    """
    Return a copy of image resized and cropped to precisely fill a shape.

    To resize a colored 2D image, pass in a shape with two entries. When
    ``len(shape) < image.ndim``, higher dimensions are ignored.

    Parameters
    ----------
    image : array
    shape : tuple
        e.g., ``(height, width)`` but any length <= ``image.ndim`` is allowed

    Returns
    -------
    cropped_image : array
    """
    # Resize smallest dimension (width or height) to fit.
    d = np.argmin(np.array(image.shape)[:2] / np.array(shape))
    enlarged_shape = (tuple(np.ceil(np.array(image.shape[:len(shape)])
                                    * shape[d]/image.shape[d]))
                      + image.shape[len(shape):])
    resized = resize(image, enlarged_shape)
    # Now the image is as large or larger than the shape along all dimensions.
    # Crop any overhang in the other dimension.
    crop_width = []
    for actual, target in zip(resized.shape, shape):
        overflow = actual - target
        # Center the image and crop, biasing left if overflow is odd.
        left_margin = np.floor(overflow / 2)
        right_margin = np.ceil(overflow / 2)
        crop_width.append((left_margin, right_margin))
    # Do not crop any additional dimensions beyond those given in shape.
    for _ in range(resized.ndim - len(shape)):
        crop_width.append((0, 0))
    cropped = crop(resized, crop_width)
    return cropped


def generate_tile_pool(target_dir, shape=(10, 10)):
    """
    Generate 5832 small solid-color tiles for experimentation and testing.

    This function takes about 30 seconds to run. It includes a progress bar.

    Parameters
    ----------
    target_dir : string
    shape : tuple, optional
        default is (10, 10)
    """
    canvas = np.ones(shape + (3,))
    for r in range(0, 256, 15):
        for g in range(0, 256, 15):
            for b in range(0, 256, 15):
                img = (canvas * [r, g, b]).astype(np.uint8)
                filename = '{:03d}-{:03d}-{:03d}.gif'.format(r, g, b)
                imsave(os.path.join(target_dir, filename), img)
