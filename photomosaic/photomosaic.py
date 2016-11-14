import glob
import json
import warnings
import copy
import os
from collections import OrderedDict
from functools import partial
from tqdm import tqdm
import colorspacious
import numpy as np
from skimage import draw, img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import gray2rgb
from skimage.util import crop
from scipy.spatial import cKDTree
from scipy.cluster import vq
import matplotlib.pyplot as plt


options = {'imread': {},
           'perceptual': {"name": "J'a'b'",
                          "ciecam02_space": colorspacious.CIECAM02Space.sRGB,
                          "luoetal2006_space": colorspacious.CAM02UCS},
           'rgb': 'sRGB1'}


def set_options(imread=None, perceptual=None, rgb=None, flickr_api_key=None):
    """
    Set global options

    Parameters
    ----------
    imread : dict
        keyword arguments passed through to every call to ``imread``
        e.g., ``{'plugin': 'matplotlib'}``
    perceptual : string or dict
        perceptually-uniform colorspace used for color comparisions; see
        colorspacious documentation for details
    rgb : string or dict
        specific RGB colorspace used for color conversion
    flickr_api_key : string
        API key used by photomosaic.flickr module
    """
    global options
    if imread is not None:
        options['imread'].update(imread)
    if perceptual is not None:
        options['perceptual'] = perceptual
    if rgb is not None:
        options['rgb'] = rgb
    if flickr_api_key is not None:
        options['flickr_api_key'] = flickr_api_key


def basic_mosaic(image, pool, grid_dims, *, mask=None, depth=0):
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
    grid_dims : tuple
        Number of tiles along height, width.
    mask : array or None
        must have same shape as ``image``
    depth : int, optional
        Each tile can be subdividing this many times in regions of high
        contrast or along mask edges (if applicable). Default is 0.

    Returns
    -------
    mosaic : array

    Examples
    --------
    Before making the mosaic, you need a collection of images to use as tiles.
    A collection of analyzed images is a "pool". Analyzing the images takes
    much more time that making the mosaic, so it is a separate step.

        >>> pool = make_pool('directory_of_images/*.jpg')

    Load an image to turn into mosaic.

        >>> from skimage.io import imread, imsave
        >>> my_image = imread('my_image.jpg')

    Make the mosaic and save it.

        >>> mosaic = basic_mosaic(my_image, pool, (15, 15))
        >>> imsave('my_mosaic.jpg', mosaic)
    """
    # Size the image to be evenly divisible by the tiles.
    image = img_as_float(image)
    image = rescale_commensurate(image, grid_dims, depth)
    if mask is not None:
        mask = rescale_commensurate(mask)

    # Use perceptually uniform colorspace for all analysis.
    converted_img = perceptual(image)

    # Adapt the color palette of the image to resemble the palette of the pool.
    adapted_img = adapt_to_pool(converted_img, pool)

    # Partition the image into tiles and characterize each one's color.
    tiles = partition(adapted_img, grid_dims=grid_dims, mask=mask, depth=depth)
    tile_colors = [np.mean(adapted_img[tile].reshape(-1, 3), 0)
                   for tile in tqdm(tiles, desc='analyzing tiles')]

    # Match a pool image to each tile.
    match = simple_matcher(pool)
    matches = [match(tc) for tc in tqdm(tile_colors, desc='matching')]

    # Draw the mosaic.
    canvas = np.ones_like(image)  # white canvas same shape as input image
    return draw_mosaic(canvas, tiles, matches)


def perceptual(image):
    """
    Convert color from RGB (sRGB1) to a perceptually uniform colorspace.

    This is a convenience function wrapping ``colorspacious.csapce_convert``.
    To configure the specific perceptual colorspace used, change
    ``photomosaic.options['colorspace']``.

    Parameters
    ----------
    image : array
    """
    return colorspacious.cspace_convert(image, options['rgb'],
                                        options['perceptual'])


def rgb(image, clip=True):
    """
    Convert color from a perceptually uniform colorspace to RGB.

    This is a convenience function wrapping ``colorspacious.csapce_convert``.
    To configure the specific perceptual colorspace used, change
    ``photomosaic.options['perceptual']`` and ``photomosaic.options['rgb']``.

    Parameters
    ----------
    image : array
    clip : bool, option
        Clip values out of the gamut [0, 1]. True by default.
    """
    result = colorspacious.cspace_convert(image, options['perceptual'],
                                          options['rgb'])
    if clip:
        result = np.clip(result, 0, 1)
    return result


def adapt_to_pool(image, pool):
    """
    Adjust the color timing of an image to use colors available in the pool.

    For meaningful results, ``image`` and ``pool`` must be in the same
    colorspace.

    This is a convenience function wrapping ``color_palette`` and
    ``palette_map``.

    Parameters
    ----------
    image : array
    pool : dict

    Returns
    -------
    adapted_image : array

    Examples
    --------
    If the image is RGB, first convert to perceptual space. Finally, before
    visualizing, convert back.

        >>> rgb(adapt_to_pool(perceptual(image), pool)
    """
    image_palette = color_palette(image)
    pool_palette = color_palette(list(pool.values()))
    return palette_map(image_palette, pool_palette)(image)


def rescale_commensurate(image, grid_dims, depth=0):
    """
    For given grid dimensions and grid subdivision depth, scale image.

    The image is rescaled so that its shape can be evenly split into tiles.
    If necessary, one dimension is cropped to fit.

    Parameters
    ----------
    image : array
    grid_dims : tuple
        Number of tiles along height, width.
    depth : int, optional
        Each tile can be subdivided this many times. Default is 0.

    Returns
    -------
    rescaled_image : array
    """
    factor = np.array(grid_dims) * 2**depth
    new_shape = [int(factor[i] * np.ceil(image.shape[i] / factor[i]))
                 for i in [0, 1]]
    return crop_to_fit(image, new_shape)


def sample_pixels(image, size, replace=True):
    """
    Randomly sample pixels from an image.

    This is a wrapper around ``np.random.choice`` (which only works directly
    on 1-dimensional arrays).

    Parameters
    ----------
    image : array
    size : int
        number of pixels to sample
    replace : boolean, optional
        whether to sample with or without replacement; default True
    """
    num_pixels = np.product(image.shape[:-1])
    pixels = image.reshape(num_pixels, image.shape[-1])
    random_indexes = np.random.choice(num_pixels, size=size, replace=replace)
    return pixels[random_indexes]


def dominant_color(pixels, n_clusters=5):
    """
    Cluster colors and identify the "central" color of the largest cluster.

    Parameters
    ----------
    pixels : array
        List of pixels. The second axis is expected to be the color axis.
    n_clusters : int, optional
        number of clusters; default 5

    Returns
    -------
    dominant_color : array
    """
    colors, dist = vq.kmeans(pixels, n_clusters)
    vecs, dist = vq.vq(pixels, colors)
    counts, bins = np.histogram(vecs, len(colors))
    return colors[counts.argmax()]


def make_pool(glob_string, *, pool=None, skip_read_failures=True,
              analyzer=None, sample_size=1000):
    """
    Analyze a collection of images.

    For each file:
    (1) Read image.
    (2) Convert to perceptually-uniform color space.
    (3) Characterize the colors in the image as a vector.

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
        The default analyzer is :func:`numpy.mean` along the 0th axis.
    sample_size : int or None, optional
        Number of pixels to randomly sample before converting to perceptual
        colorspace and passing to ``analyzer``; if None, do not subsample.
        Default is 1000.

    Returns
    -------
    pool : dict-like
        mapping arguments for opening file to analyzer's result, e.g.:
        ``{(filename,): [1, 2, 3]}``
    """
    if pool is None:
        pool = {}
    if analyzer is None:
        analyzer = partial(np.mean, axis=0)
    filenames = glob.glob(glob_string)
    if not filenames:
        raise ValueError("No matches found for {}".format(glob_string))
    for filename in tqdm(filenames, desc='analyzing pool'):
        try:
            raw_image = imread(filename, **options['imread'])
        except Exception as err:
            if skip_read_failures:
                warnings.warn("Skipping {}; raised exception:\n    {}"
                              "".format(filename, err))
                continue
            raise
        image = standardize_image(raw_image)
        # Subsample before doing expensive color space conversion.
        if sample_size is not None:
            sample = sample_pixels(image, sample_size)
        else:
            sample = image.reshape(-1, 3)  # list of pixels
        # Convert color to perceptually-uniform color space.
        converted_sample = perceptual(sample)
        vector = analyzer(converted_sample)
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


def simple_matcher_unique(pool):
    """
    Build a matching function that matches to the closest color not yet used.

    It maintains an internal copy of the pool to tracky which items have been
    used.

    Parameters
    ----------
    pool : dict

    Returns
    -------
    match_func : function
        function that accepts a color vector and returns a match
    """
    pool = OrderedDict(pool)
    total = len(pool)

    def match(vector):
        """
        Return the key of the "nearest" unused pool image.

        Parameters
        ----------
        vector : array
            characterizing the color to be matched

        Returns
        -------
        args : tuple
            arguments that specify how to open the image
        """
        if len(pool) < 2:
            raise RuntimeError("All but one of the {total} pool images have "
                               "been used.".format(total=total))
        args = list(pool.keys())
        data = np.array([vector for vector in pool.values()])
        tree = cKDTree(data)
        distance, index = tree.query(vector, k=1)
        key = args[index]
        pool.pop(key)  # Remove this tile from our local copy of pool.
        return key

    return match


def simple_matcher(pool):
    """
    Build a matching function that simply matches to the closest color.

    It maintains an internal tree representation of the pool for fast lookups.

    Parameters
    ----------
    pool : dict

    Returns
    -------
    match_func : function
        function that accepts a color vector and returns a match
    """
    pool = OrderedDict(pool)  # same iteration order over keys and vals below
    args = list(pool.keys())
    data = np.array([vector for vector in pool.values()])
    tree = cKDTree(data)

    def match(vector):
        """
        Return the key of the pool image that is "nearest" (in color space).

        Parameters
        ----------
        vector : array
            characterizing the color to be matched

        Returns
        -------
        args : tuple
            arguments that specify how to open the image
        """
        distance, index = tree.query(vector, k=1)
        return args[index]

    return match


def draw_mosaic(image, tiles, matches, scale=1, resized_copy_cache=None):
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
    resized_copy_cache : dict or None, optional
        cache of images from the pool, sized to fit tiles
        entries look like: ``(pool_key, (height, width))``

    Returns
    -------
    image : array

    Examples
    --------
    Basic usage:

        >>> draw_mosaic(image, tiles, matches)

    Cache the resized pool images to speed up repeated drawings:

        >>> cache = {}  # any mutable mapping
        >>> draw_mosiac(image, tiles, matches, resized_copy_cache=cache)

    The above populated ``cache`` with every resized pool image used in a tile.
    Now, ``draw_mosaic`` will check the cache before loading the pool image and
    resizing it, which is the most expensive step.

        >>> draw_mosiac(image, tiles, matches, resized_copy_cache=cache)
    """
    scale = int(scale)
    if resized_copy_cache is None:
        resized_copy_cache = {}
    for tile, match_args in zip(tqdm(tiles, desc='drawing mosaic'), matches):
        if scale != 1:
            tile = tuple(slice(scale * s.start, scale * s.stop)
                         for s in tile)
        target_shape = _tile_shape(tile)
        cache_key = (match_args, target_shape)
        try:
            sized_match_image = resized_copy_cache[cache_key]
        except KeyError:
            target_shape = _tile_shape(tile)
            raw_match_image = imread(*match_args, **options['imread'])
            match_image = standardize_image(raw_match_image)
            sized_match_image = crop_to_fit(match_image, target_shape)
            resized_copy_cache[cache_key] = sized_match_image
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


def partition(image, grid_dims, mask=None, depth=0, split_thresh=10):
    """
    Parition the target image into tiles.

    Optionally, subdivide tiles that straddle a mask edge or contain high
    contrast, creating a grid with tiles of varied size.

    Parameters
    ----------
    grid_dims : int or tuple
        number of (largest) tiles along each dimension
    mask : array, optional
        Tiles that straddle a mask edge will be subdivided, creating a smooth
        edge.
    depth : int, optional
        Default is 0. Maximum times a tile can be subdivided.
    split_thresh : float or None, optional
        Threshold of standard deviation in color above which tile should be
        subdivided. This only applies if depth > 0. The (somewhat arbitrary)
        default is 10; a lower value leads to more aggressive splitting.

    Returns
    -------
    tiles : list
        list of pairs of slice objects
    """
    # Validate inputs.
    image = np.atleast_3d(np.asarray(image))
    if isinstance(grid_dims, int):
        tile_dims, = 2 * (grid_dims,)
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

    # Partition into equal-sized tiles. Each tile is a pair of slice objects.
    tile_height = image.shape[0] // grid_dims[0]
    tile_width = image.shape[1] // grid_dims[1]
    tiles = []
    total = np.product(grid_dims)
    with tqdm(total=total, desc='partitioning: depth 0') as pbar:
        for y in range(grid_dims[0]):
            for x in range(grid_dims[1]):
                tile = (slice(y * tile_height, (1 + y) * tile_height),
                        slice(x * tile_width, (1 + x) * tile_width))
                tiles.append(tile)
                pbar.update()

    # Discard any tiles that reside fully outside the mask.
    if mask is not None:
        tiles = [tile for tile in tiles if np.any(mask[tile])]

    # If depth > 0, subdivide any tiles that straddle a mask edge or that
    # contain an image with high contrast.
    num_channels = image.shape[-1]
    for d in range(depth):
        new_tiles = []
        for tile in tqdm(tiles, desc='partitioning: depth %d' % d):
            if ((mask is not None) and
                    np.any(mask[tile]) and np.any(~mask[tile])):
                # This tile straddles a mask edge.
                subtiles = _subdivide(tile)
                # Discard subtiles that reside fully outside the mask.
                subtiles = [tile for tile in subtiles if np.any(mask[tile])]
                new_tiles.extend(subtiles)
                continue
            if split_thresh is not None:
                num_pixels = np.product(image[tile].shape[:-1])
                pixels = image[tile].reshape(num_pixels, num_channels)
                if np.mean(np.std(pixels, 0)) > split_thresh:
                    # This tile has high color variation.
                    new_tiles.extend(_subdivide(tile))
                    continue
            new_tiles.append(tile)
        tiles = new_tiles
    return tiles


def scatter(tiles, margin):
    """
    Randomly nudge the tiles off center within a given margin.

    Also, shift all tiles in the positive direction by ``margin`` so that no
    slices are randomly placed < 0.

    Parameters
    ----------
    tiles : list
        list of tuples of slices
    margin : tuple
        maximum distance off tile center, given as ``(y, x)``

    Returns
    -------
    tiles : list
        a copy; the input is unchaged
    """
    y_margin, x_margin = margin
    new_tiles = []
    for tile in tiles:
        # random  shift + constant shift to ensure positive result
        dy = np.random.randint(-y_margin, 1 + y_margin) + y_margin
        dx = np.random.randint(-x_margin, 1 + x_margin) + x_margin
        new_tiles.append(translate(tile, (dy, dx)))
    return new_tiles


def translate(tile, offset):
    """
    Move a tile. This returns a moved copy; the original is unchaged.

    Parameters
    ----------
    tile : tuple
        containing y and x slice objects
    offset : tuple
        translation, given as ``(y, x)``

    Returns
    -------
    tile : tuple
        a copy; the input is unchaged
    """
    dy, dx = offset
    y, x = tile
    new_tile = (slice(y.start + dy, y.stop + dy),
                slice(x.start + dx, x.stop + dx))
    return new_tile


def pad(tile, padding):
    """
    Shrink a tile in place, leaving a padding. This returns a copy.

    Parameters
    ----------
    tile : tuple
        containing y and x slice objects
    padding : tuple
        number of pixels to leave along each side, given as ``(y, x)``

    Returns
    -------
    tile : tuple
        a copy; the input is unchaged
    """
    dy, dx = padding
    y, x = tile
    new_tile = (slice(y.start + dy, y.stop - dy),
                slice(x.start + dx, x.stop - dx))
    return new_tile


def color_palette(image, bins=256, **kwargs):
    """
    Compute the distribution of each color channel.

    This wraps ``numpy.histogram``, providing data munging relevant to
    image array with color channels. See numpy documentation for details on the
    meaning of the parameters.

    Parameters
    ----------
    image : array
        The last axis is expected to be the color axis.
    bins : int or list
        default 256; passed through to ``numpy.historgram``
    kwargs :
        passed through to ``numpy.histogram``

    Returns
    -------
    tuple :
        ``((counts, bins), (counts, bins), ...)`` -- one pair for color channel
    """
    image = np.asarray(image)
    num_channels = image.shape[-1]
    pixels = image.reshape(-1, num_channels)
    results = []
    for i in range(num_channels):
        counts, bin_edges = np.histogram(pixels[:, i], bins=bins, **kwargs)
        results.append((counts, bin_edges))
    return tuple(results)


def palette_map(old_palette, new_palette):
    """
    Build a function that maps from one color palette onto another.

    These is wrapper around :func:`hist_map` that applies it to each color
    channel.

    Parameters
    ----------
    old_palette: tuple
        list of histogram arrays ``(count, bins)`` for each color channel
    new_palette : tuple
        list of histogram arrays ``(count, bins)`` for each color channel

    Returns
    -------
    f : function
    """
    # Make a mapping function for each channel.
    functions = []
    for old, new in zip(old_palette, new_palette):
        f = hist_map(old, new)
        functions.append(f)

    # Make a function that applies each mapping function to its channel.
    def f(image):
        "Adapt colors in image from old palette to new."
        image = np.asarray(image)
        num_channels = image.shape[-1]
        if num_channels != len(functions):
            raise ValueError("expected image with {} color channels; this has "
                             "{}".format(len(functions), num_channels))
        num_pixels = np.product(image.shape[:-1])
        pixels = image.reshape(num_pixels, num_channels)

        result = np.empty_like(pixels)
        for i, f in enumerate(functions):
            result[:, i] = f(pixels[:, i])

        return result.reshape(image.shape)

    return f


def hist_map(old_hist, new_hist):
    """
    Build a function that maps from one distribution onto another.

    Parameters
    ----------
    old_hist : tuple
        Histogram arrays ``(count, bins)`` like those from ``numpy.histogram``,
        where bins give edges, such that ``len(bins) == 1 + len(counts)``.
    new_hist : tuple
        Histogram arrays ``(count, bins)`` like those from ``numpy.histogram``,
        where bins give edges, such that ``len(bins) == 1 + len(counts)``.

    Returns
    -------
    f : function
    """
    old_counts, old_bins = old_hist
    new_counts, new_bins = new_hist

    # validate input
    if 1 + len(old_counts) != len(old_bins):
        raise ValueError("The input old_hist is invalid. "
                         "Length of bins should be 1 + length of counts")
    if 1 + len(new_counts) != len(new_bins):
        raise ValueError("The input new_hist is invalid. "
                         "Length of bins should be 1 + length of counts")

    # cumulative distribution functions
    old_cdf = np.insert(np.cumsum(old_counts), 0, 0) / np.sum(old_counts)
    new_cdf = np.insert(np.cumsum(new_counts), 0, 0) / np.sum(new_counts)

    def f(arr):
        """
        Rescale values in ``arr`` from old distribution to new.
        """
        # Where in the old cdf did value(s) in arr fall?
        old_y = np.interp(arr, old_bins, old_cdf)

        # Find the value at the corresponding position in the new cdf.
        return np.interp(old_y, new_cdf, new_bins)

    return f


def _tile_center(tile):
    "Compute (y, x) center of tile."
    return tuple((s.stop + s.start) // 2 for s in tile)


def _tile_shape(tile):
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
    for y, x in tqdm(tiles):
        edges = ((y.start, x.start, y.stop - 1, x.start),
                 (y.stop - 1, x.start, y.stop - 1, x.stop - 1),
                 (y.stop - 1, x.stop - 1, y.start, x.stop - 1),
                 (y.start, x.stop - 1, y.start, x.start))
        for edge in edges:
            rr, cc = draw.line(*edge)
            annotated_image[rr, cc] = color  # tile edges
            annotated_image[_tile_center((y, x))] = color  # dot at center
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
    enlarged_shape = (tuple(np.ceil(np.array(image.shape[:len(shape)]) *
                                    shape[d]/image.shape[d])) +
                      image.shape[len(shape):])
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


def rainbow_of_squares(target_dir, shape=(10, 10), range_params=(0, 256, 15)):
    """
    Generate 5832 small solid-color tiles for experimentation and testing.

    Parameters
    ----------
    target_dir : string
    shape : tuple, optional
        default is (10, 10)
    range_params : tuple, optional
        Passed to ``range()`` to stride through each color channel.
        Default is ``(0, 256, 15)``.
    """
    os.makedirs(target_dir, exist_ok=True)
    with tqdm(total=3 * len(range(*range_params))) as pbar:
        canvas = np.ones(shape + (3,))
        for r in range(*range_params):
            for g in range(*range_params):
                for b in range(*range_params):
                    img = (canvas * [r, g, b]).astype(np.uint8)
                    filename = '{:03d}-{:03d}-{:03d}.png'.format(r, g, b)
                    # imsave warns when saving low-contrast images.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", ".*low contrast.*")
                        imsave(os.path.join(target_dir, filename), img)
                    pbar.update()


def export_pool(pool, filepath, abspath=True):
    """
    Export pool to json.

    This is a thin convenience wrapper around ``json.dump``. The pool is just a
    dict, but it contains numpy arrays, which must be converted to plain lists
    before being exported to JSON.

    Unlike the rest of this package, the export and import functions assume
    that the pool is keyed on a tuple with a string (e.g., a filepath) as its
    only element.

    Parameters
    ----------
    pool : dict
    filepath : string
    abspath : boolean, optional
        Convert pool keys (assumed to be filenames) to absolute paths if True.
    """
    with open(filepath, 'w') as f:
        json.dump({os.path.abspath(k[0]): list(v) for k, v in pool.items()}, f)


def import_pool(filepath):
    """
    Import pool from json.

    This is a thin convenience wrapper around ``json.load``. It puts the data
    into the expected data structures, which don't directly translate to JSON.

    Unlike the rest of this package, the export and import functions assume
    that the pool is keyed on a tuple with a string (e.g., a filepath) as its
    only element.

    Parameters
    ----------
    filepath : string

    Returns
    -------
    pool : dict
    """
    with open(filepath, 'r') as f:
        return {tuple([k]): np.array(v) for k, v in json.load(f).items()}


def plot_palette(palette, **kwargs):
    """
    Plot color palette (histograms of each channel).

    Parameters
    ----------
    palette : tuple
        color palette, such as created by :func:`color_palette`
    **kwargs
        passed through to ``matplotlib.Axes.plot``

    Returns
    -------
    lines : line artists created by matplotlib
    """
    fig, axes = plt.subplots(len(palette))
    lines = []
    for ax, (counts, bins) in zip(axes, palette):
        bin_centers = np.mean([bins[1:], bins[:-1]], 0)
        lines.append(ax.plot(bin_centers, counts, **kwargs))
    return lines
