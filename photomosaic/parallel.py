import warnings
import glob
from skimage.io import imread
import colorspacious
import dask.bag
from dask.diagnostics import ProgressBar
from .photomosaic import (options, standardize_image, sample_pixels,
                          dominant_color)


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
    filenames = glob.glob(glob_string)

    def analyze(filename):
        # closure over pbar
        try:
            raw_image = imread(filename, **options['imread'])
        except Exception as err:
            if skip_read_failures:
                warnings.warn("Skipping {}; raised exception:\n    {}"
                              "".format(filename, err))
                return
            raise
        image = standardize_image(raw_image)
        # Convert color to perceptually-uniform color space.
        sample = sample_pixels(image, 1000)
        converted_sample = colorspacious.cspace_convert(sample, "sRGB1",
                                                        options['colorspace'])
        vector = analyzer(converted_sample)
        return vector

    with ProgressBar():
        vectors = dask.bag.from_sequence(filenames).map(analyze).compute()

    if pool is None:
        pool = {}
    for filename, vector in zip(filenames, vectors):
        pool[filename] = vector
    return pool
