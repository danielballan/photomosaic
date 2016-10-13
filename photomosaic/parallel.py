import warnings
import glob
from functools import partial
import numpy as np
from skimage.io import imread
import colorspacious
import dask.bag
from dask.diagnostics import ProgressBar
from .photomosaic import (options, standardize_image, sample_pixels)


def make_pool(glob_string, *, pool=None, skip_read_failures=True,
              analyzer=None, sample_size=1000):
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
        The default analyzer is :func:`numpy.mean` along the 0th axis.
    sample_size : int or None, optional
        Number of pixels to randomly sample before converting to perceptual
        colorspace and passing to ``analyzer``; if None, do not subsample.
        Default is 1000.


    Returns
    -------
    cache : dict-like
        mapping arguments for opening file to analyzer's result, e.g.:
        ``{(filename,): [1, 2, 3]}``
    """
    filenames = glob.glob(glob_string)
    if not filenames:
        raise ValueError("No matches found for {}".format(glob_string))

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
        # Subsample before doing expensive color space conversion.
        if sample_size is not None:
            sample = sample_pixels(image, sample_size)
        else:
            sample = image.reshape(-1, 3)  # list of pixels
        # Convert color to perceptually-uniform color space.
        converted_sample = colorspacious.cspace_convert(sample,
                                                        options['rgb'],
                                                        options['perceptual'])

        if analyzer is None:
            analyzer = partial(np.mean, axis=0)
        vector = analyzer(converted_sample)
        return vector

    with ProgressBar():
        vectors = dask.bag.from_sequence(filenames).map(analyze).compute()

    if pool is None:
        pool = {}
    for filename, vector in zip(filenames, vectors):
        pool[filename] = vector
    return pool
