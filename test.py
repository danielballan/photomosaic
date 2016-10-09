import numpy as np
import tempfile
import photomosaic as pm
from skimage import draw


def test_pool_fixture(pool):
    # pool is generated in conftest.py using pytest 'fixtures' magic
    pass


def test_basic_mosiac(image, pool):
    "a smoke test of the integrated workflow"
    pm.basic_mosaic(image, pool, (5, 5))


def test_depth(pool):
    "using greater depth should trace out the mask edge more closely"
    image = np.zeros((1000, 1000))
    rr, cc = draw.circle(300, 500, 150)
    image[rr, cc] = 1
    image = pm.rescale_commensurate(image, (5, 5), depth=2)
    mask = image.astype(bool)
    tiles0 = pm.partition(image, (5, 5), mask=mask, depth=0)
    tiles1 = pm.partition(image, (5, 5), mask=mask, depth=1)
    tiles2 = pm.partition(image, (5, 5), mask=mask, depth=2)
    assert len(tiles0) < len(tiles1) < len(tiles2)


def test_roundtrip_pool(pool):
    "save a pool as JSON and reload it"
    tf = tempfile.NamedTemporaryFile()
    pm.export_pool(pool, tf.name)
    pool2 = pm.import_pool(tf.name)
    for (k1, v1), (k2, v2) in zip(sorted(pool.items()), sorted(pool2.items())):
        assert k1 == k2
        assert np.all(v1 == v2)


def test_palette_map():
    "Map a color between two complete different color palettes."
    # two simulated 1d images with one color channel
    # img1 and img2 use completely different colors
    img1 = [[c] for c in np.linspace(0, 0.5, 1000)]
    img2 = [[c] for c in np.linspace(0.5, 1, 1000)]
    pal1 = pm.color_hist(img1)
    pal2 = pm.color_hist(img2)
    f = pm.palette_map(pal1, pal2)
    # the color 0.25 in pal1 maps onto the color 0.75 in pal2
    assert np.allclose(f([0.25]), 0.75)
