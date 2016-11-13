import pytest
import os
import numpy as np
import tempfile
import photomosaic as pm
import photomosaic.parallel as pa
from skimage import draw


def test_pool_fixture(pool):
    # pool is generated in conftest.py using pytest 'fixtures' magic
    pass


def test_basic_mosiac(image, pool):
    "a smoke test of the integrated workflow"
    pm.basic_mosaic(image, pool, (5, 5))


def test_exhaust_simple_matcher_unique(pool):
    m = pm.simple_matcher_unique(pool)
    for _ in range(len(pool) - 1):
        m([0, 0, 0])
    with pytest.raises(RuntimeError):
        m([0, 0, 0])


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
    pal1 = pm.color_palette(img1)
    pal2 = pm.color_palette(img2)
    f = pm.palette_map(pal1, pal2)
    # the color 0.25 in pal1 maps onto the color 0.75 in pal2
    assert np.allclose(f([0.25]), 0.75)


def test_conversion(image):
    "just a smoke test of the convenience functions"
    p = pm.perceptual(image)
    pm.rgb(p)  # clip=True by default
    pm.rgb(p, clip=False)


def test_pool_parallel(pool):
    # reverse-engineer what the temporary pool is
    for k in pool.keys():
        path, = k
        break
    pool_dir = os.path.dirname(path)
    parallel_pool = pa.make_pool(os.path.join(pool_dir, '*.png'))
    assert pool.keys() == parallel_pool.keys()
    for k in pool:
        v1 = parallel_pool[k]
        v2 = pool[k]
        assert np.all(v1 == v2)


def test_hist_map():
    # check that number of bins must be 1 + number of counts
    with pytest.raises(ValueError):
        pm.hist_map(([1, 2, 3],  [1, 2, 3]),  ([1, 2, 3],  [1, 2, 3, 4]))
    with pytest.raises(ValueError):
        pm.hist_map(([1, 2, 3],  [1, 2, 3, 4]),  ([1, 2, 3],  [1, 2, 3]))

    # trivial case: map to self
    old = ([0, 0, 1, 1], [0, 1, 2, 3, 4])
    new = ([0, 0, 1, 1], [0, 1, 2, 3, 4])
    f = pm.hist_map(old, new)
    assert f(1.5) == 2
    assert f(2) == 2
    assert f(2.3) == 2.3
    assert f(2.5) == 2.5
    assert f(2.6) == 2.6
    assert f(3) == 3
    assert f(3.5) == 3.5
    assert f(4) == 4
    # limits are bin edges
    assert f(-1) == 2
    assert f(1) == 2
    assert f(10) == 4

    # map a step function to another nonoverlapping step function to its left
    old = ([1, 1], [2, 3, 4])
    new = ([1, 1], [0, 1, 2])
    f = pm.hist_map(old, new)
    assert f(-1) == 0
    assert f(0) == 0
    assert f(1) == 0
    assert f(2) == 0
    assert np.allclose(f(2.01), 0.01)
    assert np.allclose(f(2.49), 0.49)
    assert np.allclose(f(2.51), 0.51)
    assert np.allclose(f(2.99), 0.99)
    assert f(3) == 1
    assert np.allclose(f(3.01), 1.01)
    assert np.allclose(f(3.49), 1.49)
    assert np.allclose(f(3.51), 1.51)
    assert np.allclose(f(3.99), 1.99)
    assert f(4) == 2
    assert f(4.01) == 2
    assert f(100) == 2

    # map a step function to another nonoverlapping step function to its right
    old = ([1, 1], [0, 1, 2])
    new = ([1, 1], [2, 3, 4])
    f = pm.hist_map(old, new)
    assert f(-1) == 2
    assert f(0) == 2
    assert np.allclose(f(0.01), 2.01)
    assert np.allclose(f(0.5), 2.5)
    assert f(1) == 3
    assert f(2) == 4
    assert f(10) == 4

    # map a 'valley' to a 'peak'
    old = ([1, 0, 1], [0, 1, 2, 3])
    new = ([1], [1, 2])  # has intentionally different normalization
    f = pm.hist_map(old, new)
    assert np.allclose(f(0), 1)
    assert np.allclose(f(0.5), 1.25)
    assert np.allclose(f(1), 1.5)
    assert np.allclose(f(1.5), 1.5)
    assert np.allclose(f(2), 1.5)
    assert np.allclose(f(2.5), 1.75)
    assert np.allclose(f(3), 2)

    # map a 'peak' to a 'valley'
    old = ([1], [1, 2])
    new = ([1, 0, 1], [0, 1, 2, 3])
    f = pm.hist_map(old, new)
    assert np.allclose(f(1), 0)
    assert np.allclose(f(1.25), 0.5)
    assert np.allclose(f(1.4999999), 1)
    assert np.allclose(f(1.5), 2)
    assert np.allclose(f(1.75), 2.5)
    assert np.allclose(f(2), 3)
