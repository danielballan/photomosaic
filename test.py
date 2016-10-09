import numpy as np
import tempfile
import photomosaic as pm
from skimage import draw


def test_pool_fixture(pool):
    pass


def test_basic_mosiac(image, pool):
    "a smoke test of the integrated workflow"
    pm.basic_mosaic(image, pool, (5, 5))


def test_depth(pool):
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
    tf = tempfile.NamedTemporaryFile()
    pm.export_pool(pool, tf.name)
    pool2 = pm.import_pool(tf.name)
    for (k1, v1), (k2, v2) in zip(sorted(pool.items()), sorted(pool2.items())):
        assert k1 == k2
        assert np.all(v1 == v2)
