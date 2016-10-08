import tempfile
import photomosaic as pm
from skimage.data import chelsea
from numpy.testing import assert_array_equal


def test_basic_mosiac(pool):
    "a smoke test of the integrated workflow"
    img = chelsea()
    mos = pm.basic_mosaic(img, pool, (15, 15))


def test_roundtrip_pool(pool):
    tf = tempfile.NamedTemporaryFile()
    pm.export_pool(pool, tf.name)
    pool2 = pm.import_pool(tf.name)
    for k1, k2 in zip(pool1.keys(), pool2.keys()):
        assert k1 == k2
    for v1, v2 in zip(pool1.values(), pool2.values()):
        assert_array_equal(v1, v2)
