import photomosaic as pm
import matplotlib.pyplot as plt
from skimage.data import chelsea


def test_basic_mosiac(pool):
    "a smoke test of the integrated workflow"
    img = chelsea()
    mos = pm.basic_mosaic(img, pool, (15, 15))
    plt.imshow(mos)
