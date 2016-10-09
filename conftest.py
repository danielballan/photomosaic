import tempfile
import os
import shutil
import pytest
import photomosaic as pm
from skimage.data import chelsea


@pytest.fixture(scope='module')
def pool():
    tempdirname = tempfile.mkdtemp()
    pm.generate_tile_pool(tempdirname, range_params=(0, 256, 30))
    pool = pm.make_pool(os.path.join(tempdirname, '*.png'))

    def delete_dm():
        shutil.rmtree(tempdirname)

    return pool


@pytest.fixture(scope='module')
def image():
    # sample image from scikit-image
    return chelsea()
