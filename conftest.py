import tempfile
import shutil
import pytest
import photomosaic as pm


@pytest.fixture(scope='module')
def pool():
    tempdirname = tempfile.mkdtemp()
    pm.generate_tile_pool(tempdirname)
    pool = pm.make_pool(tempdirname)
    shutil.rmtree(tempdirname)
