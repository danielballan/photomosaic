import tempfile
import pytest
import photomosaic as pm


@pytest.fixture(scope='module')
def pool():
    pool_dir = tempfile.mkdtemp()
    pm.generate_tile_pool(pool_dir)
    pool = pm.make_pool(pool_dir)
