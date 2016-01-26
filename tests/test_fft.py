import os.path
import shutil
import tempfile
import numpy as np
import pytest
from h5py_cache import File


def test_something():
    d = tempfile.mkdtemp()
    try:
        with File(os.path.join(d, 'defaults.h5')) as f:
            pass
    finally:
        shutil.rmtree(d)
