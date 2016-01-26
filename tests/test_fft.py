from __future__ import division, print_function
import os.path
import shutil
import tempfile
import numpy as np
import h5py
import out_of_core_fft

oneGB_complex = 64 * 1024**2


def test_nothing():
    d = tempfile.mkdtemp()
    try:
        with h5py.File(os.path.join(d, 'defaults.h5')) as f:
            list(f)
    finally:
        shutil.rmtree(d)


def test_transpose():
    temp_dir = tempfile.mkdtemp()
    print()
    try:
        # Write a test file of 1-d data to be reinterpreted as 2-d and transposed
        np.random.seed(1234)
        N = oneGB_complex * 2
        N_creation = min(16*1024**2, N)
        print("\tCreating file with test data, N={0}".format(N))
        with h5py.File(os.path.join(temp_dir, 'test_in.h5'), 'w') as f:
            x = f.create_dataset('x', shape=(N,), dtype=complex)
            N_creation = 16*1024**2
            for k in range(0, N, N_creation):
                size = min(N-k, N_creation)
                x[k:k+size] = np.random.random(size) + 1j*np.random.random(size)
        print("\t\tFinished creating file with test data")

        # Now transpose it to file
        print("\tPerforming first transpose")
        with h5py.File(os.path.join(temp_dir, 'test_in.h5'), 'r') as f:
            x = f['x']
            R2, C2 = N//1024, 1024
            f2, d = out_of_core_fft.transpose(x, os.path.join(temp_dir, 'test_transpose.h5'), 'x', R2=R2, C2=C2)
            f2.close()
        print("\t\tFinished performing first transpose")

        # Transpose it back, and check for equality
        print("\tPerforming second transpose")
        with h5py.File(os.path.join(temp_dir, 'test_transpose.h5'), 'r') as f:
            x = f['x']
            f2, d = out_of_core_fft.transpose(x, os.path.join(temp_dir, 'test_transpose2.h5'), 'x')
            try:
                assert np.all([np.array_equal(x[c2a:c2b, r2a:r2b].T, d[r2a:r2b, c2a:c2b])
                               for r2a in range(0, R2, min(R2, C2)) for r2b in [min(R2, r2a+min(R2, C2))]
                               for c2a in range(0, C2, min(R2, C2)) for c2b in []])
            finally:
                f2.close()
        print("\t\tFinished performing second transpose")
    finally:
        shutil.rmtree(temp_dir)
