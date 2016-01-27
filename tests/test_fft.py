from __future__ import division, print_function
import os.path
import pytest
import numpy as np
import h5py
import out_of_core_fft

oneGB_complex = 64 * 1024**2
oneMB_complex = 64 * 1024


@pytest.mark.parametrize("N", [
    oneMB_complex * 2,
    oneGB_complex * 2,
])
def test_transpose(N):
    print()
    with out_of_core_fft._TemporaryDirectory() as temp_dir:
        # Write a test file of 1-d data to be reinterpreted as 2-d and transposed
        np.random.seed(1234)
        N_creation = min(16*1024**2, N)
        print("\tCreating file with test data, N={0}".format(N))
        with h5py.File(os.path.join(temp_dir, 'test_in.h5'), 'w') as f:
            x = f.create_dataset('x', shape=(N,), dtype=complex)
            for k in range(0, N, N_creation):
                size = min(N-k, N_creation)
                x[k:k+size] = np.random.random(size) + 1j*np.random.random(size)
        print("\t\tFinished creating file with test data")

        # Now transpose it to file
        print("\tPerforming first transpose")
        with h5py.File(os.path.join(temp_dir, 'test_in.h5'), 'r') as f:
            x = f['x']
            R2, C2 = N//1024, 1024
            f2, d = out_of_core_fft.transpose(x, os.path.join(temp_dir, 'test_transpose.h5'), 'x', R2=R2, C2=C2,
                                              show_progress=True)
            f2.close()
        print("\t\tFinished performing first transpose")

        # Transpose it back, and check for equality
        print("\tPerforming second transpose")
        with h5py.File(os.path.join(temp_dir, 'test_transpose.h5'), 'r') as f:
            x = f['x']
            f2, d = out_of_core_fft.transpose(x, os.path.join(temp_dir, 'test_transpose2.h5'), 'x',
                                              show_progress=True)
            try:
                assert np.all([np.array_equal(x[c2a:c2b, r2a:r2b].T, d[r2a:r2b, c2a:c2b])
                               for r2a in range(0, R2, min(R2, C2)) for r2b in [min(R2, r2a+min(R2, C2))]
                               for c2a in range(0, C2, min(R2, C2)) for c2b in []])
            finally:
                f2.close()
        print("\t\tFinished performing second transpose")


@pytest.mark.parametrize("myfunc, npfunc", [
    (out_of_core_fft.ifft, np.fft.ifft),
    (out_of_core_fft.fft, np.fft.fft),
])
def test_dft(myfunc, npfunc):
    print()
    with out_of_core_fft._TemporaryDirectory() as temp_dir:
        # Write a test file
        np.random.seed(1234)
        N = oneMB_complex * 4
        fname_in = os.path.join(temp_dir, 'test_in.h5')
        fname_out = os.path.join(temp_dir, 'test_out.h5')
        print("\tCreating file with test data, N={0}".format(N))
        with h5py.File(fname_in, 'w') as f:
            f.create_dataset('X', data=(np.random.random(N) + 1j*np.random.random(N)))
        print("\t\tFinished creating file with test data")

        # FFT it
        print("\tPerforming out-of-core FFT")
        myfunc(fname_in, 'X', fname_out, 'x', mem_limit=1024**2)
        print("\t\tFinished performing out-of-core FFT")

        # Compare to in-core FFT
        with h5py.File(fname_in, 'r') as f_in, h5py.File(fname_out, 'r') as f_out:
            assert np.allclose(npfunc(f_in['X']), f_out['x'])


def test_roundtrip_fft():
    print()
    with out_of_core_fft._TemporaryDirectory() as temp_dir:
        fname_x = os.path.join(temp_dir, 'x.h5')
        fname_X = os.path.join(temp_dir, 'Xtilde.h5')
        fname_xx = os.path.join(temp_dir, 'xx.h5')

        # Write a test file of 1-d data
        np.random.seed(1234)
        N = oneGB_complex // 2
        mem_limit = N  # Note that the units of mem_limit and N are different, but this ensures out-of-core
        N_creation = min(16*1024**2, N)
        print("\tCreating file with test data, N={0}".format(N))
        with h5py.File(fname_x, 'w') as f:
            x = f.create_dataset('x', shape=(N,), dtype=complex)
            for k in range(0, N, N_creation):
                size = min(N-k, N_creation)
                x[k:k+size] = np.random.random(size) + 1j*np.random.random(size)
        print("\t\tFinished creating file with test data")

        # Now FFT it to file
        print("\tPerforming out-of-core FFT")
        out_of_core_fft.fft(fname_x, 'x', fname_X, 'X', show_progress=True, mem_limit=mem_limit)
        print("\t\tFinished performing out-of-core FFT")

        # Now inverse FFT it to file
        print("\tPerforming out-of-core inverse FFT")
        out_of_core_fft.ifft(fname_X, 'X', fname_xx, 'xx', show_progress=True, mem_limit=mem_limit)
        print("\t\tFinished performing out-of-core inverse FFT")

        # Check for equality
        print("\tTesting equality of x and ifft(fft(x))")
        with h5py.File(fname_x, 'r') as f_in, h5py.File(fname_xx, 'r') as f_out:
            x = f_in['x']
            xx = f_out['xx']
            step = oneGB_complex // 16
            assert np.all([np.allclose(x[i_a:i_b], xx[i_a:i_b])
                           for i_a in range(0, x.shape[0], step) for i_b in [min(x.shape[0], i_a+step)]])
        print("\tFinished testing equality")
