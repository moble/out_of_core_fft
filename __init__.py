# Copyright (c) 2016, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/out_of_core_fft/blob/master/LICENSE>

from __future__ import division


def _bytes_per_complex():
    import numpy as np
    _finfo=np.finfo(np.float64)
    return 2*(_finfo.nexp+_finfo.nmant+1)//8
bytes_per_complex = _bytes_per_complex()


try:
    from tempfile import _TemporaryDirectory
except ImportError:
    from contextlib import contextmanager as _contextmanager
    @_contextmanager
    def _TemporaryDirectory():
        import tempfile
        import shutil
        d = tempfile.mkdtemp()
        try:
            yield d
        except:
            raise
        finally:
            shutil.rmtree(d)


try:
    from contextlib import ExitStack as _ExitStack
except ImportError:
    class _ExitStack(object):

        def __init__(self):
            from collections import deque
            self._exit_callbacks = deque()

        def _push_cm_exit(self, cm, cm_exit):
            def _exit_wrapper(*exc_details):
                return cm_exit(cm, *exc_details)
            _exit_wrapper.__self__ = cm
            self.push(_exit_wrapper)

        def push(self, exit):
            _cb_type = type(exit)
            try:
                exit_method = _cb_type.__exit__
            except AttributeError:
                self._exit_callbacks.append(exit)
            else:
                self._push_cm_exit(exit, exit_method)
            return exit

        def enter_context(self, cm):
            _cm_type = type(cm)
            _exit = _cm_type.__exit__
            result = _cm_type.__enter__(cm)
            self._push_cm_exit(cm, _exit)
            return result

        def close(self):
            self.__exit__(None, None, None)

        def __enter__(self):
            return self

        def __exit__(self, *exc_details):
            import sys

            received_exc = exc_details[0] is not None

            frame_exc = sys.exc_info()[1]
            def _fix_exception_context(new_exc, old_exc):
                while 1:
                    exc_context = new_exc.__context__
                    if exc_context is old_exc:
                        return
                    if exc_context is None or exc_context is frame_exc:
                        break
                    new_exc = exc_context
                new_exc.__context__ = old_exc

            suppressed_exc = False
            pending_raise = False
            while self._exit_callbacks:
                cb = self._exit_callbacks.pop()
                try:
                    if cb(*exc_details):
                        suppressed_exc = True
                        pending_raise = False
                        exc_details = (None, None, None)
                except:
                    new_exc_details = sys.exc_info()
                    _fix_exception_context(new_exc_details[1], exc_details[1])
                    pending_raise = True
                    exc_details = new_exc_details
            if pending_raise:
                try:
                    fixed_ctx = exc_details[1].__context__
                    raise exc_details[1]
                except BaseException:
                    exc_details[1].__context__ = fixed_ctx
                    raise
            return received_exc and suppressed_exc


def transpose(dset_in, file_name_out, dset_name_out, chunk_cache_mem_size=1024**3, w0=1.0,
              R2=None, C2=None, close_file_when_done=False, access_axis=1, show_progress=False):
    """Transpose large matrix in HDF5 for fast access on transposed row

    Assuming the input dataset is chunked along its column (second) index, this function
    transposes it to an output dataset that is chunked along its column index.  Done naively,
    this operation can be extraordinarily slow, because we need to access rows in the input
    in order to write columns in the output.  But row access is slow because the entire
    column chunk needs to be read.  A similar problem occurs for writing.

    This function splits the difference, using the chunk cache to maximum advantage.  Roughly
    speaking, if we have sufficient memory to store M elements, we can set the output chunk
    size to sqrt(M) and process sqrt(M) chunks at a time.

    The input dataset may actually be one-dimensional, but interpreted as a matrix of shape
    R1xC1 = C2xR2.  Or the input may simply be a matrix of that shape, in which case R2 and
    C2 will be inferred.


    Parameters
    ----------
    dset_in : h5py.Dataset
        Open dataset in an h5py File object used as input.

    file_name_out : str
    dset_name_out : str
        Names of file and dataset in that file used for output.

    chunk_cache_mem_size : int
    w0 : float
        Parameters passed to `h5py_File_with_cache`.  See that function's documentation.

    R2, C2 : int
        Number of rows and columns in returned dataset.  These default to None, in which case
        the shape is inferred from the shape of the input.  Note that if the input is 1-D,
        this could turn out badly because each row will have just one element, which will be
        slow.

    close_file_when_done : bool
        If True, close output file and return None.  If False, return file handle and dataset.
        Default is False.

    access_axis : int
        Axis along which the output array will be accessed.  Default is 1, meaning the second
        (column) index.  Anything else will be interpreted as 0.

    show_progress : bool
        Periodically show progress through the main loop.  Defaults to False.


    """
    import numpy as np
    import h5py
    import h5py_cache

    # Figure out the basic output sizes
    if R2 is None:
        if len(dset_in.shape)>1:
            R2 = dset_in.shape[1]
        else:
            R2 = 1
    if C2 is None:
        C2 = dset_in.shape[0]

    bytes_per_object = np.dtype(dset_in.dtype).itemsize
    num_chunk_elements = chunk_cache_mem_size // bytes_per_object
    sqrt_n_c_e = int(np.sqrt(num_chunk_elements))

    assert R2*C2 == dset_in.size, ("Requested size {0}*{1}={2}".format(R2, C2, R2*C2)
                                   + " is incompatible with input size {0}".format(dset_in.size))

    # If the transposition can be done in memory, just do it
    if dset_in.size <= num_chunk_elements:
        # print("Doing transpose in memory")
        file_out = h5py.File(file_name_out, 'a')
        if dset_name_out in file_out:
            del file_out[dset_name_out]
        dset_out = file_out.create_dataset(dset_name_out, shape=(R2, C2), dtype=dset_in.dtype)
        dset_out[:] = dset_in[:].reshape(C2, R2).T

    else:

        # Set up output file and dset
        if access_axis == 1:
            n_cache_chunks = min(sqrt_n_c_e, R2)
            chunk_size = min(num_chunk_elements//n_cache_chunks, C2)
            chunks = (1, chunk_size)
        else:
            n_cache_chunks = min(sqrt_n_c_e, C2)
            chunk_size = min(num_chunk_elements//n_cache_chunks, R2)
            chunks = (chunk_size, 1)
        file_out = h5py_cache.File(file_name_out, 'a', chunk_cache_mem_size, w0, n_cache_chunks)
        if dset_name_out in file_out:
            del file_out[dset_name_out]
        dset_out = file_out.create_dataset(dset_name_out, shape=(R2, C2), dtype=dset_in.dtype, chunks=chunks)

        # Depending on whether input is 1-D or 2-D, we do this differently
        if len(dset_in.shape)==1:
            def submatrix_dset_in(r2a, r2b, c2a, c2b):
                temp = np.empty((c2b-c2a, r2b-r2a), dtype=dset_in.dtype)
                C1 = R2
                c1a, c1b = r2a, r2b
                for r1 in range(c2a, c2b):
                    try:
                        temp[r1-c2a] = dset_in[r1*C1+c1a:r1*C1+c1b]
                    except ValueError:
                        print(r2a, r2b, "\t", c2a, c2b, "\n", r1, c1a, c1b, "\t", C1)
                        print(r1-c2a, r1*C1+c1a, r1*C1+c1b, dset_in.shape, temp.shape)
                        raise
                return temp
        else:
            def submatrix_dset_in(r2a, r2b, c2a, c2b):
                return dset_in[c2a:c2b, r2a:r2b]

        # Do the transposition
        i = 1
        for c2a in range(0, C2, chunk_size):
            for r2a in range(0, R2, n_cache_chunks):
                if show_progress:
                    print("\t\t\t{0} of {1}".format(i, int(np.ceil(C2 / chunk_size) * np.ceil(R2 / n_cache_chunks))))
                c2b = min(C2, c2a+chunk_size)
                r2b = min(R2, r2a+n_cache_chunks)
                dset_out[r2a:r2b, c2a:c2b] = submatrix_dset_in(r2a, r2b, c2a, c2b).T
                i += 1

    if close_file_when_done:
        file_out.close()
        return
    else:
        return file_out, dset_out


def _general_fft(infile, ingroup, outfile='', outgroup='', overwrite=False, mem_limit=1024**3, inverse_fft=False,
                 show_progress=False):
    """

    This code is based on the algorithm presented in "Determining an out-of-core FFT decomposition strategy for
    parallel disks by dynamic programming", by Thomas H. Cormen in "Algorithms for parallel processing" (1999).  I
    have modified it slightly to take advantage of numpy's ability to broadcast the in-memory FFT efficiently,
    and to agree with numpy's conventions for the signs and normalizations of the FFT and inverse FFT.

    I also have not implemented the recursive strategy, though that should be simple enough.  But this should be
    sufficient for extraordinarily large datasets (with N around 10**18 or so), so I don't expect much need for
    recursion.

    Parameters
    ==========
    infile, ingroup : str, str
        Name of HDF5 file and group within that file in which to find the data to be FFTed.  The data should be a
        single set of 1-D data, either float or complex.
    outfile, outgroup : str, str
        Name of HDF5 file and group within that file in which to place the FFTed data.  If `overwrite` is True, these
        values are ignored (and default to empty strings, so that they need not be present).
    overwrite : bool
        If True, place the output data in the same file and group as the input data.  Note that this is only allowed
        if the input dataset is of type complex.
    mem_limit : int
        Size of memory (in bytes) to use for cache.  Default value is 1024**3 (1GB).
    show_progress : bool
        If True, print occasional messages about how much work has been done.  This is particularly useful for
        continuous-integration services, which typically like to see output every few minutes, or assumes the job is
        dead.  Default is False.

    """
    import h5py
    import numbers
    import os
    import numpy as np
    import contextlib

    n_cache_elements = mem_limit / bytes_per_complex
    sqrt_n_c_e = int(np.sqrt(n_cache_elements-1))

    with _ExitStack() as context_managers:

        # Open files and datasets for read/write
        if infile == outfile or overwrite:
            f_in = context_managers.enter_context(h5py.File(infile, 'r+'))
            f_out = f_in
        else:
            f_in = context_managers.enter_context(h5py.File(infile, 'r'))
            f_out = context_managers.enter_context(h5py.File(outfile, 'w'))
        if overwrite:
            x = f_in[ingroup]
            if not issubclass(x.dtype, numbers.Complex):
                raise ValueError("Flag `overwrite` is True, but input dtype={0} is not complex".format(x.dtype))
            X = x
        else:
            if infile == outfile and ingroup == outgroup:
                raise ValueError("Flag `overwrite` is False, but input and output files and groups are the same")
            x = f_in[ingroup]
            if outgroup in f_out:
                del f_out[outgroup]
            X = f_out.create_dataset(outgroup, shape=x.shape, dtype=x.dtype, chunks=(sqrt_n_c_e,))

        # Determine appropriate size and shape
        N = x.size
        if N != 2**int(np.log2(N)):
            raise ValueError("Size of input N={0} is not a power of 2 [log2(N)={1}]".format(N, np.log2(N)))
        R = 2**(int(np.log2(N))//2)
        M = 2**(int(np.log2(mem_limit))) // (4 * bytes_per_complex)
        M = min(M, N)
        R = min(M, R)
        C = N//R

        # Make temporary storage space
        tmpdirname = context_managers.enter_context(_TemporaryDirectory())
        fn_tmp1 = os.path.join(tmpdirname, 'temp1.h5')
        fn_tmp2 = os.path.join(tmpdirname, 'temp2.h5')

        # Step 1 of Cormen (1999):
        #   1) Interpret as RxC matrix and transpose to CxR
        # This cannot be done *quickly* by slicing because HDF5 doesn't like random
        # access -- or even determinate but long-strided access.  So we have to actually
        # transpose the original dataset into the temp file, and fft from that.
        if show_progress:
            print("\t\tStep 1: Transposing input data")
        f_tmp1, y = transpose(x, fn_tmp1, 'temp_y', R2=C, C2=R, chunk_cache_mem_size=mem_limit,
                              show_progress=show_progress)

        # Steps 2 and 3:
        #   2) Compute DFT of each R-element row individually
        #   3) Scale matrix element at index (k,l) by exp(k*l*2j*pi/N)
        # We first perform the DFT (Step 2).  Note that we use `ifft` because Cormen's DFT sign
        # convention is opposite that of numpy and FFTW, and scale the result because Cormen
        # does not scale, but numpy does.  Finally, we allow numpy to broadcast, performing
        # Step 3 easily.  This is then written back to disk by h5py.
        if show_progress:
            print("\t\tSteps 2 and 3: Computing DFTs and scaling")
        if inverse_fft:
            for k in range(C):
                if show_progress and k % max(C//50, 25) == 1:
                    print("\t\t\t{0} of {1}".format(k, C))
                y[k] = np.exp(np.arange(R)*(k*2j*np.pi/N)) * np.fft.ifft(y[k])
        else:
            for k in range(C):
                if show_progress and k % max(C//50, 25) == 1:
                    print("\t\t\t{0} of {1}".format(k, C))
                y[k] = np.exp(np.arange(R)*(-k*2j*np.pi/N)) * np.fft.fft(y[k])

        # Step 4:
        #   4) Transpose back to RxC shape
        if show_progress:
            print("\t\tStep 4: Transpose 2")
        f_tmp2, z = transpose(y, fn_tmp2, 'temp_z', R2=R, C2=C, chunk_cache_mem_size=mem_limit,
                              show_progress=show_progress)
        context_managers.enter_context(contextlib.closing(f_tmp2))

        # Delete `y` and its temp file to save on disk space
        del f_tmp1['temp_y']
        f_tmp1.close()
        os.remove(fn_tmp1)

        # Step 5:
        #   5) Compute DFT of each C-element row individually
        # We actually do`M//C` repetitions of Step 5 at a time, rather than individually,
        # to avoid the python loop overhead.  To achieve this, we need to select `M` data
        # points at a time, reshape into the appropriate 2-D array, and have numpy do all
        # those DFTs at once.  Once that is done, we reshape back to a vector.
        if show_progress:
            print("\t\tStep 5: Computing DFTs")
        if inverse_fft:
            for k in range(0, R, M//C):
                if show_progress:
                    print("\t\t\t{0} of {1}".format(k, R))
                z[k:k+M//C] = np.fft.ifft(z[k:k+M//C])
        else:
            for k in range(0, R, M//C):
                if show_progress:
                    print("\t\t\t{0} of {1}".format(k, R))
                z[k:k+M//C] = np.fft.fft(z[k:k+M//C])

        # Step 6:
        #   6) Transpose back to CxR and flatten
        if show_progress:
            print("\t\tStep 6: Transpose 3")
        i = 0
        for r in range(0, R, sqrt_n_c_e):
            for c in range(0, C, sqrt_n_c_e):
                if show_progress:
                    print("\t\t\t({0}, {1}) of ({2}, {3})".format(r+sqrt_n_c_e, c+sqrt_n_c_e, R, C))
                shape = (min(C, c+sqrt_n_c_e)-c, min(R, r+sqrt_n_c_e)-r)
                stripe = z[r:r+shape[1], c:c+shape[0]]
                for c2 in range(c, c+shape[0]):
                    X[c2*R+r:c2*R+r+shape[1]] = stripe[:, c2-c].T
                i += 1


def ifft(*args, **kwargs):
    kwargs['inverse_fft'] = True
    _general_fft(*args, **kwargs)
ifft.__doc__ = "Perform inverse FFT for very large dataset stored in HDF5 file" + _general_fft.__doc__


def fft(*args, **kwargs):
    kwargs['inverse_fft'] = False
    _general_fft(*args, **kwargs)
fft.__doc__ = "Perform FFT for very large dataset stored in HDF5 file" + _general_fft.__doc__
