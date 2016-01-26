# Copyright (c) 2016, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/out_of_core_fft/blob/master/LICENSE>


def transpose(dset_in, file_name_out, dset_name_out, chunk_cache_mem_size=1024**3, w0=1.0,
              R2=None, C2=None, close_file_when_done=False, access_axis=1):
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
        print("Doing transpose in memory")
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
        for c2a in range(0, C2, chunk_size):
            for r2a in range(0, R2, n_cache_chunks):
                c2b = min(C2, c2a+chunk_size)
                r2b = min(R2, r2a+n_cache_chunks)
                dset_out[r2a:r2b, c2a:c2b] = submatrix_dset_in(r2a, r2b, c2a, c2b).T

    if close_file_when_done:
        file_out.close()
        return
    else:
        return file_out, dset_out


