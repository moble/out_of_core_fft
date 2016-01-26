# out_of_core_fft <a href="https://travis-ci.org/moble/out_of_core_fft"><img align="right" hspace="3" alt="Status of automatic build and test suite" src="https://travis-ci.org/moble/out_of_core_fft.svg?branch=master"></a> <a href="https://github.com/moble/out_of_core_fft/blob/master/LICENSE"><img align="right" hspace="3" alt="Code distributed under the open-source MIT license" src="http://moble.github.io/spherical_functions/images/MITLicenseBadge.svg"></a>

Fourier transforms are highly nonlocal, which can cause problems when dealing with very large data sets.  In particular,
standard algorithms cannot work with data sets too large to fit into memory.  On the other hand, the classic
Cooley-Tukey FFT algorithm shows that discrete Fourier transforms can be split up into smaller sub-problems.  This
module provides functions for FFTs that can work with the data directly on disk, extracting small subsets that fit into
memory, working on each individually, and then combining back onto disk to get the final result.  This implementation is
based on the algorithm presented by Thomas H. Cormen in "Algorithms for parallel processing" (1999).  A nontrivial
part of the implementation involves transposing the data on disk, for which I created a relatively simple, but fairly
fast, function included here simply as `transpose`.

## Usage

These functions assume that the data to be manipulated are stored in `HDF5` files.  The FFT and inverse FFT are called
with something like

```python
import out_of_core_fft
out_of_core_fft.fft('input.h5', 'x', 'output.h5', 'X')
out_of_core_fft.ifft('input2.h5', 'X', 'output2.h5', 'x')
```

Here, `x` and `X` are names for the datasets within the `HDF5` files.  Note that nothing is returned, because the result
is stored on disk, as requested.

See the docstrings for more details.



## Acknowledgments

The work of creating this code was supported in part by the Sherman Fairchild
Foundation and by NSF Grants No. PHY-1306125 and AST-1333129.
