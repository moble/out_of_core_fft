#!/usr/bin/env python

# Copyright (c) 2016, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/out_of_core_fft/blob/master/LICENSE>

from distutils.core import setup

setup(name='out_of_core_fft',
      description='Perform FFT on data set that does not fit into memory',
      author='Michael Boyle',
      url='https://github.com/moble/out_of_core_fft',
      packages=['out_of_core_fft', ],
      package_dir={'out_of_core_fft': '.'},
      )
