from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

if __name__=='__main__':
    cymodule = 'ehrenfest_cython'

    setup(
      name='ehrenfest_simulation',
      ext_modules=[Extension(cymodule, [cymodule + '.pyx'], include_dirs=[np.get_include()])],
      cmdclass={'build_ext': build_ext},
    )