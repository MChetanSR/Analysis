from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
cymodule = 'ehrenfest_cy'

setup(
  name='ehrenfest_cy',
  ext_modules=[Extension(cymodule, [cymodule + '.pyx'],)],
  cmdclass={'build_ext': build_ext},
)