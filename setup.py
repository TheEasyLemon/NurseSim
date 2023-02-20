from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        [
            Extension('utils.find_availability', ['utils/find_availability.pyx'], include_dirs=[np.get_include()], extra_compile_args=['-O3', '-ffast-math'])
        ],
        compiler_directives = {
            'language_level': '3'
        },
    )
)

