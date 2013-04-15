# Use this to compile dontneed.pyx

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name="Cache tools",
    ext_modules=cythonize("dontneed.pyx")
)
