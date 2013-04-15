# Use this to compile dontneed.pyx

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("dontneed", ["dontneed.pyx"])]

setup(
    name="DontNeed",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)
