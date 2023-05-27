from setuptools import setup
from Cython.Build import cythonize
import numpy
import os

basedir = os.path.dirname(os.path.realpath(__file__))

setup(
    name="Helpers",
    ext_modules=cythonize(os.path.join(basedir, 'helpers.pyx')),
    zip_safe=False,
    include_dirs=[numpy.get_include()],
)
