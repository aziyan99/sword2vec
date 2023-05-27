import setuptools
from Cython.Build import cythonize
import numpy
import os

basedir = os.path.dirname(os.path.realpath(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

extensions = [
    setuptools.Extension('sword2vec_INIEZZY.helpers', [os.path.join(basedir, 'src/sword2vec_INIEZZY/utils/helpers.pyx')])
]

setuptools.setup(
    name="sword2vec_INIEZZY",
    version="0.0.1",
    author="Raja Azian",
    author_email="rajaazian08@gmail.com",
    description="A simple skipgram word2vec implementation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aziyan99/sword2vec",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    ext_modules=cythonize(extensions),
    install_require=[
        'click==8.1.3',
        'colorama==0.4.6',
        'Cython==0.29.35',
        'joblib==1.2.0',
        'nltk==3.8.1',
        'numpy==1.24.3',
        'regex==2023.5.5'
    ],
    include_dirs=[numpy.get_include()]
)
