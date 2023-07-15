from setuptools import setup, find_packages, Extension
import numpy
import os
from Cython.Build import cythonize

basdir = os.path.dirname(os.path.realpath(__file__))

long_description = ""
with open("README.md", "r") as fh:
    long_description = fh.read()

extensions = [
    Extension("sword2vec.helpers", [os.path.join(basdir, "src/sword2vec/helpers.pyx")])
]


setup(
    name="sword2vec",
    version="3.4.7",
    author="Raja Azian",
    author_email="rajaazian08@gmail.com",
    description="A simple skipgram word2vec implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aziyan99/sword2vec",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "Cython==0.29.35",
        "joblib==1.2.0",
        "nltk==3.8.1",
        "numpy==1.24.3",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
