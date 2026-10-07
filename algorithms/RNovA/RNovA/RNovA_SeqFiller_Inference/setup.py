import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="RNOVA",
    ext_modules=cythonize("data/knapsack_build.pyx"),  # 指向正确的路径
    include_dirs=[numpy.get_include()],  # 如果你有额外的库路径，可以在此处添加
)