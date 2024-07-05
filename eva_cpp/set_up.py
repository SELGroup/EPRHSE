from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules0 = [
    Extension(
        "apt_tools",
        sources=["/workspace/UPRTH/eva_cpp/apt_tools.pyx"],
        include_dirs=[numpy.get_include()],  # Add this line to include numpy headers
    )
]
ext_modules1 = [
    Extension(
        "apt_evaluate_foldout",
        sources=["/workspace/UPRTH/eva_cpp/apt_evaluate_foldout.pyx"],
        include_dirs=[numpy.get_include()],  # Add this line to include numpy headers
    )
]

setup(
    ext_modules = cythonize(ext_modules0)
)
setup(
    ext_modules = cythonize(ext_modules1)
)
'''
from setuptools import setup
from distutils.extension import Extension
import numpy

setup(
    name='MyProject',
    ext_modules=[
        Extension('Module', ['source.c'],
                  include_dirs=[numpy.get_include()]),
    ],
)
'''
