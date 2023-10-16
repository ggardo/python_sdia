from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "knn_cy",
        sources=["knn_cy.pyx"],
        include_dirs=[np.get_include()],  # Add this line
    )
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
)