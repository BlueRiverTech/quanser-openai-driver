from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("aero", ["aero.pyx"],
        include_dirs = ["/opt/quanser/hil_sdk/include", numpy.get_include()],
        libraries = ["hil", "quanser_runtime", "quanser_common", "rt", "pthread", "dl", "m", "c"],
        library_dirs = ["/opt/quanser/hil_sdk/lib"],
    )
]
setup(ext_modules=cythonize(extensions), include_dirs=[numpy.get_include()])
