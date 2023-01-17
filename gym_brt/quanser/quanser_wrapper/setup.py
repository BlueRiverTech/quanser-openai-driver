from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "quanser_wrapper",
        ["quanser_wrapper.pyx"],
        include_dirs=["/opt/quanser/hil_sdk/include", numpy.get_include()],
        libraries=[
            "hil",
            "quanser_runtime",
            "quanser_common",
            "rt",
            "pthread",
            "dl",
            "m",
            "c",
        ],
        library_dirs=["/opt/quanser/hil_sdk/lib"],
    )
]
cy_extensions = cythonize(extensions, compiler_directives={"language_level" : "3"})
setup(ext_modules=cy_extensions, include_dirs=[numpy.get_include()])
