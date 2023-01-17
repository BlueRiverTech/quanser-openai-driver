import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


extensions = [
    Extension(
        "gym_brt.quanser.quanser_wrapper.quanser_wrapper",
        ["gym_brt/quanser/quanser_wrapper/quanser_wrapper.pyx"],
        include_dirs=["/opt/quanser/hil_sdk/include"],
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

# If Cython is installed build from source otherwise use the precompiled version
try:
    from Cython.Build import cythonize

    extensions = cythonize(extensions, compiler_directives={"language_level" : "3"})
except ImportError:
    pass


# Hacky way to check if the HIL SDK is installed (allows to run on Mac OS)
is_hil_sdk_installed = False
if os.path.isdir("/opt/quanser/hil_sdk/lib"):
    is_hil_sdk_installed = True

setup(
    name="gym_brt",
    version=0.1,
    cmdclass={"build_ext": build_ext} if is_hil_sdk_installed else {},
    install_requires=["numpy", "gym", "scipy", "vpython"],
    setup_requires=["numpy"],
    ext_modules=extensions if is_hil_sdk_installed else None,
    description="Blue River's OpenAI Gym wrapper around Quanser hardware.",
    url="https://github.com/BlueRiverTech/quanser-openai-driver/",
    author="Blue River Technology",
    license="MIT",
)
