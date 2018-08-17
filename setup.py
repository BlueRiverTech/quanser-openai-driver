from setuptools import setup, Extension
import argparse

try:
    import numpy as np
except ImportError:  # Numpy is not installed
    build_requires = ['numpy']


VERSION = 0.1
INSTALL_REQUIRES = ['numpy', 'gym']

extensions = [
    Extension('gym_brt.quanser.quanser_wrapper.quanser_wrapper',
        ['gym_brt/quanser/quanser_wrapper/quanser_wrapper.pyx'],
        include_dirs=['/opt/quanser/hil_sdk/include', np.get_include()],
        libraries=['hil', 'quanser_runtime', 'quanser_common', 'rt', 'pthread', 'dl', 'm', 'c'],
        library_dirs=['/opt/quanser/hil_sdk/lib'])
]

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--compile', action='store_true', help='Recompile the Cython code (Cython is required for this).')
args, _ = parser.parse_known_args()


try:
    from Cython.Build import cythonize
    # Recompile
    extensions=cythonize(extensions)
except ImportError:  # Cython is not installed
    pass  # Just use the precompiled extension

setup(name='gym_brt',
      version=VERSION,
      install_requires=INSTALL_REQUIRES,
      ext_modules=extensions,
      description='Blue River\'s OpenAI Gym wrapper around Quanser hardware.',
      url='https://github.com/BlueRiverTech/quanser-openai-driver/',
      author='Blue River Technology',
      license='MIT')
