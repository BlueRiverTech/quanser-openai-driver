from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    '''Allows the use of `numpy.get_include()` without having nmupy installed
    before before running the setup script.
    '''
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


extensions = [
Extension(
    'gym_brt.quanser.quanser_wrapper.quanser_wrapper',
    ['gym_brt/quanser/quanser_wrapper/quanser_wrapper.pyx'],
    include_dirs=['/opt/quanser/hil_sdk/include'],
    libraries=[
        'hil', 'quanser_runtime', 'quanser_common', 'rt', 'pthread',
        'dl', 'm', 'c'],
    library_dirs=['/opt/quanser/hil_sdk/lib'])
]


try:
    from Cython.Build import cythonize
    extensions=cythonize(extensions)  # Recompile
except:
    raise ImportError('Cython must be installed to compile from source.')


setup(name='gym_brt',
      version=0.1,
      cmdclass={'build_ext':build_ext},
      install_requires=['numpy', 'gym'],
      setup_requires=['numpy'],
      ext_modules=extensions,
      description='Blue River\'s OpenAI Gym wrapper around Quanser hardware.',
      url='https://github.com/BlueRiverTech/quanser-openai-driver/',
      author='Blue River Technology',
      license='MIT')
