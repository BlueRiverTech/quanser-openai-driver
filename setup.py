from setuptools import setup, Extension
import numpy as np


VERSION = 0.1
INSTALL_REQUIRES = ['gym', 'numpy']

extensions = [
    Extension('gym_brt.envs.QuanserWrapper.quanser_wrapper',
    	['gym_brt/envs/QuanserWrapper/quanser_wrapper.pyx'],
        include_dirs=['/opt/quanser/hil_sdk/include', np.get_include()],
        libraries=['hil', 'quanser_runtime', 'quanser_common', 'rt', 'pthread', 'dl', 'm', 'c'],
        library_dirs=['/opt/quanser/hil_sdk/lib']
    )
]

setup(name='gym_brt',
      version=VERSION,
      install_requires=INSTALL_REQUIRES,
      ext_modules=extensions,
      description='Blue River\'s OpenAI Gym wrapper around Quanser hardware.',
      url='https://github.com/BlueRiverTech/quanser-openai-driver/',
      author='Blue River Technology',
      license='MIT'
)
