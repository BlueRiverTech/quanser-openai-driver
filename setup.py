from setuptools import setup


VERSION = 0.1
INSTALL_REQUIRES = ['gym', 'numpy']

setup(name='gym_brt',
      version=VERSION,
      install_requires=INSTALL_REQUIRES,
      description='Blue River\'s OpenAI Gym wrapper around Quanser hardware.',
      url='https://github.com/BlueRiverTech/quanser-openai-driver/',
      author='Blue River Technology',
      license='MIT'
)
