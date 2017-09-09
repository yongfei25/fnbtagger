"""
Food name tagger
"""
from setuptools import setup

setup(name='fnbtagger',
      version='0.1',
      description='Tag food names',
      packages=['fnbtagger'],
      install_requires=[
          'tensorflow-gpu'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
      