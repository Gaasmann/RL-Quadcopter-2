#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='Supercopter',
      version='0.0.1.dev1',
      description='Nicolas Haller\'s RL-Quadcopter-2 project',
      author='Nicolas Haller',
      author_email='nicolas@boiteameuh.org',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6'
      ],
      keywords='udacity nanodegree quadcopter',
      packages=find_packages(exclude=['.ipynb_checkpoints', '.pytest_cache',
                                      'tests', 'venv']),
      install_requires=[
          'matplotlib',
          'numpy',
          'pandas',
          'pytest',
      ])
