#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages
    setup
except ImportError:
    from distutils.core import setup
    setup

from codecs import open
from os import path

setup(
    name='SEDkit',
    version='0.1.0',
    description='Spectral energy distribution creation and analysis tools that work with the astrodbkit module.',
    url='https://github.com/hover2pi/SEDkit.git',
    author='Joe Filippazzo',
    author_email='jcfilippazzo@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
    ],
    keywords='astrophysics',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy','astropy','itertools','matplotlib','astrodbkit','emcee'],

)