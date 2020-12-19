#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages
    setup
except ImportError:
    from distutils.core import setup
    setup


setup(
    name='sedkit',
    version='1.0.8',
    description='Spectral energy distribution construction and analysis tools',
    url='https://github.com/hover2pi/sedkit',
    author='Joe Filippazzo',
    author_email='jfilippazzo@stsci.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='astrophysics',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy','astropy','bokeh','pysynphot','scipy','astroquery','dustmaps', 'pandas', 'svo_filters', 'healpy'],
    include_package_data=True,

)