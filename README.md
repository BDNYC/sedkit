# SEDkit

[![Powered by Astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org)
[![Build Status](https://travis-ci.org/hover2pi/sedkit.svg?branch=master)](https://travis-ci.org/hover2pi/sedkit)

`sedkit` is a collection of pure Python 3.5 modules for simple SED construction and analysis. Users can create individual SEDs or SED catalogs from spectra and/or photometry and calculate fundamental parameters (f<sub>bol</sub>, M<sub>bol</sub>, L<sub>bol</sub>, T<sub>eff</sub>, mass, log(g)) using the methods presented in [Filippazzo et al. (2015)](http://adsabs.harvard.edu/abs/2015ApJ...810..158F).

Requirements:
- `astropy>=3.0.2`
- `astroquery>=0.3.8`
- `bokeh>=0.12.6`
- `dustmaps>=1.0`
- `numpy>=1.13.3`
- `svo_filters>=0.2.5`

## Installation

Install via PYPI with

```
pip install sedkit
```

or via Github with

```
git clone https://github.com/hover2pi/sedkit.git
python sedkit/setup.py install
```

## Licensed

This project is Copyright (c) Joe Filippazzo and licensed under the terms of the BSD 3-Clause license. See the licenses folder for more information.
