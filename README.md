# SEDkit

[![Powered by Astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org)
[![Build Status](https://travis-ci.org/hover2pi/sedkit.svg?branch=master)](https://travis-ci.org/hover2pi/sedkit)
[![Coverage Status](https://coveralls.io/repos/github/hover2pi/sedkit/badge.svg?branch=master&service=github)](https://coveralls.io/github/hover2pi/sedkit?branch=master)
[![Documentation Status](https://readthedocs.org/projects/sedkit/badge/?version=latest)](https://sedkit.readthedocs.io/en/latest/?badge=latest)

`sedkit` is a collection of pure Python modules for simple SED construction and analysis. Users can create individual SEDs or SED catalogs from spectra and/or photometry and calculate fundamental parameters (f<sub>bol</sub>, M<sub>bol</sub>, L<sub>bol</sub>, T<sub>eff</sub>, mass, log(g)) using the methods presented in [Filippazzo et al. (2015)](http://adsabs.harvard.edu/abs/2015ApJ...810..158F).

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

## Demo

An SED can be constructed by importing and initializing an `SED` object

```
from sedkit import SED
trap1 = SED(name='Trappist-1')
```

The `name` argument triggers a lookup in the Simbad database for meta, astrometric, and spectral type data. Interstellar reddening is calculated when possible.

Photometry can be added manually...

```
trap1.add_photometry('Johnson.V', 18.798, 0.082)
trap1.add_photometry('Cousins.R', 16.466, 0.065)
trap1.add_photometry('Cousins.I', 14.024, 0.115)
```

...and/or retrieved from catalogs.

```
trap1.find_2MASS()
```

Spectrum arrays or files can also be added to the SED data.

```
from pkg_resources import resource_filename
spec_file = resource_filename('sedkit', 'data/Trappist-1_NIR.fits')
trap1.add_spectrum_file(spec_file)
```

Other data which may affect the inferred fundamantal parameters can be set as well.

```
import astropy.units as u
trap1.spectral_type = 'M8'
trap1.age = 7.6*u.Gyr, 2.2*u.Gyr
trap1.radius = 0.121*u.R_sun, 0.003*u.R_sun
```

A variety of evolutionary model grids can be used to infer fundamental parameters,

```
trap1.evo_model = ‘DUSTY00’
trap1.mass_from_age()
```
![Lbol v. mass](/sedkit/data/figures/Lbol_v_mass.png | height=300px)

A variety or atmospheric model grids can be fit to the data,

```
from sedkit import BTSettl
trap1.fit_modelgrid(BTSettl())
```

And any arbitrary atlas of models can be applied as well.

``` =
from sedkit import SpexPrismLibrary
trap1.fit_modelgrid(SpexPrismLibrary())
```

Then the results can be printed and plotted.

```
trap1.results
trap1.plot()
```

![SED for Trappist-1](/sedkit/data/figures/sed_plot.png | height=300px)

Entire catalogs of `SED` objects can also be created and their properties can be arbitrarily compared and analyzed with the `sedkit.catalog.Catalog()` object.

![Lbol v. Spectral Type for a Catalog](/sedkit/data/figures/Lbol_v_SpT.png | height=300px)

Please read the full documentation for details on this functionality and much more.

## Documentation

Full documentation for the latest build can be found on [ReadTheDocs](https://sedkit.readthedocs.io/en/latest/).

The package also contains detailed Jupyter notebooks highlighting the core functionality of its primary classes, including

- [sedkit.spectrum.Spectrum](https://github.com/hover2pi/sedkit/blob/master/sedkit/notebooks/working_with_spectra.ipynb)
- [sedkit.sed.SED](https://github.com/hover2pi/sedkit/blob/master/sedkit/notebooks/create_sed.ipynb)
- [sedkit.catalog.Catalog](https://github.com/hover2pi/sedkit/blob/master/sedkit/notebooks/create_catalog.ipynb)

If you use or reference this software, please cite [Filippazzo et al. (submitted to PASP)]()

## Licensed

This project is Copyright (c) Joe Filippazzo and licensed under the terms of the BSD 3-Clause license. See the licenses folder for more information.
