SEDkit Documentation
====================

This documentation describes `sedkit`, a collection of pure Python modules for simple SED construction and analysis. Users can create individual SEDs or SED catalogs from spectra and/or photometry and calculate fundamental parameters (f<sub>bol</sub>, M<sub>bol</sub>, L<sub>bol</sub>, T<sub>eff</sub>, mass, log(g)) using the methods presented in [Filippazzo et al. (2015)](http://adsabs.harvard.edu/abs/2015ApJ...810..158F).

Installation
------------
Install via PyPI with::

    pip install sedkit

or via ``conda`` with::

    git clone https://github.com/hover2pi/sedkit.git
    cd sedkit
    conda env create -f environment.yml --force
    conda activate sedkit
    python setup.py install

Demo
----
An SED can be constructed by importing and initializing an ``SED``
object.

.. code:: python

    from sedkit import SED
    trap1 = SED(name='Trappist-1')

The ``name`` argument triggers a lookup in the Simbad database for meta,
astrometric, and spectral type data. Interstellar reddening is
calculated when possible.

Photometry can be added manually...

.. code:: python

    trap1.add_photometry('Johnson.V', 18.798, 0.082)
    trap1.add_photometry('Cousins.R', 16.466, 0.065)
    trap1.add_photometry('Cousins.I', 14.024, 0.115)

...and/or retrieved from Vizier catalogs with built-in methods.

.. code:: python

    trap1.find_2MASS()

Spectrum arrays or ASCII/FITS files can also be added to the SED data.

.. code:: python

    from pkg_resources import resource_filename
    spec_file = resource_filename('sedkit', 'data/Trappist-1_NIR.fits')
    import astropy.units as u
    trap1.add_spectrum_file(spec_file, wave_units=u.um, flux_units=u.erg/u.s/q.cm**2/u.AA)

Other data which may affect the calculated and inferred fundamantal
parameters can be set at any time.

.. code:: python

    trap1.spectral_type = 'M8'
    trap1.age = 7.6*u.Gyr, 2.2*u.Gyr
    trap1.radius = 0.121*u.R_sun, 0.003*u.R_sun

Results can be calculated at any time by checking the ``results``
property.

.. code:: python

    trap1.results

.. raw:: html

   <table>
    <thead>
     <tr>
      <th>

param

.. raw:: html

   </th>
      <th>

value

.. raw:: html

   </th>
      <th>

unc

.. raw:: html

   </th>
      <th>

units

.. raw:: html

   </th>
     </tr>
    </thead>
    <tr>
     <td>

Lbol

.. raw:: html

   </td>
     <td>

2.24e+30

.. raw:: html

   </td>
     <td>

6.49e+28

.. raw:: html

   </td>
     <td>

erg / s

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

Lbol\_sun

.. raw:: html

   </td>
     <td>

-3.23

.. raw:: html

   </td>
     <td>

0.013

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

Mbol

.. raw:: html

   </td>
     <td>

12.836

.. raw:: html

   </td>
     <td>

0.031

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

SpT

.. raw:: html

   </td>
     <td>

M8V

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

SpT\_fit

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

Teff

.. raw:: html

   </td>
     <td>

2581

.. raw:: html

   </td>
     <td>

37

.. raw:: html

   </td>
     <td>

K

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

Teff\_bb

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

Teff\_evo

.. raw:: html

   </td>
     <td>

2658.0666666666666

.. raw:: html

   </td>
     <td>

22.96666666666715

.. raw:: html

   </td>
     <td>

K

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

age

.. raw:: html

   </td>
     <td>

7.6

.. raw:: html

   </td>
     <td>

2.2

.. raw:: html

   </td>
     <td>

Gyr

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

bb\_source

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

blackbody

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

dec

.. raw:: html

   </td>
     <td>

-5.0413974999999995

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

fbol

.. raw:: html

   </td>
     <td>

1.21e-10

.. raw:: html

   </td>
     <td>

3.49e-12

.. raw:: html

   </td>
     <td>

erg / (cm2 s)

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

gravity

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

logg

.. raw:: html

   </td>
     <td>

5.281466666666667

.. raw:: html

   </td>
     <td>

0.005382456140353042

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

luminosity\_class

.. raw:: html

   </td>
     <td>

V

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

mass

.. raw:: html

   </td>
     <td>

0.0921333333333333

.. raw:: html

   </td>
     <td>

0.0013456140350877333

.. raw:: html

   </td>
     <td>

solMass

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

mbol

.. raw:: html

   </td>
     <td>

13.308

.. raw:: html

   </td>
     <td>

0.031

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

membership

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

name

.. raw:: html

   </td>
     <td>

Trappist-1

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

parallax

.. raw:: html

   </td>
     <td>

80.4512

.. raw:: html

   </td>
     <td>

0.12110000103712082

.. raw:: html

   </td>
     <td>

mas

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

prefix

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

ra

.. raw:: html

   </td>
     <td>

346.6223683333333

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

radius

.. raw:: html

   </td>
     <td>

0.121

.. raw:: html

   </td>
     <td>

0.003

.. raw:: html

   </td>
     <td>

solRad

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

reddening

.. raw:: html

   </td>
     <td>

9.259104263037443e-05

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
    <tr>
     <td>

spectral\_type

.. raw:: html

   </td>
     <td>

68.0

.. raw:: html

   </td>
     <td>

0.5

.. raw:: html

   </td>
     <td>

--

.. raw:: html

   </td>
    </tr>
   </table>

A variety of evolutionary model grids can be used to infer fundamental
parameters,

.. code:: python

    trap1.evo_model = 'DUSTY00'
    trap1.mass_from_age()

A variety of atmospheric model grids can be fit to the data,

.. code:: python

    from sedkit import BTSettl
    trap1.fit_modelgrid(BTSettl())

And any arbitrary atlas of models can be applied as well.

.. code:: python

    from sedkit import SpexPrismLibrary
    trap1.fit_modelgrid(SpexPrismLibrary())

Inspect the SED at any time with the interactive plotting method.

.. code:: python

    trap1.plot()

Entire catalogs of ``SED`` objects can also be created and their
properties can be arbitrarily compared and analyzed with the
``sedkit.catalog.Catalog()`` object.

Contents
========

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   spectrum.rst
   sed.rst
   catalog.rst
   modelgrid.rst

Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
