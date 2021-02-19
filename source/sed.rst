.. _sed:

SED
===

An SED (spectral energy distribution) can be constructed by importing and initializing an :py:class:`~sed.SED` object.

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
    trap1.find_WISE()
    trap.find_SDSS()

Spectrum arrays or ASCII/FITS files can also be added to the SED data.

.. code:: python

    from pkg_resources import resource_filename
    spec_file = resource_filename('sedkit', 'data/Trappist-1_NIR.fits')
    import astropy.units as q
    trap1.add_spectrum_file(spec_file, wave_units=q.um, flux_units=q.erg / q.s / q.cm**2 / q.AA)

Other data which may affect the calculated and inferred fundamantal
parameters can be set at any time.

.. code:: python

    trap1.spectral_type = 'M8'
    trap1.age = 7.6 * q.Gyr, 2.2 * q.Gyr
    trap1.radius = 0.121 * q.R_sun, 0.003 * q.R_sun

Results can be calculated at any time by checking the :py:attr:`~sed.SED.results` property.

.. code:: python

    trap1.results

A variety of evolutionary model grids can be used to infer fundamental parameters,

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

Entire catalogs of :py:class:`~sed.SED` objects can also be created and their
properties can be arbitrarily compared and analyzed with the
:py:class:`~catalog.Catalog` object.