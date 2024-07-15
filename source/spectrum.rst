.. _spectrum:

Spectrum
========

The :py:class:`~spectrum.Spectrum` class handles any 1D data representing the light from an astronomical source. A :py:class:`~spectrum.Spectrum` object is created by passing the wavelength, flux density, and (optional) uncertainty with ``astropy.units`` to the class.

.. code:: python

    import astropy.units as q
    import numpy as np
    from sedkit import Spectrum
    wavelength = np.linspace(1, 2, 100) * q.um
    flux = np.ones(100) * q.erg / q.s / q.cm**2 / q.AA
    unc = flux / 100.
    spec = Spectrum(wavelength, flux, unc, name='My spectrum')

The :py:class:`~spectrum.Spectrum` has a number of useful attributes.

.. code:: python

    spec.name           # The object's name
    spec.ref            # Reference(s)
    spec.header         # A header for the data
    spec.wave           # Unitless wavelength array
    spec.flux           # Unitless flux density array
    spec.unc            # Unitless uncertainty array (optional)
    spec.data           # Unitless array of [W, F] or [W, F, E]
    spec.spectrum       # List of [W, F] or [W, F, E] with units
    spec.wave_max       # The max wavelength
    spec.wave_min       # The min wavelength
    spec.wave_units     # The wavelength units
    spec.flux_units     # The flux density units
    spec.size           # The number of data points

After the :py:class:`~spectrum.Spectrum` has been created, it be manipulated in a number of ways.

It can be trimmed by passing a list of lower and upper bounds to :py:meth:`~spectrum.Spectrum.trim` method. The ``include`` argument accepts bounds for wavelength regions to include and the ``exclude`` argument accepts bounds for regions to exclude. A list of :py:class:`~spectrum.Spectrum` objects are returned unless the ``concat`` argument is set to ``True``, which simply concatenated the trimmed segment(s) into one :py:class:`~spectrum.Spectrum`.

.. code:: python

    trim_spec_include = spec.trim(include=[(1.2 * q.um, 1.6 * q.um)])
    trim_spec_exclude = spec.trim(exclude=[(1.2 * q.um, 1.6 * q.um)], concat=True)

The :py:meth:`~spectrum.Spectrum.interpolate` method accepts a new wavelength array and returns a new :py:class:`~spectrum.Spectrum` object interpolated to those values. The :py:meth:`~spectrum.Spectrum.resamp` method accepts the same input and resamples the spectrum onto the new wavelength array while preserving the total flux.

.. code:: python

    new_wav = np.linspace(1.2, 1.6, 50) * q.um
    new_wav_interp = spec.interpolate(new_wav)
    new_wav_resamp = spec.resamp(new_wav)

The :py:meth:`~spectrum.Spectrum.integrate` method integrates the curve to calculate the area underneath using the trapezoidal rule.

.. code:: python

    area = spec.integrate()

The :py:class:`~spectrum.Spectrum` can be smoothed using a Kaiser-Bessel smoothing window of narrowness ``beta`` and a given ``window`` size.

.. code:: python

    smooth_spec = spec.smooth(beta=2, window=11)

A :py:class:`~spectrum.Spectrum` may be flux calibrated to a given distance by passing a distance to the :py:meth:`~spectrum.Spectrum.flux_calibrate` method.

.. code:: python

    cal_spec = spec.flux_calibrate(5.1 * q.pc, 10 * q.pc)   # Flux calibrates the spectrum from 5.1 to 10 pc

A bandpass name or ``svo_filters.svo.Filter`` object can be used to convolve the spectrum, calculate a synthetic flux or magnitude, or renormalize it to a given magnitude.

.. code:: python

    from svo_filters import Filter
    jband = Filter('2MASS.J')
    conv_spec = spec.convolve_filter(jband)     # Convolved spectrum
    norm_spec = spec.renormalize(12.3, jband)   # Renormalized spectrum
    jmag = spec.synthetic_magnitude(jband)      # Synthetic magnitude
    jflux = spec.synthetic_flux(jband)          # Synthetic flux

It can also be normalized to a table of photometry weighted by the magnitude uncertainties with the :py:meth:`~spectrum.Spectrum.norm_to_mags` method. See the :ref:`SED` class for an example.

A :py:class:`~spectrum.Spectrum` object may also interact with another :py:class:`~spectrum.Spectrum` object in a number of ways. The :py:meth:`~spectrum.Spectrum.norm_to_spec` method creates a new object normalized to the input :py:class:`~spectrum.Spectrum` object and the :py:meth:`~spectrum.Spectrum.__add__` operation combines two :py:class:`~spectrum.Spectrum` objects in their common wavelength region or concatenates the segments.

.. code:: python

    spec2 = Spectrum(np.linspace(1.5, 2.5, 100) * q.um, flux * 1E-11, unc * 1E-11, name='Redder spectrum')
    normed_spec = spec.norm_to_spec(spec2)  # spec normalized to spec2
    combined_spec = spec + spec2            # New combined spectrum

Any :py:class:`~spectrum.Spectrum` may also be fit by a :py:class:`~modelgrid.ModelGrid` object to find the best fit model or spectrum. The :py:meth:`~spectrum.Spectrum.best_fit_model` method performs a simple goodness of fit test and returns the model with the best fit. The :py:meth:`~spectrum.Spectrum.mcmc_fit` method performs a MCMC fit to the grid and returns the best fit parameters with uncertainties. The details of any fit are stored as a dictionary in the :py:attr:`~spectrum.Spectrum.best_fit` attribute.

.. code:: python

    from sedkit import BTSettl
    bt_grid = BTSettl()
    spec.best_fit_model(bt)             # Goodness of fit
    spec.mcmc_fit(bt, params=['teff'])  # MCMC fit

For visual inspection of an interactive ``bokeh.plotting.figure``, use the :py:meth:`~spectrum.Spectrum.plot` method.

.. code:: python

    spec.plot()

Several child classes are also available. The :py:class:`~spectrum.Blackbody` class creates a blackbody spectrum for a given wavelength range and effective temperature.

.. code:: python

    from sedkit.spectrum import Blackbody
    bb_spec = Blackbody(np.linspace(1.5, 2.5, 100) * q.um, teff=2456 * q.K)

The ever useful spectrum of :py:class:`~spectrum.Vega` is easily created.

.. code:: python

    from sedkit.spectrum import Vega
    vega = Vega()

And a :py:class:`~spectrum.Spectrum` object can be created directly from a FITS or ASCII file with the :py:class:`~spectrum.FileSpectrum` class.

.. code:: python

    from sedkit.spectrum import FileSpectrum
    file_spec = FileSpectrum('/path/to/the/file.fits', wave_units=q.um, flux_units=q.erg/q.s/q.cm**2/q.AA)

Finally, the data can be exported to an ASCII file by passing a filepath to the :py:meth:`~spectrum.Spectrum.export` method.

.. code:: python

    spec.export('path/to/the/new/file.txt')