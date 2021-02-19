.. _modelgrid:

ModelGrid
=========

Theoretical model atmosphere grids and spectral atlases are useful tools for characterizing stellar and substellar atmospheres. These data can be managed in ``sedkit`` with the :py:class:`~modelgrid.ModelGrid` class.

To use this resource, create a :py:class:`~modelgrid.ModelGrid` object, specify the parameters to track, and load it with data from a directory of XML files.

.. code:: python

    from sedkit import ModelGrid
    import astropy.units as q
    params = ['alpha', 'logg', 'teff', 'meta']
    mgrid = ModelGrid('BT-Settl', params, q.AA, q.erg/q.s/q.cm**2/q.AA, ref='2014IAUS..299..271A', **kwargs)
    mgrid.load('/path/to/data/models')

The table of model data can be viewed via the :py:attr:`~modelgrid.ModelGrid.index`` property and the parameter values can be returned via a parameter name + '_vals' property.

.. code:: python

    mgrid.index
    mgrid.teff_vals
    mgrid.logg_vals
    mgrid.meta_vals
    mgrid.alpha_vals

An individual model can be retrieved as a :py:class:`~spectrum.Spectrum` object by passing the desired parameter values as keyword arguments to the :py:meth:`~modelgrid.ModelGrid.get_spectrum` method. If the given parameter values do not correspond to a point on the grid, the spectrum can be interpolated or the closest grid point spectrum can be retrieved.

.. code:: python

    spec1 = mgrid.get_spectrum(teff=3500, logg=5.5, meta=0, alpha=0)
    spec2 = mgrid.get_spectrum(teff=3534, logg=5.3, meta=0.1, alpha=0, interp=True)
    spec3 = mgrid.get_spectrum(teff=3500, logg=5.5, meta=0, alpha=0, closest=True)

The :py:class:`~modelgrid.ModelGrid`. can be resampled to new parameter values by passing arrays to the desired keyword arguments.

.. code:: python

    import numpy as np
    new_mgrid = mgrid.resample_grid(teff=np.array())

Models can be inspected by passing the desired parameter values to the :py:meth:`~modelgrid.ModelGrid.plot` method.

.. code:: python

    mgrid.plot(teff=3500, logg=5.5, meta=0, alpha=0)

And a grid can be saved as a pickle file :py:meth:`~modelgrid.ModelGrid.save` method and loaded into a new object with the :py:func:`~modelgrid.load_ModelGrid`` function.

.. code:: python

    from sedkit import modelgrid as mg
    mgrid_path = '/path/to/model/grid/file.p'
    mgrid.save(mgrid_path)
    new_grid = mg.load_ModelGrid(mgrid_path)

Several :py:class:`~modelgrid.ModelGrid`. child classes exist for convenience.

.. code:: python

    btsettl = mg.BTSettl()          # BT-Settl model atmosphere grid
    spl = mg.SpexPrismLibrary()     # Spex Prism Library substellar spectral atlas
    fili15 = mg.Filippazzo2016()    # Substellar SED atlas from Filippazzo (2016)

The true utility of the :py:class:`~modelgrid.ModelGrid`. class is that it can be passed to a :py:class:`~spectrum.Spectrum` or :py:class:`~sed.SED` object to find a best fit model or best fit parameters.