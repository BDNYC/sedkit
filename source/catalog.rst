.. _catalog:

Catalog
=======

Collections of :py:class:`~sed.SED` objects can be stored and analyzed in a :py:class:`~catalog.Catalog` object. One can be initialized and populated with an :py:class:`~sed.SED` object using the :py:meth:`~catalog.Catalog.add_SED` method.

.. code:: python

    from sedkit import Catalog, VegaSED
    vega = VegaSED()
    cat1 = Catalog(name='My New Catalog')
    cat1.add_SED(vega)

Catalogs can be merged with the addition operator.

.. code:: python

    from sedkit import SED
    sirius = SED('Sirius', spectral_type='A1V', method_list=['find_2MASS', 'find_WISE'])
    cat2 = Catalog('My Second Catalog')
    cat2.add_SED(sirius)
    cat = cat1 + cat2

To check the table of data and calculated parameters, just call the :py:attr:`~catalog.Catalog.results`` property. The wavelength and flux density units of all the SEDs can be checked and set with the :py:attr:`~catalog.Catalog.wave_units`` and :py:attr:`~catalog.Catalog.flux_units`` properties.

.. code:: python

    cat.results
    import astropy.units as q
    cat.wave_units = q.AA
    cat.flux_units = q.W / q.m**3

Additional columns of data can be added to the results table with the :py:meth:`~catalog.Catalog.add_column` method.

.. code:: python

    rv = np.array([-13.9, -5.5]) * q.km / q.s
    rv_unc = np.array([0.9, 0.1]) * q.km / q.s
    cat.add_column('radial_velocity', rv, rv_unc)

Data for individual columns (and associated uncertainties when applicable) can be retrieved by passing the desired column names to the :py:meth:`~catalog.Catalog.get_data` method.

.. code:: python

    spt_data, plx_data = cat.get_data('spectral_type', 'parallax')

The :py:class:`~sed.SED` object for a source can be retrieved with the :py:meth:`~catalog.Catalog.get_SED` method.

.. code:: python

    vega = cat.get_SED('Vega')

An interactive scatter plot of any two numeric columns can be made by passing the desired `x` and `y` parameter names from the results table to the :py:meth:`~catalog.Catalog.plot` method. Photometric colors can be calculated by passing two photometric band names with a ``-`` sign. The ``order`` argument accepts an integer and plots a polynomial of the given order. For busy plots, individual sources can be identified by passing the SED name to the ``identify`` argument. Similarly, setting the argument ``label_points=True`` prints the name of each source next to its data point.

.. code:: python

    cat.plot('Lbol', 'spectral_type', order=1)      # Lbol v. SpT plot with first order polynomial fit
    cat.plot('spectral_type', '2MASS.J-2MASS.H')    # SpT v. J-H color plot
    cat.plot('age', 'distance', identify=['Vega'])  # Age v. Dist with Vega circled in red
    cat.plot('parallax', 'mbol', label_points=True) # Plx v. mbol with labeled points

The SEDs can be plotted for visual comparison with the :py:meth:`~catalog.Catalog.plot_SEDs` method. The can be normalized to 1 by setting the argument ``normalize=True``.

.. code:: python

    cat.plot_SEDs('*', normalize=True)  # Plot of all SEDs
    cat.plot_SEDs(['Vega', 'Sirius'])   # Normalized plot of Vega and Sirius

The results table, photometry, and plots of each SED can be exported to a zip file or directory with the :py:meth:`~catalog.Catalog.export` method.

.. code:: python

    cat.export('/path/to/target/dir', zip=True)

The whole :py:class:`~catalog.Catalog` object can be serialized and loaded with the :py:meth:`~catalog.Catalog.save` and :py:meth:`~catalog.Catalog.load` methods, respectively.

.. code:: python

    cat_file = '/path/to/cat.p'
    cat.save(cat_file)
    new_cat = Catalog('A-type stars')
    new_cat.load(cat_file)

A catalog can also be made from an ASCII file with column names ``name``, ``ra``, and ``dec`` by passing the filepath to the :py:meth:`~catalog.Catalog.from_file` method. For each source in the list, an SED is created, the methods in the ``run_methods`` argument are run, and the SED is added to the catalog.

.. code:: python

    source_list = '/path/to/sources.csv'
    new_cat = Catalog()
    new_cat.from_file(source_list, run_methods=['find_2MASS', 'find_WISE', 'find_Gaia'])
