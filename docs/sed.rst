.. _sed:

The ``SED`` class
=================

In its most basic application, an SED can be created simply by importing and instantiating the :class:`sedkit.sed.SED` class::

    from sedkit import SED
    x = SED()

The `name` argument can be specified to assign a string to the `name` attribute to just name your SED. However, this also triggers a Simbad lookup to find ancillary, meta, and astrometric data for the target when a match is found.






:ref:`spectrum <spectrum>`