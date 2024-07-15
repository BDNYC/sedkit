.. sedkit documentation master file, created by
   sphinx-quickstart on Fri Feb 19 11:20:05 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SEDkit Documentation
====================

This documentation describes `sedkit`, a collection of pure Python modules for simple SED construction and analysis. Users can create individual SEDs or SED catalogs from spectra and/or photometry and calculate fundamental parameters (:math:`f_\mbox{bol}`, :math:`M_\mbox{bol}`, :math:`L_\mbox{bol}`, :math:`T_\mbox{eff}`, mass, log(g)) using the methods presented in `Filippazzo et al. (2015) <http://adsabs.harvard.edu/abs/2015ApJ...810..158F>`_.

Installation
------------
Install via PyPI with

.. code::

    pip install sedkit

or via ``conda`` with

.. code::

    git clone https://github.com/hover2pi/sedkit.git
    cd sedkit
    conda env create -f environment.yml --force
    conda activate sedkit
    python setup.py install

Contents
========

.. automodapi:: sedkit/spectrum
.. automodapi:: sedkit/sed
.. automodapi:: sedkit/modelgrid
.. automodapi:: sedkit/catalog


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   spectrum.rst
   sed.rst
   catalog.rst
   modelgrid.rst

Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
