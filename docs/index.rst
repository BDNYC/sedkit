SEDkit Documentation
====================

This documentation describes `sedkit`, a collection of pure Python modules for simple SED construction and analysis. Users can create individual SEDs or SED catalogs from spectra and/or photometry and calculate fundamental parameters (f<sub>bol</sub>, M<sub>bol</sub>, L<sub>bol</sub>, T<sub>eff</sub>, mass, log(g)) using the methods presented in [Filippazzo et al. (2015)](http://adsabs.harvard.edu/abs/2015ApJ...810..158F).

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
