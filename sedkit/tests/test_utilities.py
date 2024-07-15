"""A suite of tests for the utilities.py module"""
import copy
from pkg_resources import resource_filename
import pytest
import unittest

import astropy.units as q
import astropy.table as at
from bokeh.plotting import figure
import numpy as np
from svo_filters import Filter

from .. import utilities as u


def test_convert_mag():
    """Test the convert_mag function"""
    mag, mag_unc = 12.3, 0.1

    # AB to Vega
    corr, corr_unc = u.convert_mag('2MASS.J', mag, mag_unc, old='AB', new='Vega')
    assert corr < mag

    # Vega to AB
    corr, corr_unc = u.convert_mag('2MASS.J', mag, mag_unc, old='Vega', new='AB')
    assert corr > mag

    # Other
    corr, corr_unc = u.convert_mag('2MASS.J', mag, mag_unc, old='ST', new='SDSS')
    assert corr == mag


def test_equivalent():
    """Test for equivalent function"""
    # Positive tests
    assert u.equivalent(np.arange(10)*q.um, q.cm)
    assert u.equivalent(2*q.um, q.cm)
    assert u.equivalent([2*q.um, 0.1*q.m], q.cm)

    # Negative units test
    assert not u.equivalent(np.arange(10)*q.um, q.Jy)
    assert not u.equivalent(np.arange(10), q.Jy)
    assert not u.equivalent(1, q.Jy)
    assert not u.equivalent([2*q.um, 2], q.Jy)


def test_errorbars():
    """Test for errorbars function"""
    # Make figure
    fig = figure()

    # Data
    x = np.arange(10)
    y = np.arange(10)

    # Test symmetric x errors
    _ = u.errorbars(fig, x, y, xerr=x / 10.)

    # Test symmetric y errors
    _ = u.errorbars(fig, x, y, yerr=y / 10.)

    # Test asymmetric x errors
    _ = u.errorbars(fig, x, y, xlower=x / 10., xupper=x / 11.)

    # Test asymmetric y errors
    _ = u.errorbars(fig, x, y, ylower=y / 10., yupper=y / 11.)


def test_isnumber():
    """Test for isnumber function"""
    # Positive test
    assert u.isnumber('12.345')

    # Negative test
    assert not u.isnumber('foo')


def test_issequence():
    """Test for proper sequence"""
    # Positive tests
    assert u.issequence((2, 3))
    assert u.issequence([2, 3, 4], length=3)

    # Negative tests
    assert not u.issequence(1)
    assert not u.issequence((1, 2, 3), length=4)


class TestSpectres(unittest.TestCase):
    """Tests for the spectres function"""
    def setUp(self):
        """Setup the tests"""
        # Generate the spectrum
        self.wave = np.linspace(0.8, 2.5, 200)
        self.flux = np.random.normal(loc=1E-13, size=200, scale=1E-14)

    def test_none(self):
        """Test no overlap"""
        # |--- wave ---|
        #                   |--- new_wave ---|
        new_wave = np.linspace(2.6, 3, 200)
        args = new_wave, self.wave, self.flux
        self.assertRaises(ValueError, u.spectres, *args)

    def test_complete(self):
        """Complete overlap"""
        #    |----- wave -----|
        # |------ new_wave ------|
        new_wave = np.linspace(0.6, 3, 200)
        binned = u.spectres(new_wave, self.wave, self.flux)
        self.assertEqual(new_wave.size, binned[0].size)

    def test_subset(self):
        """Subset overlap"""
        # |-------- wave --------|
        #    |--- new_wave ---|
        new_wave = np.linspace(0.9, 2.1, 200)
        binned = u.spectres(new_wave, self.wave, self.flux)
        self.assertEqual(new_wave.size, binned[0].size)

    def test_partial_right(self):
        """Partial overlap"""
        # |--- wave ---|
        #        |--- new_wave ---|
        new_wave = np.linspace(1, 2.7, 200)
        binned = u.spectres(new_wave, self.wave, self.flux)
        self.assertEqual(new_wave.size, binned[0].size)

    def test_partial_left(self):
        """Inverted overlap"""
        #   |--- wave ---|
        # |--- new_wave ---|
        new_wave = np.linspace(0.6, 2.7, 200)
        binned = u.spectres(new_wave, self.wave, self.flux)
        self.assertEqual(new_wave.size, binned[0].size)

    def test_uncertainties(self):
        """Test that it works with uncertainties too"""
        new_wave = np.linspace(0.9, 2.1, 200)

        # Without uncertainties
        binned = u.spectres(new_wave, self.wave, self.flux)
        self.assertEqual(len(binned), 2)

        # With uncertainties
        binned = u.spectres(new_wave, self.wave, self.flux, self.flux/100.)
        self.assertEqual(len(binned), 3)


def test_idx_exclude():
    """Test the idx_exclude function"""
    arr = np.arange(10)
    assert u.idx_exclude(arr, [(2, 4), (5, 8)]).size == 7
    assert u.idx_exclude(arr, [(2, 4)]).size == 9
    assert u.idx_exclude(arr, (2, 4)).size == 9
    assert u.idx_exclude(arr, 2).size == 10


def test_idx_include():
    """Test the idx_include function"""
    arr = np.arange(10)
    assert u.idx_include(arr, [(2, 4), (5, 8)]).size == 3
    assert u.idx_include(arr, [(2, 6)]).size == 3
    assert u.idx_include(arr, (2, 6)).size == 3
    assert u.idx_include(arr, 2).size == 10


def test_idx_overlap():
    """Test idx_overlap function does what I expect"""
    # Base array
    arr = np.arange(10)

    # Test subset overlap
    idx = u.idx_overlap(arr, np.arange(5, 8))
    assert len(idx) == 3

    # Test complete overlap
    idx = u.idx_overlap(arr, np.arange(-5, 20))
    assert len(idx) == 8

    # Test partial right overlap
    idx = u.idx_overlap(arr, np.arange(5, 20))
    assert len(idx) == 4

    # Test partial left overlap
    idx = u.idx_overlap(arr, np.arange(-5, 5))
    assert len(idx) == 4

    # Test no overlap
    idx = u.idx_overlap(arr, np.arange(10, 20))
    assert len(idx) == 0


def test_filter_table():
    """Test filter_table function"""
    # Test table
    table = at.Table([[1, 2, 3], [4, 5, 6], [7, 8, 9]], names=('a', 'b', 'c'))

    # Successful number search
    _ = u.filter_table(table, a=1)
    _ = u.filter_table(table, a='>1')
    _ = u.filter_table(table, a='<3')
    _ = u.filter_table(table, a='>=1')
    _ = u.filter_table(table, a='<=3')

    # Failure

class TestFilterTable(unittest.TestCase):
    """Tests for the filter_table function"""
    def setUp(self):
        """Setup the tests"""
        # Generate the table
        self.table = at.Table([[1, 2, 3], [4, 5, 6], ['meow', 'mix', 'cow']], names=('a', 'b', 'c'))

    def test_number_ranges(self):
        """Test filter by number range"""
        # Copy table
        table = copy.copy(self.table)

        # Successful number search
        self.assertEqual(len(u.filter_table(table, a=1)), 1)
        self.assertEqual(len(u.filter_table(table, a='1')), 1)
        self.assertEqual(len(u.filter_table(table, a='>1')), 2)
        self.assertEqual(len(u.filter_table(table, a='<3')), 2)
        self.assertEqual(len(u.filter_table(table, a='>=1')), 3)
        self.assertEqual(len(u.filter_table(table, a='<=3')), 3)

        # Raise
        self.assertRaises(KeyError, u.filter_table, table, foo=2)

        # Wildcard
        self.assertEqual(len(u.filter_table(table, c='m*')), 2)
        self.assertEqual(len(u.filter_table(table, c='*w')), 2)


def test_finalize_spec():
    """Test finalize_spec function"""
    # Make spectrum
    spec = [np.linspace(1, 2, 100), np.ones(100), np.ones(100)]

    # Run it
    spec_final = u.finalize_spec(spec)


def test_flux2mag():
    """Test flux2mag function"""
    # Test data
    flux = np.random.normal(loc=1E-14, scale=1E-15, size=10)*q.erg/q.s/q.cm**2/q.AA
    unc = flux/100.
    filt = Filter('2MASS.J')

    # Functions
    mag = u.flux2mag(flux, filt)
    mag = u.flux2mag([flux, unc], filt)


def test_fnu2flam():
    """Test fnu2flam function"""
    # Test data
    fnu = np.random.normal(loc=1E-14, scale=1E-15, size=10)*q.Jy
    lam = 1*q.um

    # Functions
    flam = u.fnu2flam(fnu, lam)


def test_goodness():
    """Test goodness function"""
    # Make spectra
    f1, e1 = np.random.normal(size=10)+10, np.abs(np.random.normal(size=10))
    f2, e2 = np.random.normal(size=10)+15, np.abs(np.random.normal(size=10))
    w = np.arange(2, 12)

    # All variables combinations
    _, _ = u.goodness(f1, f2, e1, e2, w)
    _, _ = u.goodness(f1, f2, None, e2, w)
    _, _ = u.goodness(f1, f2, e1, None, w)
    _, _ = u.goodness(f1, f2, e1, e2, None)

    # Failure
    assert pytest.raises(ValueError, u.goodness, f1, f2[:8])


def test_group_spectra():
    """Test the group_spectra function"""
    # Make spectra
    spec1 = [np.arange(10), np.ones(10)]
    spec2 = [np.arange(5, 15), np.ones(10)]
    spec3 = [np.arange(20, 30), np.ones(10)]

    # Two overlapping
    grp1 = u.group_spectra([spec1, spec2])
    assert len(grp1) == 1

    # Two separate
    grp2 = u.group_spectra([spec1, spec3])
    assert len(grp2) == 2

    # Two overlap, one separate
    grp3 = u.group_spectra([spec1, spec2, spec3])
    assert len(grp3) == 2


def test_spectrum_from_fits():
    """Test spectrum_from_fits function"""
    # Get the file
    f = resource_filename('sedkit', '/data/Gl752B_NIR.fits')

    # Get the spectrum
    spec = u.spectrum_from_fits(f)


def test_str2Q():
    """Test str2Q function"""
    qnt = u.str2Q('um', target='A')
    qnt = u.str2Q(None)
