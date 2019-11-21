"""A suite of tests for the utilities.py module"""
import unittest
import pytest

import numpy as np
import astropy.units as q
import astropy.table as at
from svo_filters import Filter

from .. import utilities as u


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


def test_str2Q():
    """Test str2Q function"""
    qnt = u.str2Q('um', target='A')
    qnt = u.str2Q(None)

# class TestSpectres(unittest.TestCase):
#     """Tests for the spectres function"""
#     def setUp(self):
#         """Setup the tests"""
#         # Generate the spectrum
#         self.wave = np.linspace(0.8, 2.5, 200)
#         self.flux = u.blackbody_lambda(self.wave, 3000*q.K).value
#
#     def test_none(self):
#         """Test no overlap"""
#         # |--- wave ---|
#         #                   |--- new_wave ---|
#         new_wave = np.linspace(2.6, 3, 200)
#         args = new_wave, self.wave, self.flux
#         self.assertRaises(ValueError, u.spectres, *args)
#
#     def test_complete(self):
#         """Complete overlap"""
#         #    |----- wave -----|
#         # |------ new_wave ------|
#         new_wave = np.linspace(0.6, 3, 200)
#         idx = u.idx_overlap(self.wave, new_wave).size
#         binned = u.spectres(new_wave, self.wave, self.flux)
#         self.assertEqual(idx, binned[0].size)
#
#     def test_subset(self):
#         """Subset overlap"""
#         # |-------- wave --------|
#         #    |--- new_wave ---|
#         new_wave = np.linspace(0.9, 2.1, 200)
#         idx = u.idx_overlap(self.wave, new_wave).size
#         binned = u.spectres(new_wave, self.wave, self.flux)
#         self.assertEqual(idx, binned[0].size)
#
#     def test_partial_right(self):
#         """Partial overlap"""
#         # |--- wave ---|
#         #        |--- new_wave ---|
#         new_wave = np.linspace(1, 2.7, 200)
#         idx = u.idx_overlap(self.wave, new_wave).size
#         binned = u.spectres(new_wave, self.wave, self.flux)
#         self.assertEqual(idx, binned[0].size)
#
#     def test_partial_left(self):
#         """Inverted overlap"""
#         #   |--- wave ---|
#         # |--- new_wave ---|
#         new_wave = np.linspace(0.6, 2.7, 200)
#         idx = u.idx_overlap(self.wave, new_wave).size
#         binned = u.spectres(new_wave, self.wave, self.flux)
#         self.assertEqual(idx, binned[0].size)
