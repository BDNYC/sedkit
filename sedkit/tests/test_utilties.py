"""A suite of tests for the utilities.py module"""
import unittest
import pytest

import numpy as np
import astropy.units as q

from .. import utilities as u


def test_equivalent():
    """Test for equivalent function"""
    # Positive test
    assert u.equivalent(np.arange(10)*q.um, q.cm)

    # Negative units test
    assert not u.equivalent(np.arange(10)*q.um, q.Jy)

    # Negative dtype test
    assert not u.equivalent(np.arange(10), q.um)


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
