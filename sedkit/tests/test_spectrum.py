"""A suite of tests for the spectrum.py module"""
import unittest
import copy

import numpy as np
import astropy.units as q
from svo_filters import Filter

from .. import modelgrid as mg
from .. import spectrum as sp
from .. import utilities as u


class TestSpectrum(unittest.TestCase):
    """Tests for the Spectrum class"""
    def setUp(self):
        """Setup the tests"""
        # Make 'real' spectrum
        wave = np.linspace(0.8, 2.5, 200)*q.um
        flux = u.blackbody_lambda(wave, 3000*q.K)*q.sr
        self.spec = sp.Spectrum(wave, flux, flux/100.)

        # Make a flat spectrum
        w1 = np.linspace(0.6, 1, 230)*q.um
        f1 = np.ones_like(w1.value)*q.erg/q.s/q.cm**2/q.AA
        self.flat1 = sp.Spectrum(w1, f1, f1*0.01)

        # Make another flat spectrum
        w2 = np.linspace(0.8, 2, 300)*q.um
        f2 = np.ones_like(w2.value)*q.erg/q.s/q.cm**2/q.AA
        self.flat2 = sp.Spectrum(w2, f2*3, f2*0.03)

    def test_data(self):
        """Test that Spectrum is initialized properly"""
        # Test good data loads properly
        self.assertIsNotNone(self.spec.data)

        # Test data with no units throws an error
        args = np.arange(10), np.arange(10)
        self.assertRaises(TypeError, sp.Spectrum, *args)

        # Test that unequal size arrays fail as well
        args = np.arange(10)*q.um, np.arange(9)*q.erg/q.s/q.cm**2/q.AA
        self.assertRaises(TypeError, sp.Spectrum, *args)

    def test_units(self):
        """Test that units are reassigned properly"""
        s = copy.copy(self.spec)

        # Change the wave units
        wu = q.AA
        s.wave_units = wu
        
        # Change the flux units
        fu = q.W/q.m**2/q.um
        s.flux_units = fu

        # Make sure the units are being updated
        self.assertEqual(s.spectrum[0].unit, wu)
        self.assertEqual(s.spectrum[1].unit, fu)
        self.assertEqual(s.spectrum[2].unit, fu)

    def test_model_fit(self):
        """Test that a model grid can be fit"""
        # Empty fit results
        self.spec.best_fit = []

        # Grab the SPL and fit
        spl = mg.SpexPrismLibrary()
        self.spec.best_fit_model(spl)
        self.assertEqual(len(self.spec.best_fit), 1)

        # Test fit works as expected by loading a spectrum then fitting for it
        label = 'Opt:L4'
        spec = spl.get_spectrum(label=label)
        spec.best_fit_model(spl)
        self.assertEqual(spec.best_fit[0]['label'], label)

    def test_addition(self):
        """Test that spectra are normalized and combined properly"""
        # Add them
        spec3 = self.flat1 + self.flat2

        # Test that non-overlapped spec1 is the same
        seg1 = spec3.flux[spec3.wave < self.flat2.wave.min()]
        self.assertTrue(all([i == 1 for i in seg1]))

        # Test that non-overlapped spec2 is the same
        seg2 = spec3.flux[spec3.wave > self.flat1.wave.max()]
        self.assertTrue(all([i == 3 for i in seg2]))

        # Test that the overlapped area is the mean
        olap = spec3.flux[(spec3.wave > self.flat2.wave.min()) & (spec3.wave < self.flat1.wave.max())]
        self.assertTrue(all([i == 2 for i in olap]))

        # Test that adding None returns the original Spectrum
        spec4 = self.spec + None
        self.assertEqual(spec4.size, self.spec.size)

    def test_integrate(self):
        """Test that a spectum is integrated properly"""
        # No nans
        fbol = self.flat1.integrate()
        self.assertAlmostEqual(fbol[0].value, 4000, places=1)

        # With nans
        w1 = np.linspace(0.6, 1, 230)*q.um
        f1 = np.ones_like(w1.value)*q.erg/q.s/q.cm**2/q.AA
        f1[100: 150] = np.nan
        flat1 = sp.Spectrum(w1, f1, f1*0.01)
        fbol = flat1.integrate()
        self.assertAlmostEqual(fbol[0].value, 4000, places=1)
        self.assertNotEqual(str(fbol[1].value), 'nan')

    def test_renormalize(self):
        """Test that a spectrum is properly normalized to a given magnitude"""
        # Make a bandpass
        bp = Filter('2MASS.J')
        mag = 10

        # Normalize flat spectrum to it
        s = self.flat2
        norm = s.renormalize(mag, bp, no_spec=True)
        self.assertIsInstance(norm, float)

    def test_resamp(self):
        """Test that the spectrum can be interpolated to new wavelengths"""
        # New wavelength array
        new_wave = np.linspace(9000, 11000, 123)*q.AA

        # Check resampling onto new wavelength array
        new_spec = self.spec.resamp(new_wave)
        self.assertEqual(new_wave.size, new_spec.size)
        self.assertEqual(new_wave.unit, new_spec.wave_units)

        # Test resampling to new resolution
        new_spec = self.spec.resamp(resolution=100)
        self.assertEqual(self.spec.wave_units, new_spec.wave_units)
        self.assertNotEqual(self.spec.size, new_spec.size)

    def test_norm_to_spec(self):
        """Test that a spectrum is properly normalized to another spectrum"""
        # Get two flat spectra
        s1 = self.flat1
        s2 = self.flat2

        # Normalize 1 to 2 and check that they are close
        s3 = s1.norm_to_spec(s2)
        self.assertAlmostEqual(np.nanmean(s2.flux), np.nanmean(s3.flux), places=4)
        self.assertNotEqual(s2.size, s3.size)
        self.assertEqual(s1.size, s3.size)

    def test_trim(self):
        """Test that the trim method works"""
        # Get a flat spectra
        s1 = self.flat1

        # Successfully trim it
        trimmed = s1.trim([(0.8*q.um, 2*q.um)])
        self.assertNotEqual(self.flat1.size, trimmed.size)

        # Unsuccessfully trim it
        untrimmed = s1.trim([(1.1*q.um, 2*q.um)])
        self.assertEqual(self.flat1.size, untrimmed.size)


class TestVega(unittest.TestCase):
    """Tests for the Vega class"""
    def setUp(self):
        """Setup the tests"""
        # Make Spectrum class for testing
        self.spec = sp.Vega()

    def test_Vega_data(self):
        """Test that it loaded properly"""
        self.assertEqual(self.spec.name, 'Vega')
        self.assertIsNotNone(self.spec.data)
