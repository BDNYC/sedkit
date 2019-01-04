"""A suite of tests for the spectrum.py module"""
import unittest
import copy

import numpy as np
import astropy.units as q

from .. import modelgrid as mg
from .. import spectrum as sp
from .. import utilities as u


class TestSpectrum(unittest.TestCase):
    """Tests for the Spectrum class"""
    def setUp(self):
        """Setup the tests"""
        # Generate the spectrum
        wave = np.linspace(0.8, 2.5, 200)*q.um
        flux = u.blackbody_lambda(wave, 3000*q.K)*q.sr

        # Make Spectrum class for testing
        self.spec = sp.Spectrum(wave, flux, flux/100.)

    def test_Spectrum_data(self):
        """Test that Spectrum is initialized properly"""
        # Test good data loads properly
        self.assertIsNotNone(self.spec.data)

        # Test data with no units throws an error
        args = np.arange(10), np.arange(10)
        self.assertRaises(TypeError, sp.Spectrum, *args)

        # Test that unequal size arrays fail as well
        args = np.arange(10)*q.um, np.arange(9)*q.erg/q.s/q.cm**2/q.AA
        self.assertRaises(TypeError, sp.Spectrum, *args)

    def test_Spectrum_units(self):
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
        model = spl.get_spectrum(label=label)
        spec = sp.Spectrum(model[0]*spl.wave_units, model[1]*spl.flux_units)
        spec.best_fit_model(spl)
        self.assertEqual(spec.best_fit[0]['label'], label)


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
