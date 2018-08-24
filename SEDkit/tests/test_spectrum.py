import unittest
import copy

import numpy as np
import astropy.units as q

from .. import synphot as syn
from .. import spectrum as sp


SPEC = [np.linspace(0.8,2.5,200)*q.um, abs(np.random.normal(size=200))*1E-15*q.erg/q.s/q.cm**2/q.AA, abs(np.random.normal(size=200))*1E-16*q.erg/q.s/q.cm**2/q.AA]


class TestSpectrum(unittest.TestCase):
    """Tests for the Spectrum class"""
    def setUp(self):
        # Make Spectrum class for testing
        self.spec = sp.Spectrum(*SPEC)

    def test_Spectrum_data(self):
        """Test that Spectrum is initialized properly"""
        s = copy.copy(self.spec)
        check_shape = self.spec.data.shape == (3, 200)
        self.assertTrue(check_shape)

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
        self.failUnless((s.spectrum[0].unit == wu) &
                        (s.spectrum[1].unit == fu) &
                        (s.spectrum[2].unit == fu))

if __name__ == '__main__':
    unittest.main()