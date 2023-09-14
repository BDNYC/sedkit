"""Series of unit tests for the isochrone.py module"""
import unittest

import astropy.units as q

from .. import isochrone as iso
from .. import utilities as u


class TestIsochrone(unittest.TestCase):
    """Tests for the hybrid_solar_age model isochrones"""
    def setUp(self):
        # Make Spectrum class for testing
        self.hsa = iso.Isochrone('hybrid_solar_age')

    def test_evaluate(self):
        """Test the evaluate method"""
        # With uncertainties
        result = self.hsa.evaluate((-4, 0.1), (4*q.Gyr, 0.1*q.Gyr), 'Lbol',
                                   'mass')
        self.assertTrue(isinstance(result, tuple))

        # No xparam uncertainty
        result = self.hsa.evaluate(-4, (4*q.Gyr, 0.1*q.Gyr), 'Lbol', 'mass')
        self.assertTrue(isinstance(result, tuple))

        # No yparam uncertainty
        result = self.hsa.evaluate((-4, 0.1), 4*q.Gyr, 'Lbol', 'mass')
        self.assertTrue(isinstance(result, tuple))

        # No xparam or yparam uncertainties
        result = self.hsa.evaluate(-4, 4*q.Gyr, 'Lbol', 'mass')
        self.assertTrue(isinstance(result, tuple) and result[1] == 0)

    def test_interp(self):
        """Test that the model isochrone can be interpolated"""
        # Successful interpolation
        result = self.hsa.interpolate(-4, 4*q.Gyr, 'Lbol', 'mass')
        self.assertTrue(isinstance(result, u.UNITS))

        # Unsuccessful interpolation
        val = self.hsa.interpolate(-400000, 4*q.Gyr, 'Lbol', 'mass')
        self.assertIsNone(val)

    def test_age_units(self):
        """Test the unit conversions"""
        # Test that the age_units property is updated
        old = self.hsa.age_units
        self.hsa.age_units = q.Myr
        new = self.hsa.age_units
        self.assertEqual(old.to(new), 1000)

        # Test that the ages are updated
        self.assertEqual(self.hsa.ages.unit, new)

    def test_plot(self):
        """Test that the plotting works"""
        plt = self.hsa.plot('Lbol', 'mass')
        dtype = "<class 'bokeh.plotting.figure.Figure'>"
        self.assertTrue(str(type(plt)), dtype)
