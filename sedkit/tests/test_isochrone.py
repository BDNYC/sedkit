"""Series of unit tests for the isochrone.py module"""
import unittest
import pytest
import astropy.units as q
import numpy as np

from .. import isochrone as iso
from .. import utilities as u


@pytest.mark.parametrize('xval,age,xparam,yparam,expected_result', [
    ((-4, 0.1), (4 * q.Gyr, 0.1 * q.Gyr), 'Lbol', 'mass', 0.072),  # With uncertainties
    (-4, (4 * q.Gyr, 0.1 * q.Gyr), 'Lbol', 'mass', 0.072),  # No xparam uncertainty
    ((-4, 0.1), 4 * q.Gyr, 'Lbol', 'mass', 0.072),  # No yparam uncertainty
    (-4, 4 * q.Gyr, 'Lbol', 'mass', 0.020)  # No xparam and yparam uncertainties
])
def test_evaluate(xval, age, xparam, yparam, expected_result):
    # average, lower, upper
    """Test the evaluate method"""
    hsa = iso.Isochrone('hybrid_solar_age')
    result = hsa.evaluate(xval, age, xparam, yparam)
    assert (isinstance(result, tuple)) is True
    assert (np.isclose(result[0].value, expected_result, atol=0.005))
    # assert (np.isclose(result[0],value,))
    # try == first but if it can't happen then use isclose
    # test the value of the y param uncertainties (the three values - lower,average and upper)
    # test lbol to radius
    # test lbol to logg
    # test for different lbols and age/age ranges (cold, mid-temp, warm object)
    # Add the three results for the three different values

class TestIsochrone(unittest.TestCase):
    """Tests for the hybrid_solar_age model isochrones"""
    def setUp(self):
        # Make Spectrum class for testing
        self.hsa = iso.Isochrone('hybrid_solar_age')

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
