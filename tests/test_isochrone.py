"""Series of unit tests for the isochrone.py module"""
import unittest
import pytest
import astropy.units as q
import numpy as np

from sedkit import isochrone as iso
from sedkit import utilities as u


@pytest.mark.parametrize('xval,age,xparam,yparam,expected_result,expected_result_low,expected_result_up', [
    # Mass
    ((-4, 0.1), (4 * q.Gyr, 0.1 * q.Gyr), 'Lbol', 'mass', 0.072, 0.072, 0.072),  # With uncertainties
    (-4, (4 * q.Gyr, 0.1 * q.Gyr), 'Lbol', 'mass', 0.072, 0.072, 0.072),  # No xparam uncertainty
    ((-4, 0.1), 4 * q.Gyr, 'Lbol', 'mass', 0.072, 0.072, 0.072),  # No yparam uncertainty
    (-4, 4 * q.Gyr, 'Lbol', 'mass', 0.020, 0, 0.058),  # No xparam and yparam uncertainties
    # Radius
    ((-4, 0.1), (4 * q.Gyr, 0.1 * q.Gyr), 'Lbol', 'radius', 0.095, 0.095, 0.095),  # With uncertainties
    (-4, (4 * q.Gyr, 0.1 * q.Gyr), 'Lbol', 'radius', 0.095, 0.095, 0.095),  # No xparam uncertainty
    ((-4, 0.1), 4 * q.Gyr, 'Lbol', 'radius', 0.095, 0.095, 0.095),  # No yparam uncertainty
    (-4, 4 * q.Gyr, 'Lbol', 'radius', 0.045, 0.01, 0.080),  # No xparam and yparam uncertainties
    # Logg
    ((-4, 0.1), (4 * q.Gyr, 0.1 * q.Gyr), 'Lbol', 'logg', 5.345, 5.34, 5.35),  # With uncertainties
    (-4, (4 * q.Gyr, 0.1 * q.Gyr), 'Lbol', 'logg', 5.345, 5.34, 5.35),  # No xparam uncertainty
    ((-4, 0.1), 4 * q.Gyr, 'Lbol', 'logg', 5.345, 5.34, 5.35),  # No yparam uncertainty
    (-4, 4 * q.Gyr, 'Lbol', 'logg', 5.395, 5.36, 5.43),  # No xparam and yparam uncertaintiesd
    # Young age
    ((-4, 0.1), (0.4 * q.Gyr, 0.1 * q.Gyr), 'Lbol', 'mass', 0.0515, 0.045, 0.055),   # mass with uncertainties
    ((-4, 0.1), (0.4 * q.Gyr, 0.1 * q.Gyr), 'Lbol', 'logg', 5.08, 5.01, 5.15),   # logg with uncertainties
    ((-4, 0.1), (0.4 * q.Gyr, 0.1 * q.Gyr), 'Lbol', 'radius', 0.105, 0.10, 0.11),  # radius with uncertainties
    # Old age
    ((-4, 0.1), (9 * q.Gyr, 0.1 * q.Gyr), 'Lbol', 'mass', 0.074, 0.070, 0.080),  # mass with uncertainties
    ((-4, 0.1), (9 * q.Gyr, 0.1 * q.Gyr), 'Lbol', 'logg', 5.345, 5.34, 5.35),  # logg with uncertainties
    ((-4, 0.1), (9 * q.Gyr, 0.1 * q.Gyr), 'Lbol', 'radius', 0.095, 0.09, 0.10)  # radius with uncertainties
])
def test_evaluate(xval, age, xparam, yparam, expected_result, expected_result_low, expected_result_up):
    """Test the evaluate method"""
    hsa = iso.Isochrone('hybrid_solar_age')
    result = hsa.evaluate(xval, age, xparam, yparam)
    average = result[0]  # Average yparam value
    lower = result[0] - result[1]   # Lower yparam value
    upper = result[0] + result[2]   # Upper yparam value
    assert (isinstance(result, tuple)) is True
    if yparam == 'logg':
        assert (np.isclose(average, expected_result, atol=0.006))
        assert (np.isclose(lower, expected_result_low, atol=0.015))
        assert (np.isclose(upper, expected_result_up, atol=0.015))
    else:
        assert (np.isclose(average.value, expected_result, atol=0.005))
        assert (np.isclose(lower.value, expected_result_low, atol=0.01))
        assert (np.isclose(upper.value, expected_result_up, atol=0.01))


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
