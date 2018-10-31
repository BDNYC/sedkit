import unittest
import copy

import astropy.units as q

from .. import relations as rel


class TestRelations(unittest.TestCase):
    """Tests for the SpectralTypeRadius class"""
    def setUp(self):
        # Make Spectrum class for testing
        self.radius = rel.SpectralTypeRadius()

    def test_get_radius_bounds(self):
        """Test that the get_radius method works"""
        # Test valid input
        rad, unc = self.radius(58)
        self.assertTrue(isinstance(rad, q.quantity.Quantity))
        self.assertTrue(isinstance(unc, q.quantity.Quantity))
        
        # Test out of bounds
        self.assertRaises(ValueError, self.radius.get_radius(104))
        self.assertRaises(ValueError, self.radius.get_radius(-23))

        # Test alphanumeric
        self.assertRaises(ValueError, self.radius.get_radius('A0'))

    def test_radius_generate(self):
        """Test that the generate method works"""
        # Copy the object
        new_rel = copy.copy(self.radius)

        # Generate new relation with polynomial order 6
        new_rel.generate(6)

        # Check that the order has changed
        self.assertNotEquals(self.radius.order, new_rel.order)

        # Check that the polynomial has changed
        old = self.radius.get_radius(62)
        new = new_rel.get_radius(62)
        self.assertNotEquals(new, old)


if __name__ == '__main__':
    unittest.main()
