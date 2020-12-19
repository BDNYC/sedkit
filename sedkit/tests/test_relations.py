import copy
from pkg_resources import resource_filename
import unittest

import astropy.units as q
import numpy as np

from .. import relations as rel


class TestSpectralTypeRadius(unittest.TestCase):
    """Tests for the SpectralTypeRadius class"""
    def setUp(self):
        # Make Spectrum class for testing
        self.radius = rel.SpectralTypeRadius()

    def test_get_radius_bounds(self):
        """Test that the get_radius method works"""
        # Test valid input
        rad, unc = self.radius.get_radius(58)
        self.assertTrue(isinstance(rad, q.quantity.Quantity))
        self.assertTrue(isinstance(unc, q.quantity.Quantity))
        
        # Test out of bounds
        self.assertRaises(ValueError, self.radius.get_radius, 104)
        self.assertRaises(ValueError, self.radius.get_radius, -23)

        # Test alphanumeric
        self.assertRaises(ValueError, self.radius.get_radius, 'A0')

    def test_radius_generate(self):
        """Test that the generate method works"""
        # Copy the object
        new_rel = copy.copy(self.radius)

        # Generate new relation with polynomial order 6
        new_rel.generate((2, 2))

        # Check that the order has changed
        self.assertNotEqual(self.radius.MLTY['order'], new_rel.MLTY['order'])
        self.assertNotEqual(self.radius.AFGK['order'], new_rel.AFGK['order'])

        # Check that the polynomial has changed
        old = self.radius.get_radius(62)
        new = new_rel.get_radius(62)
        self.assertNotEqual(new, old)


class TestRelation(unittest.TestCase):
    """Tests for the Relation base class"""
    def setUp(self):
        # Set the file
        self.file = resource_filename('sedkit', 'data/dwarf_sequence.txt')

    def test_init(self):
        """Test class initialization"""
        fill_values = [('...', np.nan), ('....', np.nan), ('.....', np.nan)]

        # Just the file
        r = rel.Relation(self.file, fill_values=fill_values)
        self.assertIsNotNone(r.data)
        self.assertIsNone(r.coeffs)

        # Auto-derive
        r = rel.Relation(self.file, xparam='logL', yparam='Mbol', order=1, fill_values=fill_values)
        self.assertIsNotNone(r.data)
        self.assertIsNotNone(r.coeffs)

        # Add_columns
        columns = {'spt': np.ones_like(r.data['SpT'])}
        r = rel.Relation(self.file, fill_values=fill_values, add_columns=columns)
        self.assertTrue('spt' in r.data.colnames)

    def test_derive(self):
        """Tests for the derive method"""
        # Generate object
        r = rel.DwarfSequence()

        # Derive
        r.derive('logL', 'Mbol', 1)
        self.assertIsNotNone(r.coeffs)
        self.assertEqual(r.order, 1)

        # Bad param name
        args = {'xparam': 'foo', 'yparam': 'Mbol', 'order': 1}
        self.assertRaises(NameError, r.derive, **args)

    def test_add_column(self):
        """Tests for add_column method"""
        # Generate object
        r = rel.DwarfSequence()

        # Bad colname
        args = {'colname': 'Mbol', 'values': np.ones_like(r.data['SpT'])}
        self.assertRaises(KeyError, r.add_column, **args)

        # Bad values length
        args = {'colname': 'foo', 'values': np.ones(3)}
        self.assertRaises(ValueError, r.add_column, **args)

        # Good add_column
        args = {'colname': 'foo', 'values': np.ones_like(r.data['SpT'])}
        r.add_column(**args)
        self.assertTrue('foo' in r.data.colnames)

    def test_evaluate(self):
        """Test evaluate method"""
        # Generate object
        r = rel.DwarfSequence()

        # Check not derived initially
        self.assertFalse(r.derived)
        r.evaluate(5)

        # Try again
        r.derive('logL', 'Mbol', 1)
        self.assertTrue(r.derived)

        # Evaluate
        self.assertTrue(isinstance(r.evaluate(5), tuple))

        # Evaluate with plot
        self.assertTrue(isinstance(r.evaluate(5, plot=True), tuple))

    def test_plot(self):
        """Test plot method"""
        # Generate object
        r = rel.DwarfSequence(xparam='logL', yparam='Mbol', order=1)

        # No explicit params
        fig = r.plot()

        # Explicit params
        fig = r.plot('logL', 'Mbol')