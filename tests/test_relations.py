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

        # Check the data
        self.assertIsNotNone(r.data)
        self.assertTrue(len(r.relations) == 0)
        self.assertTrue(len(r.parameters) > 0)

    def test_add_relation(self):
        """Tests for the add_relation method"""
        # Generate object
        r = rel.DwarfSequence()

        # Derive
        r.add_relation('Teff(Mbol)', 1, yunit=q.K)
        self.assertTrue(len(r.relations) > 0)

        # Bad param name
        args = {'rel_name': 'foo(bar)', 'order': 1}
        self.assertRaises(NameError, r.add_relation, **args)

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

        # Add a relation
        rel_name = 'Teff(Lbol)'
        r.add_relation(rel_name, 9, yunit=q.K, plot=False)

        # Evaluate
        self.assertTrue(isinstance(r.evaluate(rel_name, -2), tuple))

        # Evaluate with no errors and plot
        self.assertTrue(isinstance(r.evaluate(rel_name, -1, plot=True), tuple))

        # Evaluate with errors and plot
        self.assertTrue(isinstance(r.evaluate(rel_name, (-1, 0.1), plot=True), tuple))

    def test_plot(self):
        """Test plot method"""
        # Generate object
        r = rel.DwarfSequence()

        # Add a relation
        rel_name = 'Teff(Lbol)'
        r.add_relation(rel_name, 9, yunit=q.K, plot=False)

        # In relations
        fig = r.plot(rel_name)

        # Not in relations
        fig = r.plot('BCv(Mbol)')