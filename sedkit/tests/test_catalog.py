"""A suite of tests for the catalog.py module"""
import unittest
import copy
import os
from pkg_resources import resource_filename

from .. import sed
from .. import catalog


class TestCatalog(unittest.TestCase):
    """Tests for the Catalog class"""
    def setUp(self):

        # Make a test Catalog
        self.cat = catalog.Catalog("Test Catalog")

        # Make SEDs
        self.vega = sed.VegaSED()
        self.sirius = sed.SED('Sirius', spectral_type='A1V', method_list=['find_2MASS', 'find_WISE'])

    def test_add(self):
        """Test catalog adding works"""
        # First catalog
        cat1 = catalog.Catalog("Cat 1")
        cat1.add_SED(self.vega)

        # Second catalog
        cat2 = catalog.Catalog("Cat 2")
        cat2.add_SED(self.vega)

        # Add and check
        cat3 = cat1 + cat2
        self.assertEqual(len(cat3.results), 2)

        # Bad catalog add
        self.assertRaises(TypeError, cat1.__add__, 'foo')

    def test_add_SED(self):
        """Test that an SED is added properly"""
        cat = copy.copy(self.cat)

        # Add the SED
        cat.add_SED(self.vega)
        self.assertEqual(len(cat.results), 1)

        # Remove the SED
        cat.remove_SED('Vega')
        self.assertEqual(len(cat.results), 0)

    def test_export(self):
        """Test that export works"""
        cat = copy.copy(self.cat)
        cat.add_SED(self.vega)

        # Test export
        cat.export()
        os.system('rm -R {}'.format('Test_Catalog'))
        cat.export(zipped=True)
        os.system('rm {}'.format('Test_Catalog.zip'))

        # Bad dir
        self.assertRaises(IOError, cat.export, 'foo')

    def test_filter(self):
        """Test filter method"""
        # Make the catalog
        cat = copy.copy(self.cat)
        cat.add_SED(self.vega)

        # Add another
        s = copy.copy(self.vega)
        s.spectral_type = 50, 0.5
        s.name = 'Foobar'
        cat.add_SED(s)

        # Check there are two SEDs
        self.assertEqual(len(cat.results), 2)

        # Filter so there is only one result
        f_cat = cat.filter('spectral_type', '>30')
        self.assertEqual(len(f_cat.results), 1)

    def test_from_file(self):
        """Test from_file method"""
        cat = self.cat
        file = resource_filename('sedkit', 'data/sources.txt')
        cat.from_file(file)
        self.assertTrue(len(cat.results) > 0)

    def test_get_data(self):
        """Test get_data method"""
        # Make the catalog
        cat = copy.copy(self.cat)
        cat.add_SED(self.vega)
        cat.add_SED(self.sirius)

        # Get the data
        vals = len(cat.get_data('WISE.W1-WISE.W2', 'spectral_type', 'parallax'))
        self.assertEqual(vals, 3)

    def test_get_SED(self):
        """Test get_SED method"""
        # Make the catalog
        cat = copy.copy(self.cat)
        cat.add_SED(self.vega)

        # Get the SED
        s = cat.get_SED('Vega')

        self.assertEqual(type(s), type(self.vega))

    def test_plot(self):
        """Test plot method"""
        # Make the catalog
        cat = copy.copy(self.cat)
        cat.add_SED(self.vega)
        cat.add_SED(self.sirius)

        # Simple plot
        plt = cat.plot('spectral_type', 'parallax')
        self.assertEqual(str(type(plt)), "<class 'bokeh.plotting.figure.Figure'>")

        # Color-color plot
        plt = cat.plot('WISE.W1-WISE.W2', 'WISE.W1-WISE.W2')
        self.assertEqual(str(type(plt)), "<class 'bokeh.plotting.figure.Figure'>")

        # Bad columns
        self.assertRaises(ValueError, cat.plot, 'spectral_type', 'foo')
        self.assertRaises(ValueError, cat.plot, 'foo', 'parallax')

        # Fit polynomial
        cat.plot('spectral_type', 'parallax', order=1)

        # Identify sources
        cat.plot('spectral_type', 'parallax', identify=['Vega'])

    def test_plot_SEDs(self):
        """Test plot_SEDs method"""
        # Make the catalog
        cat = copy.copy(self.cat)
        cat.add_SED(self.vega)

        # Plot the SEDs
        cat.plot_SEDs(['Vega'])
        cat.plot_SEDs('*')

    def test_save_and_load(self):
        """Test save and load methods"""
        # Make the catalog
        cat = copy.copy(self.cat)
        cat.save('test.p')

        # Try to load it
        new_cat = catalog.Catalog("Loaded Catalog")
        new_cat.load('test.p')

        os.system('rm test.p')

    def test_source(self):
        """Test source attribute"""
        # Make the catalog
        cat = copy.copy(self.cat)
        cat.add_SED(self.vega)

        # Check the source
        self.assertEqual(str(type(cat.source)), "<class 'bokeh.models.sources.ColumnDataSource'>")