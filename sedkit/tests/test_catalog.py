import unittest
import copy
from pkg_resources import resource_filename

import numpy as np

from .. import sed
from .. import catalog


class TestCatalog(unittest.TestCase):
    """Tests for the Catalog class"""
    def setUp(self):

        # Make a test Catalog
        self.cat = catalog.Catalog("Test Catalog")

        # Make SEDs
        self.vega = sed.VegaSED()

    def test_add_SED(self):
        """Test that an SED is added properly"""
        cat = copy.copy(self.cat)

        # Add the SED
        cat.add_SED(self.vega)
        self.assertEqual(len(cat.results), 1)

        # Remove the SED
        cat.remove_SED('Vega')
        self.assertEqual(len(cat.results), 0)
