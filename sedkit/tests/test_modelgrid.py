import unittest
import copy
import os
from pkg_resources import resource_filename

import astropy.units as q

from .. import modelgrid as mg
from .. import utilities as u


class TestModelGrid(unittest.TestCase):
    """Tests for the ModelGrid class"""
    def setUp(self):

        # Make Model class for testing
        params = ['spty']
        grid = mg.ModelGrid('Test', params, q.AA, q.erg/q.s/q.cm**2/q.AA)

        # Load the model grid
        grid.load(resource_filename('sedkit', 'data/models/atmospheric/spexprismlibrary'))

        # Add numeric spectral type
        grid.index['SpT'] = [u.specType(i.split(',')[0].replace('Opt:','').replace('NIR:',''))[0] for i in grid.index['spty']]

        self.modelgrid = grid

    def test_filter(self):
        """Test the filter metod works"""
        filt = self.modelgrid.filter(SpT=70)
        self.assertEqual(len(filt), 1)

    def test_plot(self):
        """Test that the plot method works"""
        plt = self.modelgrid.plot(SpT=70, draw=False)
        self.assertEqual(str(type(plt)), "<class 'bokeh.plotting.figure.Figure'>")

    def test_save(self):
        """Test the save method works"""
        self.modelgrid.save('test.p')
        os.system('rm test.p')


def test_BTSettl():
    """Test the BTSettl grid"""
    grid = mg.BTSettl()
    assert hasattr(grid, 'name')


def test_SpexPrismLibrary():
    """Test the SpexPrismLibrary grid"""
    grid = mg.SpexPrismLibrary()
    assert hasattr(grid, 'name')


def test_Filippazzo2016():
    """Test the Filippazzo2016 grid"""
    grid = mg.Filippazzo2016()
    assert hasattr(grid, 'name')
