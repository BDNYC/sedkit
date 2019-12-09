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
        path = resource_filename('sedkit', 'data/models/atmospheric/spexprismlibrary')

        # Delete the pickle so the models need to be indexed
        os.system('rm {}'.format(os.path.join(path, 'index.p')))

        # Load the model grid
        grid.load(path)

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


def test_load_model():
    """Test the load_model function"""
    # Get the XML file
    path = 'data/models/atmospheric/spexprismlibrary/spex-prism_2MASPJ0345432+254023_20030905_BUR06B.txt.xml'
    filepath = resource_filename('sedkit', path)

    # Load the model
    meta = mg.load_model(filepath)
    assert isinstance(meta, dict)


def test_load_ModelGrid():
    """Test the load_ModelGrid function"""
    path = 'data/models/atmospheric/Filippazzo2016.p'
    filepath = resource_filename('sedkit', path)

    lmg = mg.load_ModelGrid(filepath)
    assert isinstance(lmg, mg.ModelGrid)


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
