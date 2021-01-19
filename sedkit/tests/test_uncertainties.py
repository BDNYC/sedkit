import unittest

import astropy.units as q

from .. import uncertainties as un


class TestUnum(unittest.TestCase):
    """Tests for the Unum class"""
    def setUp(self):
        """Setup the tests"""
        self.sym = un.Unum(10.1, 0.2)
        self.asym = un.Unum(9.3, 0.08, 0.11)

    def test_attrs(self):
        """Test attributes"""
        x = self.sym
        x.value
        x.quantiles

    def test_add(self):
        """Test add method"""
        x = self.sym + self.asym

    def test_mul(self):
        """Test mul method"""
        x = self.sym * self.asym

    def test_sub(self):
        """Test sub method"""
        x = self.sym - self.asym

    def test_pow(self):
        """Test pow method"""
        x = self.sym ** 2

    def test_truediv(self):
        """Test truediv method"""
        x = self.sym / self.asym

    def test_floordiv(self):
        """Test floordiv method"""
        x = self.sym // self.asym

    def test_plot(self):
        """Test plot method"""
        x = self.sym
        x.plot()

    def test_sample_from_errors(self):
        """Test the sample_from_errors method"""
        # Test symmetric error case
        x = self.sym
        x.sample_from_errors()
        x.sample_from_errors(low_lim=0, up_lim=100)

        # Test asymmetric error case
        y = self.asym
        y.sample_from_errors()
        y.sample_from_errors(low_lim=0, up_lim=100)



