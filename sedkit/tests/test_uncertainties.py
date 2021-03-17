import unittest

import astropy.units as q

from .. import uncertainties as un


class TestUnum(unittest.TestCase):
    """Tests for the Unum class"""
    def setUp(self):
        """Setup the tests"""
        self.sym = un.Unum(10.1, 0.2)
        self.asym = un.Unum(9.3, 0.08, 0.11)
        self.sym_u = un.Unum(12 * q.um, 0.1 * q.um)

    def test_attrs(self):
        """Test attributes"""
        x = self.sym
        x.value
        x.quantiles

    def test_add(self):
        """Test add method"""
        # Equivalent units
        x = self.sym + self.asym

        # Not equivalent
        try:
            x = self.sym + self.sym_u
        except TypeError:
            pass

    def test_mul(self):
        """Test mul method"""
        x = self.sym * self.asym

    def test_sub(self):
        """Test sub method"""
        # Equivalent units
        x = self.sym - self.asym

        # Not equivalent
        try:
            x = self.sym - self.sym_u
        except TypeError:
            pass

    def test_pow(self):
        """Test pow method"""
        x = self.sym ** 2

    def test_truediv(self):
        """Test truediv method"""
        x = self.sym / self.asym

    def test_floordiv(self):
        """Test floordiv method"""
        # Equivalent units
        x = self.sym // self.asym

        # Not equivalent
        try:
            x = self.sym // self.sym_u
        except TypeError:
            pass

    def test_polyval(self):
        """Test polyval method"""
        coeffs = [1, 2, 3]
        x = self.sym.polyval(coeffs)

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



