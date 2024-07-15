import unittest

import astropy.units as q
import numpy as np

from .. import uncertainties as un


class TestUnum(unittest.TestCase):
    """Tests for the Unum class"""
    def setUp(self):
        """Setup the tests"""
        # Symmetry
        self.sym = un.Unum(10.1, 0.2)
        self.asym = un.Unum(9.3, 0.08, 0.11)
        self.sym_u = un.Unum(12 * q.um, 0.1 * q.um)

        # Data structures
        self.u1 = un.Unum(10 * q.um, 1 * q.um, n_samples=200)
        self.u2 = un.Unum(10, 1, n_samples=20)
        self.a1 = un.UArray(np.ones(3) * q.um, np.abs(np.random.normal(size=3)) * q.um, n_samples=1000)
        self.a2 = un.UArray(np.ones(3), np.abs(np.random.normal(size=3)), n_samples=1000)
        self.i1 = 5 * q.um
        self.i2 = 5
        self.s1 = np.array([1, 2, 3]) * q.um
        self.s2 = np.array([1, 2, 3])

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

        x = self.u1 + self.u1
        x = self.u1 + self.i1
        x = self.u2 + self.u2
        x = self.u2 + self.i2
        x = self.a1 + self.u1
        x = self.a1 + self.i1
        x = self.a1 + self.s1
        x = self.a2 + self.u2
        x = self.a2 + self.i2
        x = self.a2 + self.s2

    def test_mul(self):
        """Test mul method"""
        x = self.sym * self.asym

        x = self.u1 * self.u1
        x = self.u1 * self.i1
        x = self.u1 * self.i2
        x = self.u2 * self.u2
        x = self.u2 * self.i1
        x = self.u2 * self.i2
        x = self.a1 * self.u1
        x = self.a1 * self.u2
        x = self.a1 * self.i1
        x = self.a1 * self.i2
        x = self.a1 * self.s1
        x = self.a1 * self.s2
        x = self.a1 * self.a1
        x = self.a1 * self.a2
        x = self.a2 * self.u1
        x = self.a2 * self.u2
        x = self.a2 * self.i1
        x = self.a2 * self.i2
        x = self.a2 * self.s1
        x = self.a2 * self.s2
        x = self.a2 * self.a1
        x = self.a2 * self.a2

    def test_sub(self):
        """Test sub method"""
        # Equivalent units
        x = self.sym - self.asym

        # Not equivalent
        try:
            x = self.sym - self.sym_u
        except TypeError:
            pass

        x = self.u1 - self.u1
        x = self.u1 - self.i1
        x = self.u2 - self.u2
        x = self.u2 - self.i2
        x = self.a1 - self.u1
        x = self.a1 - self.i1
        x = self.a1 - self.s1
        x = self.a2 - self.u2
        x = self.a2 - self.i2
        x = self.a2 - self.s2

    def test_pow(self):
        """Test pow method"""
        x = self.sym ** 2

        x = self.u1 ** 2
        x = self.u2 ** 2
        x = self.a1 ** 2
        x = self.a2 ** 2

    def test_truediv(self):
        """Test truediv method"""
        x = self.sym / self.asym

        x = self.u1 / self.u1
        x = self.u1 / self.i1
        x = self.u1 / self.i2
        x = self.u2 / self.u2
        x = self.u2 / self.i1
        x = self.u2 / self.i2
        x = self.a1 / self.u1
        x = self.a1 / self.u2
        x = self.a1 / self.i1
        x = self.a1 / self.i2
        x = self.a1 / self.s1
        x = self.a1 / self.s2
        x = self.a1 / self.a1
        x = self.a1 / self.a2
        x = self.a2 / self.u1
        x = self.a2 / self.u2
        x = self.a2 / self.i1
        x = self.a2 / self.i2
        x = self.a2 / self.s1
        x = self.a2 / self.s2
        x = self.a2 / self.a1
        x = self.a2 / self.a2

    def test_floordiv(self):
        """Test floordiv method"""
        # Equivalent units
        x = self.sym // self.asym

        # Not equivalent
        try:
            x = self.sym // self.sym_u
        except TypeError:
            pass

        x = self.u1 // self.u1
        x = self.u1 // self.i1
        x = self.u2 // self.u2
        x = self.u2 // self.i2
        x = self.a1 // self.u1
        x = self.a1 // self.i1
        x = self.a1 // self.s1
        x = self.a1 // self.a1
        x = self.a2 // self.u2
        x = self.a2 // self.i2
        x = self.a2 // self.s2
        x = self.a2 // self.a2

    def test_log10(self):
        """Test log10 method"""
        x = self.u2.log10()
        x = self.a2.log10()

    def test_polyval(self):
        """Test polyval method"""
        coeffs = [1, 2, 3]
        x = self.sym.polyval(coeffs)

        x = self.u1.polyval([1, 2, 3])
        x = self.u2.polyval([1, 2, 3])
        x = self.a1.polyval([1, 2, 3])
        x = self.a2.polyval([1, 2, 3])

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
