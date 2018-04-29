import unittest
import copy
from ..sed import SED

SPEC1 = [np.linspace(0.8,2.5,200)*q.um, abs(np.random.normal(size=200))*1E-15*q.erg/q.s/q.cm**2/q.AA, abs(np.random.normal(size=200))*1E-16*q.erg/q.s/q.cm**2/q.AA]
SPEC2 = [np.linspace(21000,38000,150)*q.AA, abs(np.random.normal(size=150))*5E-14*q.erg/q.s/q.cm**2/q.AA, abs(np.random.normal(size=150))*5E-15*q.erg/q.s/q.cm**2/q.AA]

class SEDTests(unittest.TestCase):
    """Tests for the SED class"""
    
    # Make Spectrum class for testing
    self.spec1 = Spectrum(*SPEC1)
    self.spec2 = Spectrum(*SPEC2)
    
    self.sed = SED()

    def test_SED_add_spectrum(self):
        """Test that spectra are added properly"""
        s = copy.copy(self.sed)

        # Add a new spectra
        s.add_spectrum(*self.spec1)
        s.add_spectrum(*self.spec2)

        # Make sure the units are being updated
        self.failUnless(s.spectra[0].wave_units==s.spectra[1].wave_units)

def main():
    unittest.main()

if __name__ == '__main__':
    main()