import unittest
import copy
from ..spectrum import Spectrum

SPEC = [np.linspace(0.8,2.5,200)*q.um, abs(np.random.normal(size=200))*1E-15*q.erg/q.s/q.cm**2/q.AA, abs(np.random.normal(size=200))*1E-16*q.erg/q.s/q.cm**2/q.AA]

class SpectrumTests(unittest.TestCase):
    """Tests for the Spectrum class"""
    
    # Make Spectrum class for testing
    self.spec = Spectrum(*SPEC)

    def test_Spectrum_data(self):
        """Test that Spectrum is initialized properly"""
        s = copy.copy(spec)
        self.failUnless(s.spectrum == SPEC)

    def test_Spectrum_units(self):
        """Test that units are reassigned properly"""
        s = copy.copy(spec)

        # Change the wave units
        wu = q.AA
        s.wave_units = wu
        
        # Change the flux units
        fu = q.W/q.m**2/q.um
        s.flux_units = fu

        # Make sure the units are being updated
        self.failUnless((s.spectrum[0].unit==wu)&(s.spectrum[1].unit==fu)&(s.spectrum[2].unit==fu))

def main():
    unittest.main()

if __name__ == '__main__':
    main()