import unittest
import copy
import numpy as np
import astropy.units as q
from .. import sed
from .. import spectrum as sp

SPEC1 = [np.linspace(0.8,2.5,200)*q.um, abs(np.random.normal(size=200))*1E-15*q.erg/q.s/q.cm**2/q.AA, abs(np.random.normal(size=200))*1E-16*q.erg/q.s/q.cm**2/q.AA]
SPEC2 = [np.linspace(21000,38000,150)*q.AA, abs(np.random.normal(size=150))*5E-14*q.erg/q.s/q.cm**2/q.AA, abs(np.random.normal(size=150))*5E-15*q.erg/q.s/q.cm**2/q.AA]

def vega_test():
    s = sed.SED(age=(455*q.Myr,13*q.Myr), radius=(2.362*q.Rsun,0.02*q.Rjup), parallax=(130.23*q.mas,0.36*q.mas), spectral_type='A0V')
    vega = sp.Vega()
    spec1 = [vega.wave[:4000]*q.AA, vega.flux[:4000]*q.erg/q.s/q.cm**2/q.AA]
    spec2 = [vega.wave[3000:]*q.AA, vega.flux[3000:]*q.W/q.m**2/q.um]
    s.add_spectrum(*spec1)
    s.add_spectrum(*spec2)
    s.add_photometry('2MASS.J', -0.177, 0.206)
    s.add_photometry('2MASS.H', -0.029, 0.146)
    s.add_photometry('2MASS.Ks', 0.129, 0.186)
    s.add_photometry('WISE.W1', 1.452, None)
    s.add_photometry('WISE.W2', 1.143, 0.019)
    s.add_photometry('WISE.W3', -0.067, 0.008)
    s.add_photometry('WISE.W4', -0.127, 0.006)
    
    # Teff should be 9602 ± 180 K
    # Mbol should be 0.6
    # Lbol should be 40.12 ± 0.45 Lsun
    # Mass shoulf be 2.135 Msun
    
    return s, s.results

def no_spectra():
    s = sed.SED(age=(455*q.Myr,13*q.Myr), radius=(2.362*q.Rsun,0.02*q.Rjup), parallax=(130.23*q.mas,0.36*q.mas), spectral_type='A0V')
    s.add_photometry('2MASS.J', -0.177, 0.206)
    s.add_photometry('2MASS.H', -0.029, 0.146)
    s.add_photometry('2MASS.Ks', 0.129, 0.186)
    s.add_photometry('WISE.W1', 1.452, None)
    s.add_photometry('WISE.W2', 1.143, 0.019)
    s.add_photometry('WISE.W3', -0.067, 0.008)
    s.add_photometry('WISE.W4', -0.127, 0.006)
    
    # Teff should be 9602 ± 180 K
    # Mbol should be 0.6
    # Lbol should be 40.12 ± 0.45 Lsun
    # Mass shoulf be 2.135 Msun
    
    return s, s.results

class SEDTests(unittest.TestCase):
    """Tests for the SED class"""
    def __init__(self):
    
        # Make Spectrum class for testing
        self.spec1 = sp.Spectrum(*SPEC1)
        self.spec2 = sp.Spectrum(*SPEC2)
    
        self.sed = sed.SED()

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