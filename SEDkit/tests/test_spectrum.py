import unittest
import copy
import numpy as np
import astropy.units as q
from .. import synphot as syn
from .. import spectrum as sp
from bokeh.plotting import figure, output_file, show, save

SPEC = [np.linspace(0.8,2.5,200)*q.um, abs(np.random.normal(size=200))*1E-15*q.erg/q.s/q.cm**2/q.AA, abs(np.random.normal(size=200))*1E-16*q.erg/q.s/q.cm**2/q.AA]


def test_spec_norm():
    """Test to see if spectra are being normalized properly
    """
    # Create the spectrum object
    vega = sp.Vega()
    
    # Create a bandpass
    bp = syn.Bandpass('2MASS.J')
    
    # Normalize it to Vega Jmag=-0.177
    norm_spec = vega.renormalize(-0.177, bp)
    
    # Make sure the synthetic mag of the normalized spectrum matches the input Jmag
    Jmag = norm_spec.synthetic_magnitude(bp)
    
    print(Jmag)
    
    # Plot
    fig = figure()
    fig = vega.plot(fig=fig, color='blue')
    norm_spec.plot(fig=fig, color='red')
    show(fig)
    

class SpectrumTests(unittest.TestCase):
    """Tests for the Spectrum class"""
    def __init__(self):
        # Make Spectrum class for testing
        self.spec = sp.Spectrum(*SPEC)

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