import unittest
import copy
from pkg_resources import resource_filename

import numpy as np
import astropy.units as q
from astropy.modeling.blackbody import blackbody_lambda

from .. import sed
from .. import spectrum as sp
from .. import modelgrid as mg


class TestSED(unittest.TestCase):
    """Tests for the SED class"""
    def setUp(self):

        # Make Spectrum class for testing
        self.WAVE1 = np.linspace(0.8, 2.5, 200)*q.um
        self.FLUX1 = blackbody_lambda(self.WAVE1, 3000*q.K)*q.sr
        SPEC1 = [self.WAVE1, self.FLUX1, self.FLUX1/100.]
        self.spec1 = sp.Spectrum(*SPEC1)

        # Make another
        WAVE2 = np.linspace(21000, 38000, 150)*q.AA
        FLUX2 = blackbody_lambda(WAVE2, 6000*q.K)*q.sr
        SPEC2 = [WAVE2, FLUX2, FLUX2/100.]
        self.spec2 = sp.Spectrum(*SPEC2)

        self.sed = sed.SED(verbose=True)

    def test_add_photometry(self):
        """Test that photometry is added properly"""
        s = copy.copy(self.sed)

        # Add the photometry
        s.add_photometry('2MASS.J', -0.177, 0.206)
        self.assertEqual(len(s.photometry), 1)

        # Now remove it
        s.drop_photometry(0)
        self.assertEqual(len(s.photometry), 0)

    def test_add_photometry_file(self):
        """Test that photometry is added properly from file"""
        s = copy.copy(self.sed)

        # Add the photometry
        f = resource_filename('sedkit', 'data/L3_photometry.txt')
        s.add_photometry_file(f)
        self.assertEqual(len(s.photometry), 8)

    def test_add_spectrum(self):
        """Test that spectra are added properly"""
        s = copy.copy(self.sed)

        # Add a new spectra
        s.add_spectrum(self.spec1)
        s.add_spectrum(self.spec2)

        # Make sure the units are being updated
        self.assertEqual(len(s.spectra), 2)
        self.assertEqual(s.spectra[0]['spectrum'].wave_units,
                         s.spectra[1]['spectrum'].wave_units)

        # Test removal
        s.drop_spectrum(0)
        self.assertEqual(len(s.spectra), 1)

        # Test new spectrum array
        SPEC1 = [self.WAVE1, self.FLUX1, self.FLUX1/100.]
        s.add_spectrum(SPEC1)
        self.assertEqual(len(s.spectra), 2)

        # Test bad spectrum array
        self.assertRaises(TypeError, s.add_spectrum, 'foo')

    def test_attributes(self):
        """Test that the attributes are properly set"""
        s = copy.copy(self.sed)

        # Name
        s.name = b'Gl 752B'

        # Age
        s.age = 4*q.Gyr, 0.1*q.Gyr
        self.assertRaises(TypeError, setattr, s, 'age', 'foo')
        self.assertRaises(TypeError, setattr, s, 'age', (4, 0.1))
        self.assertRaises(TypeError, setattr, s, 'age', (4*q.Jy, 0.1*q.Jy))

        # Dec
        s.dec = 1.2345*q.deg
        self.assertRaises(TypeError, setattr, s, 'dec', 1.2345)

        # RA
        s.ra = 1.2345*q.deg
        self.assertRaises(TypeError, setattr, s, 'ra', 1.2345)

        # Distance
        s.distance = None
        s.distance = 1.2*q.pc, 0.1*q.pc
        self.assertRaises(TypeError, setattr, s, 'distance', (1, 2, 3, 4))
        self.assertRaises(TypeError, setattr, s, 'distance', (1, 4))

        # Parallax
        s.parallax = None
        s.parallax = 1.2*q.mas, 0.1*q.mas
        self.assertRaises(TypeError, setattr, s, 'parallax', (1, 2, 3, 4))
        self.assertRaises(TypeError, setattr, s, 'parallax', (1, 4))

        # Radius
        s.radius = None
        s.radius = 1.2*q.R_jup, 0.1*q.R_jup
        self.assertRaises(TypeError, setattr, s, 'radius', (1, 2, 3, 4))
        self.assertRaises(TypeError, setattr, s, 'radius', (1, 4))

        # Spectral type
        s.spectral_type = 'M3'
        s.radius = None
        s.spectral_type = 20
        s.age = None
        s.spectral_type = 10, 1, 'b', 'V', 'sd'
        self.assertRaises(TypeError, setattr, s, 'spectral_type', ['foo'])

        # Evolutionary Model
        s.evo_model = 'COND03'
        self.assertRaises(ValueError, setattr, s, 'evo_model', 'foo')

        # Flux units
        s.flux_units = q.erg/q.s/q.cm**2/q.AA
        self.assertRaises(TypeError, setattr, s, 'flux_units', q.cm)

        # Wave units
        s.wave_units = q.um
        self.assertRaises(TypeError, setattr, s, 'wave_units', q.Jy)

        # Membership
        s.membership = 'AB Dor'
        s.membership = None
        s.membership = 'foobar'

    def test_no_spectra(self):
        """Test that a purely photometric SED can be creted"""
        s = copy.copy(self.sed)
        s.age = 455*q.Myr, 13*q.Myr
        s.radius = 2.362*q.Rsun, 0.02*q.Rjup
        s.parallax = 130.23*q.mas, 0.36*q.mas
        s.spectral_type = 'A0V'
        s.add_photometry('2MASS.J', -0.177, 0.206)
        s.add_photometry('2MASS.H', -0.029, 0.146)
        s.add_photometry('2MASS.Ks', 0.129, 0.186)
        s.add_photometry('WISE.W1', 1.452, None)
        s.add_photometry('WISE.W2', 1.143, 0.019)
        s.add_photometry('WISE.W3', -0.067, 0.008)
        s.add_photometry('WISE.W4', -0.127, 0.006)

        s.results

        self.assertIsNotNone(s.fbol)

        # Make Wein tail
        s.make_wein_tail(teff=2000*q.K)

        # Radius from spectral type
        s.radius_from_spectral_type('foo')
        s.radius_from_spectral_type()

        # Radius from age
        s.radius_from_age()

    def test_plot(self):
        """Test plotting method"""
        s = copy.copy(self.sed)
        f = resource_filename('sedkit', 'data/L3_photometry.txt')
        s.add_photometry_file(f)
        s.make_sed()

        fig = s.plot(integral=True)

    def test_no_photometry(self):
        """Test that a purely photometric SED can be creted"""
        s = copy.copy(self.sed)
        s.age = 455*q.Myr, 13*q.Myr
        s.radius = 2.362*q.Rsun, 0.02*q.Rjup
        s.parallax = 130.23*q.mas, 0.36*q.mas
        s.spectral_type = 'A0V'
        s.add_spectrum(self.spec1)

        s.results

        self.assertIsNotNone(s.Teff)

    def test_find_methods(self):
        """Test that the find_simbad and find_photometry methods work"""
        s = sed.SED('trappist-1')
        s.find_2MASS()
        s.find_Gaia()

        self.assertNotEqual(len(s.photometry), 0)

    def test_fit_spectrum(self):
        """Test that the SED can be fit by a model grid"""
        # Grab the SPL
        spl = mg.SpexPrismLibrary()

        # Add known spectrum
        s = copy.copy(self.sed)
        spec = spl.get_spectrum()
        s.add_spectrum(spec)

        # Fit with SPL
        s.fit_spectral_type()

    def test_smooth_spectrum(self):
        """Test the smooth_spectrum method"""
        s = copy.copy(self.sed)
        s.add_spectrum(self.spec1)

        # Smooth it
        s.smooth_spectrum(0, 5)

        # Unsmooth it
        s.smooth_spectrum(0, None)

        # Bad beta
        self.assertRaises(ValueError, s.smooth_spectrum, idx=0, beta='foo')

    def test_fit_blackbody(self):
        """Test that the SED can be fit by a blackbody"""
        # Grab the SPL
        spl = mg.SpexPrismLibrary()

        # Add known spectrum
        s = copy.copy(self.sed)
        s.add_spectrum(self.spec1)

        # Add photometry
        f = resource_filename('sedkit', 'data/L3_photometry.txt')
        s.add_photometry_file(f)

        # Fit with SPL
        s.fit_blackbody()

        self.assertTrue(isinstance(s.Teff_bb, (int, float)))

def test_VegaSED():
    """Test the VegaSED class"""
    vega = sed.VegaSED()
