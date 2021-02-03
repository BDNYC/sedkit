"""A suite of tests for the spectrum.py module"""
import unittest
import copy
from pkg_resources import resource_filename

import numpy as np
import astropy.units as q
from svo_filters import Filter

from sedkit import modelgrid as mg
from sedkit import spectrum as sp
from sedkit import utilities as u


class TestSpectrum(unittest.TestCase):
    """Tests for the Spectrum class"""
    def setUp(self):
        """Setup the tests"""
        # Make 'real' spectrum
        wave = np.linspace(0.8, 2.5, 200) * q.um
        flux = u.blackbody_lambda(wave, 3000 * q.K) * q.sr
        self.spec = sp.Spectrum(wave, flux, flux / 100.)

        # Make a flat spectrum
        w1 = np.linspace(0.6, 1, 230) * q.um
        f1 = np.ones_like(w1.value) * q.erg / q.s / q.cm**2 / q.AA
        self.flat1 = sp.Spectrum(w1, f1, f1 * 0.01)

        # Make another flat spectrum
        w2 = np.linspace(0.8, 2, 300) * q.um
        f2 = np.ones_like(w2.value) * q.erg / q.s / q.cm**2 / q.AA
        self.flat2 = sp.Spectrum(w2, f2 * 3, f2 * 0.03)

    def test_data(self):
        """Test that Spectrum is initialized properly"""
        # Test good data loads properly
        self.assertIsNotNone(self.spec.data)

        # Test data with no units throws an error
        args = np.arange(10), np.arange(10)
        self.assertRaises(TypeError, sp.Spectrum, *args)

        # Test that unequal size arrays fail as well
        args = np.arange(10) * q.um, np.arange(9) * q.erg / q.s / q.cm**2 / q.AA
        self.assertRaises(TypeError, sp.Spectrum, *args)

    def test_units(self):
        """Test that units are reassigned properly"""
        s = copy.copy(self.spec)

        # Change the wave units
        wu = q.AA
        s.wave_units = wu
        
        # Change the flux units
        fu = q.W / q.m**2 / q.um
        s.flux_units = fu

        # Make sure the units are being updated
        self.assertEqual(s.spectrum[0].unit, wu)
        self.assertEqual(s.spectrum[1].unit, fu)
        self.assertEqual(s.spectrum[2].unit, fu)

    def test_model_fit(self):
        """Test that a model grid can be fit"""
        # Empty fit results
        self.spec.best_fit = {}

        # Grab the SPL and fit
        spl = mg.SpexPrismLibrary()
        self.spec.best_fit_model(spl)
        self.assertEqual(len(self.spec.best_fit), 1)

        # Test fit works as expected by loading a spectrum then fitting for it
        label = 'Opt:L4'
        spec = spl.get_spectrum(label=label)
        spec.best_fit_model(spl, name='Test', report='SpT')

        # Test MCMC fit
        bt = mg.BTSettl()
        spec = bt.get_spectrum(teff=2456, logg=5.5, meta=0, alpha=0, snr=100)
        spec.mcmc_fit(bt, name='Test', report=True)

    def test_addition(self):
        """Test that spectra are normalized and combined properly"""
        # Add them
        spec3 = self.flat1 + self.flat2

        # Test that non-overlapped spec1 is the same
        seg1 = spec3.flux[spec3.wave < self.flat2.wave.min()]
        self.assertTrue(all([i == 1 for i in seg1]))

        # Test that non-overlapped spec2 is the same
        seg2 = spec3.flux[spec3.wave > self.flat1.wave.max()]
        self.assertTrue(all([i == 3 for i in seg2]))

        # Test that the overlapped area is the mean
        olap = spec3.flux[(spec3.wave > self.flat2.wave.min()) & (spec3.wave < self.flat1.wave.max())]
        self.assertTrue(all([i == 2 for i in olap]))

        # Test that adding None returns the original Spectrum
        spec4 = self.spec + None
        self.assertEqual(spec4.size, self.spec.size)

    def test_convolve_filter(self):
        """Test convolve_filter method"""
        # Spectrum and filter
        spec = self.spec

        # Convolve them
        new_spec = spec.convolve_filter('2MASS.J')

        self.assertNotEqual(spec.data.shape, new_spec.data.shape)

    def test_export(self):
        """Test export method"""
        # Good export
        self.flat1.export('test.txt', header='Foo')

        # Bad dirname
        self.assertRaises(IOError, self.flat1.export, '/foo/bar/baz.txt')

    def test_integrate(self):
        """Test that a spectum is integrated properly"""
        # No nans
        fbol = self.flat1.integrate()
        self.assertAlmostEqual(fbol[0].value, 4000, places=1)

        # With nans
        w1 = np.linspace(0.6, 1, 230) * q.um
        f1 = np.ones_like(w1.value) * q.erg / q.s / q.cm**2 / q.AA
        f1[100: 150] = np.nan
        flat1 = sp.Spectrum(w1, f1, f1 * 0.01)
        fbol = flat1.integrate()
        self.assertAlmostEqual(fbol[0].value, 4000, places=1)
        self.assertNotEqual(str(fbol[1].value), 'nan')

    def test_interpolate(self):
        """Test interpolate method"""
        spec1 = self.flat1
        spec2 = self.flat2.interpolate(spec1)

        # Check wavelength is updated
        self.assertTrue(np.all(spec1.wave == spec2.wave))

    def test_renormalize(self):
        """Test that a spectrum is properly normalized to a given magnitude"""
        # Make a bandpass
        bp = Filter('2MASS.J')
        mag = 10

        # Normalize flat spectrum to it
        s = self.flat2
        norm = s.renormalize(mag, bp, no_spec=True)
        self.assertIsInstance(norm, float)

        # Return Spectrum object
        spec = s.renormalize(mag, bp, no_spec=False)

    def test_resamp(self):
        """Test that the spectrum can be interpolated to new wavelengths"""
        # New wavelength array
        new_wave = np.linspace(9000, 11000, 123) * q.AA

        # Check resampling onto new wavelength array
        new_spec = self.spec.resamp(new_wave)
        self.assertEqual(new_wave.size, new_spec.size)
        self.assertEqual(new_wave.unit, new_spec.wave_units)

        # Test resampling to new resolution
        new_spec = self.spec.resamp(resolution=100)
        self.assertEqual(self.spec.wave_units, new_spec.wave_units)
        self.assertNotEqual(self.spec.size, new_spec.size)

    def test_restore(self):
        """Test restore method"""
        # Smooth the spectrum
        edited_spec = self.spec.smooth(5)

        # Restore it
        restored_spec = edited_spec.restore()

        # Make sure it matched the original
        self.assertTrue(np.all(self.spec.wave == restored_spec.wave))

    def test_synthetic_mag(self):
        """Test the synthetic_magniture and synthetic_flux methods"""
        # Get a spectrum
        s1 = self.spec

        # Test mag
        filt = Filter('2MASS.J')
        mag, mag_unc = s1.synthetic_magnitude(filt)
        self.assertIsInstance(mag, float)
        self.assertIsInstance(mag_unc, float)

        # Test flux
        flx, flx_unc = s1.synthetic_flux(filt, plot=True)
        self.assertIsInstance(flx, q.quantity.Quantity)
        self.assertIsInstance(flx_unc, q.quantity.Quantity)

        # Test out of range band returns None
        filt = Filter('WISE.W4')
        mag, mag_unc = s1.synthetic_magnitude(filt)
        self.assertIsNone(mag)
        self.assertIsNone(mag_unc)

    def test_norm_to_spec(self):
        """Test that a spectrum is properly normalized to another spectrum"""
        # Get two flat spectra
        s1 = self.flat1
        s2 = self.flat2

        # Normalize 1 to 2 and check that they are close
        s3 = s1.norm_to_spec(s2, plot=True)
        self.assertAlmostEqual(np.nanmean(s2.flux), np.nanmean(s3.flux), places=4)
        self.assertNotEqual(s2.size, s3.size)
        self.assertEqual(s1.size, s3.size)

    def test_trim(self):
        """Test that the trim method works"""
        # Test include
        s1 = copy.copy(self.flat1)
        trimmed = s1.trim(include=[(0.8 * q.um, 2 * q.um)], concat=False)
        # self.assertTrue(len(trimmed) == 1)
        # self.assertNotEqual(self.flat1.size, trimmed[0].size)

        # Test exclude
        s1 = copy.copy(self.flat1)
        trimmed = s1.trim(exclude=[(0.8 * q.um, 3 * q.um)], concat=False)
        # self.assertNotEqual(self.flat1.size, trimmed[0].size)

        # Test split
        s1 = copy.copy(self.flat1)
        trimmed = s1.trim(exclude=[(0.8 * q.um, 0.9 * q.um)], concat=False)
        # self.assertTrue(len(trimmed) == 2)
        # self.assertNotEqual(self.flat1.size, trimmed[0].size)

        # Test concat
        s1 = copy.copy(self.flat1)
        trimmed = s1.trim(exclude=[(0.8 * q.um, 0.9 * q.um)], concat=True)
        # self.assertNotEqual(self.flat1.size, trimmed.size)


class TestFileSpectrum(unittest.TestCase):
    """Tests for the FileSpectrum class"""
    def setUp(self):
        """Setup the tests"""
        # Files for testing
        self.fitsfile = resource_filename('sedkit', 'data/Trappist-1_NIR.fits')
        self.txtfile = resource_filename('sedkit', 'data/STScI_Vega.txt')

    def test_fits(self):
        """Test that a fits file can be loaded"""
        spec = sp.FileSpectrum(self.fitsfile, wave_units='um', flux_units='erg/s/cm2/AA')

    def test_txt(self):
        """Test that a txt file can be loaded"""
        spec = sp.FileSpectrum(self.txtfile, wave_units='um', flux_units='erg/s/cm2/AA')


class TestVega(unittest.TestCase):
    """Tests for the Vega class"""
    def setUp(self):
        """Setup the tests"""
        # Make Spectrum class for testing
        self.spec = sp.Vega()

    def test_Vega_data(self):
        """Test that it loaded properly"""
        self.assertEqual(self.spec.name, 'Vega')
        self.assertIsNotNone(self.spec.data)
