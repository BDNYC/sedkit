#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
# !python3
"""
A module to produce spectral energy distributions
and calculate fundamental and atmospheric parameters

Author: Joe Filippazzo, jfilippazzo@stsci.edu
"""

from copy import copy
import os
import shutil
import time
import warnings

import astropy.table as at
import astropy.units as q
import astropy.io.ascii as ii
import astropy.constants as ac
import numpy as np
from astropy.modeling import fitting
from astropy.coordinates import Angle, SkyCoord
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astropy.utils.exceptions import AstropyWarning
from bokeh.io import export_png
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, Range1d, ColumnDataSource
from dustmaps.bayestar import BayestarWebQuery
from svo_filters import svo

from . import utilities as u
from . import spectrum as sp
from . import isochrone as iso
from . import query as qu
from . import relations as rel
from . import modelgrid as mg


Vizier.columns = ["**", "+_r"]
Simbad.add_votable_fields('parallax', 'sptype', 'diameter', 'ids', 'flux(U)', 'flux_error(U)', 'flux_bibcode(U)', 'flux(B)', 'flux_error(B)', 'flux_bibcode(B)', 'flux(V)', 'flux_error(V)', 'flux_bibcode(V)', 'flux(R)', 'flux_error(R)', 'flux_bibcode(R)', 'flux(I)', 'flux_error(I)', 'flux_bibcode(I)')
SptRadius = rel.SpectralTypeRadius()

warnings.simplefilter('ignore', category=AstropyWarning)


class SED:
    """
    A class to construct spectral energy distributions and calculate
    fundamental parameters of stars

    Attributes
    ----------
    Lbol: astropy.units.quantity.Quantity
        The bolometric luminosity [erg/s]
    Lbol_sun: astropy.units.quantity.Quantity
        The bolometric luminosity [L_sun]
    Mbol: float
        The absolute bolometric magnitude
    SpT: str
        The string spectral type
    Teff: astropy.units.quantity.Quantity
        The effective temperature calculated from the SED
    Teff_bb: astropy.units.quantity.Quantity
        The effective temperature calculated from the blackbody fit
    abs_SED: sequence
        The [W, F, E] of the calculated absolute SED
    abs_phot_SED: sequence
        The [W, F, E] of the calculated absolute photometric SED
    abs_spec_SED: sequence
        The [W, F, E] of the calculated absolute spectroscopic SED
    age_max: astropy.units.quantity.Quantity
        The upper limit on the age of the target
    age_min: astropy.units.quantity.Quantity
        The lower limit on the age of the target
    app_SED: sequence
        The [W, F, E] of the calculate apparent SED
    app_phot_SED: sequence
        The [W, F, E] of the calculate apparent photometric SED
    app_spec_SED: sequence
        The [W, F, E] of the calculate apparent spectroscopic SED
    bb_source: str
        The [W, F, E] fit to calculate Teff_bb
    blackbody: astropy.modeling.core.blackbody
        The best fit blackbody function
    distance: astropy.units.quantity.Quantity
        The target distance
    fbol: astropy.units.quantity.Quantity
        The apparent bolometric flux [erg/s/cm2]
    flux_units: astropy.units.quantity.Quantity
        The desired flux density units
    mbol: float
        The apparent bolometric magnitude
    name: str
        The name of the target
    parallaxes: astropy.table.QTable
        The table of parallaxes
    photometry: astropy.table.QTable
        The table of photometry
    radius: astropy.units.quantity.Quantity
        The target radius
    wait: float, int
        The number of seconds to sleep after a query
    spectra: astropy.table.QTable
        The table of spectra
    spectral_type: float
        The numeric spectral type, where 0-99 corresponds to spectral
        types O0-Y9
    spectral_types: astropy.table.QTable
        The table of spectral types
    synthetic_photometry: astropy.table.QTable
        The table of calcuated synthetic photometry
    wave_units: astropy.units.quantity.Quantity
        The desired wavelength units
    """
    def __init__(self, name='My Target', verbose=True, method_list=None, **kwargs):
        """
        Initialize an SED object

        Parameters
        ----------
        name: str (optional)
            A name for the target
        verbose: bool
            Print some diagnostic stuff
        method_list: list (optional)
            Methods to run with arguments as nested dictionaries,
            e.g. ['find_2MASS', 'find_WISE']
        """
        # Print stuff
        self.verbose = verbose
        self.message("SED initialized")
        self.wait = 1

        # Attributes with setters
        self._name = None
        self._ra = None
        self._dec = None
        self._age = None
        self._distance = None
        self._parallax = None
        self._radius = None
        self._spectral_type = None
        self._membership = None
        self._sky_coords = None
        self._evo_model = None
        self.reddening = 0
        self.evo_model = 'parsec12_solar'
        self.SpT = None

        # Dictionary to keep track of references
        self._refs = {}

        # Static attributes
        self.search_radius = 20 * q.arcsec

        # Book keeping
        self.calculated = False
        self.isochrone_radius = False
        self.params = ['name', 'ra', 'dec', 'age', 'membership', 'distance', 'parallax', 'SpT', 'spectral_type', 'fbol', 'mbol', 'Lbol', 'Lbol_sun', 'Mbol', 'Teff', 'Teff_evo', 'Teff_bb', 'logg', 'mass', 'radius']
        self.use_best_fit = False

        # Set the default wavelength and flux units
        self._wave_units = q.um
        self._flux_units = q.erg / q.s / q.cm**2 / q.AA
        self.units = [self._wave_units, self._flux_units, self._flux_units]
        self.min_phot = (999 * q.um).to(self.wave_units)
        self.max_phot = (0 * q.um).to(self.wave_units)
        self.min_spec = (999 * q.um).to(self.wave_units)
        self.max_spec = (0 * q.um).to(self.wave_units)

        # Attributes of arbitrary length
        self.all_names = []
        self.stitched_spectra = []
        self.app_spec_SED = None
        self.app_phot_SED = None
        self.best_fit = {}

        # Make empty spectra table
        spec_cols = ('name', 'spectrum', 'wave_min', 'wave_max', 'wave_bins', 'resolution', 'history', 'ref')
        spec_typs = ('O', 'O', np.float16, np.float16, int, int, 'O', 'O')
        self._spectra = at.QTable(names=spec_cols, dtype=spec_typs)
        for col in ['wave_min', 'wave_max']:
            self._spectra[col].unit = self._wave_units

        # Make empty photometry table
        self.mag_system = 'Vega'
        phot_cols = ('band', 'eff', 'app_magnitude', 'app_magnitude_unc', 'app_flux', 'app_flux_unc', 'abs_magnitude', 'abs_magnitude_unc', 'abs_flux', 'abs_flux_unc', 'bandpass', 'ref')
        phot_typs = ('U16', np.float16, np.float16, np.float16, float, float, np.float16, np.float16, float, float, 'O', 'O')
        self._photometry = at.QTable(names=phot_cols, dtype=phot_typs)
        for col in ['app_flux', 'app_flux_unc', 'abs_flux', 'abs_flux_unc']:
            self._photometry[col].unit = self._flux_units
        self._photometry['eff'].unit = self._wave_units
        self._photometry.add_index('band')

        # Make empty synthetic photometry table
        self._synthetic_photometry = at.QTable(names=phot_cols, dtype=phot_typs)
        for col in ['app_flux', 'app_flux_unc', 'abs_flux', 'abs_flux_unc']:
            self._synthetic_photometry[col].unit = self._flux_units
        self._synthetic_photometry['eff'].unit = self._wave_units
        self._synthetic_photometry.add_index('band')

        # Try to set attributes from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.name = name

        # Make a plot
        self.fig = figure()

        # Empty result attributes
        self.fbol = None
        self.mbol = None
        self.Teff = None
        self.Teff_bb = None
        self.Teff_evo = None
        self.Lbol = None
        self.Mbol = None
        self.Lbol_sun = None
        self.mass = None
        self.logg = None
        self.bb_source = None
        self.blackbody = None

        # Default parameters
        if self.age is None:
            self.age = 6 * q.Gyr, 4 * q.Gyr

        # Run methods
        if method_list is not None:
            self.run_methods(method_list)

    @property
    def abs_SED(self):
        """The flux calibrated SED"""
        if self.app_SED is not None and self.distance is not None:
            return self.app_SED.flux_calibrate(self.distance)
        else:
            return None

    @property
    def abs_spec_SED(self):
        """The flux calibrated spectral SED"""
        if self.app_spec_SED is not None and self.distance is not None:
            return self.app_spec_SED.flux_calibrate(self.distance)
        else:
            return None

    @property
    def abs_specphot_SED(self):
        """The flux calibrated spectro-photometric SED"""
        if self.app_specphot_SED is not None and self.distance is not None:
            return self.app_specphot_SED.flux_calibrate(self.distance)
        else:
            return None

    @property
    def abs_phot_SED(self):
        """The flux calibrated photometric SED"""
        if self.app_phot_SED is not None and self.distance is not None:
            return self.app_spec_SED.flux_calibrate(self.distance)
        else:
            return None

    def add_photometry(self, band, mag, mag_unc=None, system='Vega', ref=None, **kwargs):
        """
        Add a photometric measurement to the photometry table

        Parameters
        ----------
        band: name, svo_filters.svo.Filter
            The bandpass name or instance
        mag: float
            The magnitude
        mag_unc: float (optional)
            The magnitude uncertainty
        system: str
            The magnitude system of the input data, ['Vega', 'AB']
        """
        # Make sure the magnitudes are floats
        if not isinstance(mag, (float, np.float32)):
            raise TypeError("{}: Magnitude must be a float.".format(type(mag)))

        # Check the uncertainty
        if not isinstance(mag_unc, (float, np.float32, type(None), np.ma.core.MaskedConstant)):
            raise TypeError("{}: Magnitude uncertainty must be a float, NaN, or None.".format(type(mag_unc)))

        # # Make NaN if 0
        # if (isinstance(mag_unc, (float, int)) and mag_unc == 0) or isinstance(mag_unc, np.ma.core.MaskedConstant):
        #     mag_unc = np.nan

        # Get the bandpass
        if isinstance(band, str):
            bp = svo.Filter(band)
        elif isinstance(band, svo.Filter):
            bp, band = band, band.name
        else:
            self.message('Not a recognized bandpass: {}'.format(band))

        # Convert to Vega
        mag, mag_unc = u.convert_mag(band, mag, mag_unc, old=system, new=self.mag_system)

        # Convert bandpass to desired units
        bp.wave_units = self.wave_units

        # Drop the current band if it exists
        if band in self.photometry['band']:
            self.drop_photometry(band)

        # Apply the dereddening by subtracting the (bandpass extinction vector)*(source dust column density)
        mag -= bp.ext_vector * self.reddening

        # Make a dict for the new point
        new_photometry = {'band': band, 'eff': bp.wave_eff.astype(np.float16), 'app_magnitude': mag, 'app_magnitude_unc': mag_unc, 'bandpass': bp, 'ref': ref}

        # Add the kwargs
        new_photometry.update(kwargs)

        # Add it to the table
        self._photometry.add_row(new_photometry)
        self.message("Setting {} photometry to {} ({}) with reference '{}'".format(band, mag, mag_unc, ref))

        # Set SED as uncalculated
        self.calculated = False

        # Update photometry max and min wavelengths
        self._calculate_phot_lims()

    def add_photometry_file(self, file):
        """
        Add a table of photometry from an ASCII file that contains the columns 'band', 'magnitude', and 'uncertainty'

        Parameters
        ----------
        file: str
            The path to the ascii file
        """
        # Read the data
        table = ii.read(file)

        # Test to see if columns are present
        cols = ['band', 'magnitude', 'uncertainty']
        if not all([i in table.colnames for i in cols]):
            raise ValueError('File must contain columns {}'.format(cols))

        # Keep relevant cols
        table = table[cols]

        # Add the data to the SED object
        for row in table:

            # Add the magnitude
            self.add_photometry(*row)

    def add_spectrum(self, spectrum, **kwargs):
        """
        Add a new Spectrum object to the SED

        Parameters
        ----------
        spectrum: sequence, sedkit.spectrum.Spectrum
            A sequence of [W,F] or [W,F,E] with astropy units
            or a Spectrum object
        """
        # OK if already a Spectrum
        if hasattr(spectrum, 'spectrum'):
            spec = spectrum

        # or turn it into a Spectrum
        elif isinstance(spectrum, (list, tuple)):

            # Create the Spectrum object
            if len(spectrum) in [2, 3]:
                spec = sp.Spectrum(*spectrum, **kwargs)

            else:
                raise ValueError('Input spectrum must be [W,F] or [W,F,E].')

        # or it's no good
        else:
            raise TypeError('Must enter [W,F], [W,F,E], or a Spectrum object')

        # Convert to SED units
        spec.wave_units = self.wave_units
        spec.flux_units = self.flux_units

        # Add the spectrum object to the list of spectra
        mn = spec.wave_min.astype(np.float16)
        mx = spec.wave_max.astype(np.float16)
        res = int(((mx - mn) / np.nanmean(np.diff(spec.wave))).value)

        # Make sure it's not a duplicate
        if any([(row['wave_min'] == mn) & (row['wave_max'] == mx) & (row['resolution'] == res) for row in self.spectra]):
            self.message("Looks like that {}-{} spectrum is already added. Skipping...".format(mn, mx))

        # If not, add it
        else:

            # Make a dict for the new spectrum
            new_spectrum = {'name': spec.name, 'spectrum': spec, 'history': spec.history, 'wave_min': mn, 'wave_max': mx, 'resolution': res, 'wave_bins': spec.wave.size, 'ref': kwargs.get('ref', spec.ref)}

            # Add the kwargs
            override = {key: val for key, val in kwargs.items() if key in new_spectrum}
            new_spectrum.update(override)

            # Add it to the table
            self._spectra.add_row(new_spectrum)

            # Set SED as uncalculated
            self.calculated = False

            # Update spectra max and min wavelengths
            self._calculate_spec_lims()

            self.message("Spectrum added.")

    def add_spectrum_file(self, file, wave_units=None, flux_units=None, ext=0, survey=None, **kwargs):
        """
        Add a spectrum from an ASCII or FITS file

        Parameters
        ----------
        file: str
            The path to the ascii file
        wave_units: astropy.units.quantity.Quantity
            The wavelength units
        flux_units: astropy.units.quantity.Quantity
            The flux units
        ext: int, str
            The FITS extension name or index
        survey: str (optional)
            The name of the survey, e.g. 'SDSS'
        """
        # Generate a FileSpectrum
        spectrum = sp.FileSpectrum(file, wave_units=wave_units, flux_units=flux_units, ext=ext, survey=survey, **kwargs)

        # Add the data to the SED object
        self.add_spectrum(spectrum, **kwargs)

    @property
    def age(self):
        """
        A property for age
        """
        return self._age

    @age.setter
    def age(self, age):
        """
        A setter for age

        Parameters
        ----------
        age: sequence
            The age and uncertainty in distance units
        """
        if age is None:
            self._age = None
            self._refs.pop('age', None)

        else:

            # If the last value is string, it's the reference
            if isinstance(age[-1], str):
                ref = age[-1]
                age = age[:-1]
            else:
                ref = None

            # Make sure it's a sequence
            if not u.issequence(age, length=[2, 3]):
                raise TypeError("Age must be a sequence of (value, error) or (value, lower_error, upper_error).")

            # Make sure the values are in distance units
            if not all([u.equivalent(a, q.Gyr) for a in age]):
                raise TypeError("Age values must be length units of astropy.units.quantity.Quantity, e.g. 'Gyr'")

            # Set the age!
            self._age = age

            # Set reference
            self._refs['age'] = ref

            self.message("Setting age to {} with reference '{}'".format(self.age, ref))

        # Set SED as uncalculated
        self.calculated = False

    def _calculate_sed(self):
        """
        Stitch the components together and flux calibrate if possible
        """
        # Concatenate points from the Wein tail, app_specphot_SED and RJ tail
        components = [self.wein, self.app_specphot_SED, self.rj]
        ord = np.concatenate([spec.data for spec in components], axis=1)
        idx, = np.where([not np.isnan(i) for i in ord[1]])
        self.app_SED = sp.Spectrum(ord[0][idx] * self.wave_units, ord[1][idx] * self.flux_units, ord[2][idx] * self.flux_units)

        # Calculate Fundamental Params
        self.fundamental_params()

    def _calculate_phot_lims(self):
        """
        Calculate the minimum and maximum wavelengths of the photometry
        """
        # If photometry, grab the max and min
        if len(self.photometry) > 0:
            self.min_phot = np.nanmin(self.photometry['eff']).to(self.wave_units)
            self.max_phot = np.nanmax(self.photometry['eff']).to(self.wave_units)

        # Otherwise reset to infs
        else:
            self.min_phot = (999 * q.um).to(self.wave_units)
            self.max_phot = (0 * q.um).to(self.wave_units)

    def _calculate_spec_lims(self):
        """
        Calculate the minimum and maximum wavelengths of the spectra
        """
        # If spectra, grab the max and min
        if len(self.spectra) > 0:
            self.min_spec = np.nanmin([np.nanmin(i.wave) for i in self.spectra['spectrum']]) * self.wave_units
            self.max_spec = np.nanmax([np.nanmax(i.wave) for i in self.spectra['spectrum']]) * self.wave_units

        # Otherwise reset to infs
        else:
            self.min_spec = (999 * q.um).to(self.wave_units)
            self.max_spec = (0 * q.um).to(self.wave_units)

    def calculate_fbol(self, units=q.erg / q.s / q.cm**2):
        """
        Calculate the bolometric flux of the SED

        Parameters
        ----------
        units: astropy.units.quantity.Quantity
            The target untis for fbol
        """
        # Integrate the SED to get fbol
        self.fbol = self.app_SED.integrate(units=units)

    def calculate_Lbol(self):
        """
        Calculate the bolometric luminosity of the SED
        """
        # Caluclate fbol if not present
        if self.fbol is None:
            self.calculate_fbol()

        # Calculate Lbol
        if self.distance is not None:
            Lbol = (4 * np.pi * self.fbol[0] * self.distance[0]**2).to(q.erg / q.s)
            Lbol_sun = round(np.log10((Lbol / ac.L_sun).decompose().value), 3)

            # Calculate Lbol_unc
            if self.fbol[1] is None:
                Lbol_unc = None
                Lbol_sun_unc = None
            else:
                Lbol_unc = Lbol * np.sqrt((self.fbol[1] / self.fbol[0]).value**2 + (2 * self.distance[1] / self.distance[0]).value**2)
                Lbol_sun_unc = round(abs(Lbol_unc / (Lbol * np.log(10))).value, 3)

            # Update the attributes
            self.Lbol = Lbol, Lbol_unc
            self.Lbol_sun = Lbol_sun, Lbol_sun_unc

    def calculate_mbol(self, L_sun=3.86E26 * q.W, Mbol_sun=4.74):
        """
        Calculate the apparent bolometric magnitude of the SED

        Parameters
        ----------
        L_sun: astropy.units.quantity.Quantity
            The bolometric luminosity of the Sun
        Mbol_sun: float
            The absolute bolometric magnitude of the sun
        """
        # Calculate fbol if not present
        if self.fbol is None:
            self.calculate_fbol()

        # Calculate mbol
        mbol = round(-2.5 * np.log10(self.fbol[0].value) - 11.482, 3)

        # Calculate mbol_unc
        if self.fbol[1] is None:
            mbol_unc = None
        else:
            mbol_unc = round((2.5 / np.log(10)) * (self.fbol[1] / self.fbol[0]).value, 3)

        # Update the attribute
        self.mbol = mbol, mbol_unc

    def calculate_Mbol(self):
        """
        Calculate the absolute bolometric magnitude of the SED
        """
        # Calculate mbol if not present
        if self.mbol is None:
            self.calculate_mbol()

        # Calculate Mbol
        if self.distance is not None:
            Mbol = round(self.mbol[0] - 5 * np.log10((self.distance[0] / 10 * q.pc).value), 3)

            # Calculate Mbol_unc
            if self.fbol[1] is None:
                Mbol_unc = None
            else:
                Mbol_unc = round(np.sqrt(self.mbol[1]**2 + ((2.5 / np.log(10)) * (self.distance[1] / self.distance[0]).value)**2), 3)

            # Update the attribute
            self.Mbol = Mbol, Mbol_unc

    def calculate_synthetic_photometry(self, bandpasses=None):
        """
        Calculate synthetic photometry of all stitched spectra

        Parameters
        ----------
        bandpasses: sequence
            A list of the bandpasses to calculate
        """
        # Set filter list
        all_filters = svo.filters()
        if bandpasses is None:
            bandpasses = all_filters

        # Clear table
        self._synthetic_photometry = self._synthetic_photometry[:0]

        if len(self.stitched_spectra) > 0:

            # Iterate over spectra
            for spec in self.stitched_spectra:

                # and over bandpasses
                for band in bandpasses:

                    # Get the bandpass
                    try:
                        bp = svo.Filter(band)

                        # Calculate the magnitiude
                        mag, mag_unc = spec.synthetic_magnitude(bp)

                        if mag is not None and not np.isnan(mag):

                            # Make a dict for the new point
                            new_photometry = {'band': band, 'eff': bp.wave_eff.astype(np.float16), 'bandpass': bp, 'app_magnitude': mag, 'app_magnitude_unc': mag_unc, 'ref': 'sedkit'}

                            # Add it to the table
                            self._synthetic_photometry.add_row(new_photometry)

                    except IndexError:
                        self.message("'{}' not a supported bandpass. Skipping...")

            # Calibrate the synthetic photometry
            self._calibrate_photometry('synthetic_photometry')

    def calculate_Teff(self):
        """
        Calculate the effective temperature
        """
        # Calculate Teff
        if self.distance is not None and self.radius is not None:
            Teff = np.sqrt(np.sqrt((self.Lbol[0] / (4 * np.pi * ac.sigma_sb * self.radius[0]**2)).to(q.K**4))).astype(int)

            # Calculate Teff_unc
            if self.fbol[1] is None:
                Teff_unc = None
            else:
                Teff_unc = (Teff * np.sqrt((self.Lbol[1] / self.Lbol[0]).value**2 + (2 * self.radius[1] / self.radius[0]).value**2) / 4.).astype(int)

            # Update the attribute
            self.Teff = Teff, Teff_unc

    def _calibrate_photometry(self, name='photometry'):
        """
        Calculate the absolute magnitudes and flux values of all rows in the photometry table
        """
        # Fetch the table (photometry or synthetic_photometry)
        table = getattr(self, '_{}'.format(name))

        # Reset absolute photometry
        table['abs_flux'] = np.nan
        table['abs_flux_unc'] = np.nan
        table['abs_magnitude'] = np.nan
        table['abs_magnitude_unc'] = np.nan

        if len(table) > 0:

            # Update the photometry
            table['eff'] = table['eff'].to(self.wave_units)
            table['app_flux'] = table['app_flux'].to(self.flux_units)
            table['app_flux_unc'] = table['app_flux_unc'].to(self.flux_units)
            table['abs_flux'] = table['abs_flux'].to(self.flux_units)
            table['abs_flux_unc'] = table['abs_flux_unc'].to(self.flux_units)

            # Get the app_mags
            m = np.array(table)['app_magnitude']
            m_unc = np.array(table)['app_magnitude_unc']

            # Calculate app_flux values
            for n, row in enumerate(table):
                app_flux, app_flux_unc = u.mag2flux(row['bandpass'], row['app_magnitude'], sig_m=row['app_magnitude_unc'])
                table['app_flux'][n] = app_flux.to(self.flux_units)
                table['app_flux_unc'][n] = app_flux_unc.to(self.flux_units)

            # Calculate absolute mags
            if self.distance is not None:

                # Calculate abs_mags
                M, M_unc = u.flux_calibrate(m, self.distance[0], m_unc, self.distance[1])
                table['abs_magnitude'] = M
                table['abs_magnitude_unc'] = M_unc

                # Calculate abs_flux values
                for n, row in enumerate(table):
                    abs_flux, abs_flux_unc = u.mag2flux(row['bandpass'], row['abs_magnitude'], sig_m=row['abs_magnitude_unc'])
                    table['abs_flux'][n] = abs_flux.to(self.flux_units)
                    table['abs_flux_unc'][n] = abs_flux_unc.to(self.flux_units)

            if name == 'photometry':

                # Make apparent photometric SED with photometry
                app_cols = ['eff', 'app_flux', 'app_flux_unc']
                phot_array = np.array(getattr(self, name)[app_cols])
                phot_array = phot_array[(getattr(self, name)['app_flux'] > 0) & (getattr(self, name)['app_flux_unc'] > 0)]
                self.app_phot_SED = sp.Spectrum(*[phot_array[i] * Q for i, Q in zip(app_cols, self.units)])

                # Set SED as uncalculated
                self.calculated = False

    def _calibrate_spectra(self):
        """
        Create composite spectra and flux calibrate
        """
        # Reset spectra
        self.app_spec_SED = None

        if len(self.spectra) > 0:

            # Update the spectra
            for spectrum in self.spectra['spectrum']:
                spectrum.flux_units = self.flux_units

            # Group overlapping spectra and stitch together where possible
            # to form piecewise spectrum for flux calibration
            self.stitched_spectra = []
            if len(self.spectra) == 0:
                self.message('No spectra available for SED.')
            if len(self.spectra) == 1:
                self.stitched_spectra = [self.spectra['spectrum'][0]]
            else:
                groups = self.group_spectra(self.spectra['spectrum'])
                for group in groups:
                    spec = group.pop()
                    for g in group:
                        spec = g.norm_to_spec(spec, add=True)
                    self.stitched_spectra.append(spec)

            # Make apparent spectral SED
            if len(self.stitched_spectra) > 0:

                # Renormalize the stitched spectra
                if len(self.photometry) > 0:
                    self.stitched_spectra = [i.norm_to_mags(self.photometry) for i in self.stitched_spectra]

                self.app_spec_SED = np.sum(self.stitched_spectra)

        # Set SED as uncalculated
        self.calculated = False

        # Get synthetic magnitudes
        self.calculate_synthetic_photometry(self.photometry['band'])

    def compare_model(self, modelgrid, fit_to='spec', rebin=True, **kwargs):
        """
        Fit a specific model to the SED by specifying the parameters as kwargs

        Parameters
        ----------
        modelgrid: sedkit.modelgrid.ModelGrid
            The model grid to fit
        fit_to: str
            Which data to fit to, ['spec', 'phot']
        rebin: bool
            Rebin the model to the data
        """
        if not self.calculated:
            self.make_sed()

        # Get the spectrum to fit
        if fit_to == 'phot':
            spec = self.app_phot_SED
            modelgrid = modelgrid.photometry(list(self.photometry['band']))
        else:
            spec = self.app_spec_SED

        # Get the model to fit
        model = modelgrid.get_spectrum(**kwargs)

        if spec is not None:

            if rebin and fit_to == 'spec':
                model = model.resamp(spec.spectrum[0])

            # Fit the model to the SED
            gstat, yn, xn = list(spec.fit(model, wave_units='AA'))
            wave = model.wave * xn
            flux = model.flux * yn

            # Plot the SED with the model on top
            fig = self.plot(output=True)
            fig.line(wave, flux)

            show(fig)

        else:
            self.message("Sorry, could not fit model to SED")

    @property
    def dec(self):
        """
        A property for declination
        """
        return self._dec

    @dec.setter
    def dec(self, dec, dec_unc=None, frame='icrs'):
        """
        Set the declination of the source

        Padecmeters
        ----------
        dec: astropy.units.quantity.Quantity
            The declination
        dec_unc: astropy.units.quantity.Quantity (optional)
            The uncertainty
        frame: str
            The reference frame
        """
        if not isinstance(dec, (q.quantity.Quantity, str)):
            raise TypeError("{}: Cannot interpret dec".format(dec))

        # Make sure it's decimal degrees
        self._dec = Angle(dec)
        if self.ra is not None:
            sky_coords = SkyCoord(ra=self.ra, dec=self.dec, unit=(q.degree, q.degree), frame='icrs')
            self._set_sky_coords(sky_coords, simbad=False)

    @property
    def distance(self):
        """
        A property for distance
        """
        return self._distance

    @distance.setter
    def distance(self, distance):
        """
        A setter for distance

        Parameters
        ----------
        distance: sequence
            The (distance, err) or (distance, lower_err, upper_err)
        """
        if distance is None:
            self._distance = None
            self._parallax = None
            self._refs.pop('distance', None)
            self._refs.pop('parallax', None)

        else:

            # If the last value is string, it's the reference
            if isinstance(distance[-1], str):
                ref = distance[-1]
                distance = distance[:-1]
            else:
                ref = None

            # Make sure it's a sequence
            if not u.issequence(distance, length=[2, 3]):
                raise TypeError("Distance must be a sequence of (value, error) or (value, lower_error, upper_error).")

            # Make sure the values are in distance units
            if not all([u.equivalent(dist, q.pc) for dist in distance]):
                raise TypeError("Distance values must be length units of astropy.units.quantity.Quantity, e.g. 'pc'")

            # Set the distance!
            self._distance = distance

            # Update the parallax
            self._parallax = u.pi2pc(*self.distance, pc2pi=True)

            # Set reference
            self._refs['distance'] = ref
            self._refs['parallax'] = ref

            self.message("Setting distance to {} and parallax to {} with reference '{}'".format(self.distance, self.parallax, ref))

        # Try to calculate reddening
        self.get_reddening()

        # Update the absolute photometry
        self._calibrate_photometry()

        # Update the flux calibrated spectra
        self._calibrate_spectra()

        # Set SED as uncalculated
        self.calculated = False

    def drop_photometry(self, band):
        """
        Drop a photometry by its index or name in the photometry list

        Parameters
        ----------
        band: str, int
            The bandpass name or index to drop
        """
        # Remove the row
        if isinstance(band, str) and band in self.photometry['band']:
            band = self._photometry.remove_row(np.where(self._photometry['band'] == band)[0][0])

        if isinstance(band, int) and band <= len(self._photometry):
            self._photometry.remove_row(band)

        # Update photometry max and min wavelengths
        self._calculate_phot_lims()

        # Set SED as uncalculated
        self.calculated = False

    def drop_spectrum(self, idx):
        """
        Drop a spectrum by its index in the spectra list

        Parameters
        ----------
        idx: int
            The index of the spectrum to drop
        """
        # Remove the row
        self._spectra.remove_row(idx)

        # Update spectra max and min wavelengths
        self._calculate_spec_lims()

        # Set SED as uncalculated
        self.calculated = False

    def edit_spectrum(self, idx, plot=True, restore=False, **kwargs):
        """
        Edit a spectrum inplace by applying sedkit.spectrum.Spectrum methods,
        e.g. smooth, interpolate, trim

        Parameters
        ----------
        idx: int
            The index of the spectrum to edit
        plot: bool
            Plot the old and new spectra
        restore: bool
            Restore the spectrum to the original data
        """
        # Fetch the spectrum
        spec_old = self._spectra[idx]['spectrum']

        # Restore to original data
        if restore:
             spec_new = sp.Spectrum(*spec_old.raw, name=spec_old.name)

        # Or apply some methods
        else:

            # New spectrum
            spec_new = sp.Spectrum(*spec_old.spectrum, name='Edited')

            # Apply keywords as methods with dict values as kwargs
            for method, args in kwargs.items():

                # Check for valid method
                if method in dir(spec_new):

                    # Check if args are a None, list, or dict
                    args = args or {}

                    # Make sure args are a dictionary
                    if not isinstance(args, dict):
                        raise TypeError("{} arguments must be a dictionary".format(method))

                    # Run the method
                    spec_new = getattr(spec_new, method)(**args)

                    # Update the history
                    spec_new.history = spec_old.history
                    spec_new.history.update({method: args})

        if plot:

            # Plot original spectrum
            fig = spec_old.plot(color='blue', alpha=0.5)

            # Plot new spectrum
            fig = spec_new.plot(fig=fig, color='red', alpha=0.5)

            show(fig)

        # Recalculate spec params for SED
        self._calculate_spec_lims()

        # Replace the spectrum
        self.drop_spectrum(idx)
        spec_new.name = spec_old.name
        self.add_spectrum(spec_new)

        # Rearrange spectra
        n_spec = len(self.spectra) - 1
        idx = list(range(0, idx)) + [n_spec] + list(range(idx, n_spec))
        self._spectra = self._spectra[idx]

        # Set SED as uncalculated
        self.calculated = False

    @property
    def evo_model(self):
        """
        A getter for the evolutionary model
        """
        return self._evo_model

    @evo_model.setter
    def evo_model(self, model):
        """
        A setter for the evolutionary model

        Parameters
        ----------
        model: str
            The evolutionary model name
        """
        if model is None:
            self._evo_model = None

        else:

            if model not in iso.EVO_MODELS:
                raise ValueError("Please use an evolutionary model from the list: {}".format(iso.EVO_MODELS))

            self._evo_model = iso.Isochrone(model, verbose=self.verbose)

        # Set as uncalculated
        self.calculated = False

    def export(self, parentdir='.', dirname=None, zipped=False):
        """
        Exports the photometry and results tables and a file of the
        composite spectra

        Parameters
        ----------
        parentdir: str
            The parent directory for the folder or zip file
        dirname: str (optional)
            The name of the exported directory or zip file, default is SED name
        zipped: bool
            Zip the directory
        """
        # Check the parent directory
        if not os.path.exists(parentdir):
            raise IOError('{}: No such target directory'.format(parentdir))

        # Check the target directory
        name = self.name.replace(' ', '_')
        dirname = dirname or name
        dirpath = os.path.join(parentdir, dirname)
        if not os.path.exists(dirpath):
            os.system('mkdir {}'.format(dirpath))
        else:
            raise IOError('{}: Directory already exists'.format(dirpath))

        # Apparent spectral SED
        if self.app_spec_SED is not None:
            specpath = os.path.join(dirpath, '{}_apparent_SED.txt'.format(name))
            header = '{} apparent spectrum (erg/s/cm2/A) as a function of wavelength (um)'.format(name)
            self.app_spec_SED.export(specpath, header=header)

        # Absolute spectral SED
        if self.abs_spec_SED is not None:
            specpath = os.path.join(dirpath, '{}_absolute_SED.txt'.format(name))
            header = '{} absolute spectrum (erg/s/cm2/A) as a function of wavelength (um)'.format(name)
            self.abs_spec_SED.export(specpath, header=header)

        # All photometry
        if self.photometry is not None:
            photpath = os.path.join(dirpath, '{}_photometry.txt'.format(name))
            phot_table = copy(self.photometry)
            for colname in phot_table.colnames:
                phot_table.rename_column(colname, colname.replace('/', '_'))
            phot_table.write(photpath, format='ipac')

        # All synthetic photometry
        if self.synthetic_photometry is not None:
            synpath = os.path.join(dirpath, '{}_synthetic_photometry.txt'.format(name))
            syn_table = copy(self.synthetic_photometry)
            for colname in syn_table.colnames:
                syn_table.rename_column(colname, colname.replace('/', '_'))
            syn_table.write(synpath, format='ipac')

        # All results
        resultspath = os.path.join(dirpath, '{}_results.txt'.format(name))
        res_table = copy(self.results)
        for colname in res_table.colnames:
            res_table.rename_column(colname, colname.replace('/', '_'))
        res_table.write(resultspath, format='ipac')

        # The SED plot
        if self.fig is not None:
            try:
                pltopath = os.path.join(dirpath, '{}_plot.png'.format(name))
                export_png(self.fig, filename=pltopath)
            except:
                # Bokeh dropped support for PhantomJS so image saving is now browser dependent and fails occasionally
                self.message("Could not export SED for {}".format(self.name))

        # zip if desired
        if zipped:
            shutil.make_archive(dirpath, 'zip', dirpath)
            os.system('rm -R {}'.format(dirpath))

    def find_2MASS(self, **kwargs):
        """
        Search for 2MASS data
        """
        self.find_photometry('2MASS', **kwargs)

    def find_Gaia(self, search_radius=None, include=['parallax', 'photometry', 'teff', 'Lbol'], idx=0, **kwargs):
        """
        Search for Gaia data

        Parameters
        ----------
        search_radius: astropy.units.quantity.Quantity
            The radius for the cone search
        catalog: str
            The Vizier catalog to search
        idx: int
            The index of the results to use
        """
        # Get the Vizier catalog
        results = qu.query_vizier('Gaia', target=self.name, sky_coords=self.sky_coords, search_radius=search_radius or self.search_radius, verbose=self.verbose, idx=idx, **kwargs)

        # Parse the record
        if len(results) == len(include):

            if 'parallax' in include:
                self.parallax = results[0][1] * q.mas, results[0][2] * q.mas
                self._refs['parallax'] = results[0][3]

            if 'photometry' in include:
                band, mag, unc, ref = results[1]
                self.add_photometry(band, mag, unc, ref=ref)

            if 'teff' in include:
                self.Teff_Gaia = results[2][1] * q.K

            if 'Lbol' in include:
                self.Lbol_Gaia = results[3][1] or None

        # Pause to prevent ConnectionError with astroquery
        time.sleep(self.wait)

    def find_PanSTARRS(self, **kwargs):
        """
        Search for PanSTARRS data
        """
        self.find_photometry('PanSTARRS', **kwargs)

    def find_photometry(self, catalog, col_names=None, target_names=None, search_radius=None, idx=0, **kwargs):
        """
        Search Vizier for photometry in the given catalog

        Parameters
        ----------
        catalog: str
            The Vizier catalog address, e.g. 'II/246/out'
        col_names: sequence
            The list of column names to treat as bandpasses
        target_names: sequence (optional)
            The list of renamed columns, must be the same length as col_names
        search_radius: astropy.units.quantity.Quantity
            The search radius for the Vizier query
        idx: int
            The index of the record to use if multiple Vizier results
        """
        # Get the Vizier catalog
        results = qu.query_vizier(catalog, col_names=col_names, target_names=target_names, target=self.name, sky_coords=self.sky_coords, search_radius=search_radius or self.search_radius, verbose=self.verbose, idx=idx, **kwargs)

        # Parse the record
        for result in results:

            # Get result
            band, mag, unc, ref = result

            # Ensure Vegamag
            system = 'AB' if 'SDSS' in band else 'Vega'

            self.add_photometry(band, mag, unc, ref=ref, system=system)

        # Pause to prevent ConnectionError with astroquery
        time.sleep(self.wait)

    def find_SDSS(self, **kwargs):
        """
        Search for SDSS data
        """
        self.find_photometry('SDSS', **kwargs)

    def find_SDSS_spectra(self, surveys=['optical', 'apogee'], search_radius=None, **kwargs):
        """
        Search for SDSS spectra
        """
        if 'optical' in surveys:

            # Query spectra
            data, ref, header = qu.query_SDSS_optical_spectra(self.sky_coords, verbose=self.verbose, radius=search_radius or self.search_radius, **kwargs)

            # Add the spectrum to the SED
            if data is not None:
                self.add_spectrum(data, ref=ref, header=header)

            # Pause to prevent ConnectionError with astroquery
            time.sleep(self.wait)

        if 'apogee' in surveys:

            # Query spectra
            data, ref, header = qu.query_SDSS_apogee_spectra(self.sky_coords, verbose=self.verbose, search_radius=search_radius or self.search_radius, **kwargs)

            # Add the spectrum to the SED
            if data is not None:
                self.add_spectrum(data, ref=ref, header=header)

            # Pause to prevent ConnectionError with astroquery
            time.sleep(self.wait)

    def find_Simbad(self, search_radius=None, idx=0):
        """
        Search for a Simbad record to retrieve designations, coordinates,
        parallax, radius, and spectral type information

        Parameters
        ----------
        search_radius: astropy.units.quantity.Quantity
            The radius for the cone search
        idx: int
            The index of the result to use
        """
        # Check for coordinates
        if isinstance(self.sky_coords, SkyCoord):

            # Search Simbad by sky coords
            rad = search_radius or self.search_radius
            viz_cat = Simbad.query_region(self.sky_coords, radius=rad)
            crit = self.sky_coords

        elif self.name is not None and self.name != 'My Target':

            viz_cat = Simbad.query_object(self.name)
            crit = self.name

        else:
            return

        # Parse the record and save the names
        if viz_cat is not None and len(viz_cat) > 0:

            # Print info
            n_rec = len(viz_cat)
            self.message("{} record{} for {} found in Simbad.".format(n_rec, '' if n_rec == 1 else 's', crit))

            # Choose the record
            obj = viz_cat[idx]

            # Get the list of names
            main_ID = obj['MAIN_ID']
            self.all_names += obj['IDS'].split('|')

            # Remove duplicates
            self.all_names = list(set(self.all_names))

            # Set the name
            if self.name is None:
                self._name = main_ID

            # TODO: Discovery paper bibcode?
            # self._refs['discovery'] = obj['COO_BIBCODE']

            # Save the coordinates
            if self.sky_coords is None:
                sky_coords = tuple(viz_cat[0][['RA', 'DEC']])
                sky_coords = SkyCoord(ra=sky_coords[0], dec=sky_coords[1], unit=(q.degree, q.degree), frame='icrs')
                self._set_sky_coords(sky_coords, simbad=False)

            # Check for a parallax
            if not hasattr(obj['PLX_VALUE'], 'mask'):
                self.parallax = obj['PLX_VALUE'] * q.mas, obj['PLX_ERROR'] * q.mas, obj['PLX_BIBCODE']

            # Check for a spectral type
            if not hasattr(obj['SP_TYPE'], 'mask'):
                try:
                    self.spectral_type = obj['SP_TYPE'], obj['SP_BIBCODE']
                except IndexError:
                    pass

            # Check for a radius
            if not hasattr(obj['Diameter_diameter'], 'mask'):
                du = q.Unit(obj['Diameter_unit'])
                self.radius = obj['Diameter_diameter'] / 2. * du, obj['Diameter_error'] * du, obj['Diameter_bibcode']

            # Check for UBVRI photometry
            for band, label in zip(['Generic/Johnson.U', 'Generic/Johnson.B', 'Generic/Johnson.V', 'Cousins.R', 'Cousins.I'], ['U', 'B', 'V', 'R', 'I']):
                flx = obj['FLUX_{}'.format(label)]
                if not hasattr(flx, 'mask'):
                    err = np.nan if hasattr(obj['FLUX_ERROR_{}'.format(label)], 'mask') else obj['FLUX_ERROR_{}'.format(label)]
                    self.add_photometry(band, flx, err, obj['FLUX_BIBCODE_{}'.format(label)])

        # Pause to prevent ConnectionError with astroquery
        time.sleep(self.wait)

    def find_WISE(self, **kwargs):
        """
        Search for WISE data
        """
        self.find_photometry('WISE', **kwargs)

    def fit_blackbody(self, fit_to='app_phot_SED', Teff_init=4000, epsilon=0.0001, acc=0.05, trim=[], norm_to=[]):
        """
        Fit a blackbody curve to the data

        Parameters
        ----------
        fit_to: str
            The attribute name of the [W, F, E] to fit
        initial: int
            The initial guess
        epsilon: float
            The step size
        acc: float
            The acceptible error
        """
        if not self.calculated:
            self.make_sed()

        # Get the data and remove NaNs
        data = u.scrub(getattr(self, fit_to).data)

        # Trim manually
        if isinstance(trim, (list, tuple)):
            for mn, mx in trim:
                try:
                    idx, = np.where((data[0] < mn) | (data[0] > mx))
                    if any(idx):
                        data = [i[idx] for i in data]
                except TypeError:
                    self.message('Please provide a list of (lower, upper) bounds to exclude from the fit, e.g. [(0, 0.8)]')

        # Initial guess
        if self.Teff is not None:
            teff = self.Teff[0].value
        else:
            teff = Teff_init
        init = u.blackbody(temperature=teff)

        # Fit the blackbody
        fit = fitting.LevMarLSQFitter()
        norm = np.nanmax(data[1])
        weight = norm / data[2]
        if acc is None:
            acc = np.nanmax(weight)
        bb_fit = fit(init, data[0], data[1] / norm, weights=weight,
                     epsilon=epsilon, acc=acc, maxiter=500)

        # Store the results
        try:
            self.Teff_bb = int(bb_fit.temperature.value)
            self.bb_source = fit_to
            self.bb_norm_to = norm_to

            # Make the blackbody spectrum
            wav = np.linspace(0.2, 22., 400) * self.wave_units
            bb = sp.Blackbody(wav, self.Teff_bb * q.K, radius=self.radius, distance=self.distance)
            bb = bb.norm_to_mags(self.photometry[-3:], include=norm_to)
            self.blackbody = bb

            self.message('Blackbody fit: {} K'.format(self.Teff_bb))

        except IOError:
            self.message('No blackbody fit.')

    def fit_modelgrid(self, modelgrid, fit_to='spec', name=None, mcmc=False, **kwargs):
        """
        Fit a model grid to the composite spectra

        Parameters
        ----------
        modelgrid: sedkit.modelgrid.ModelGrid
            The model grid to fit
        name: str
            A name for the fit
        mcmc: bool
            Use MCMC fitting routine
        """
        if not self.calculated:
            self.make_sed()

        # Determine a name
        if name is None:
            name = modelgrid.name

        # Get the spectrum to fit
        if fit_to == 'phot':
            spec = self.app_phot_SED
            modelgrid = modelgrid.photometry(list(self.photometry['band']), weight=False if mcmc else True)
        else:
            spec = self.app_spec_SED

        if spec is not None:

            # Determine if there is spectral coverage
            model_min, model_max = modelgrid.wave_limits
            spec_min, spec_max = spec.wave_min, spec.wave_max
            if model_max < spec_min or model_min > spec_max:
                self.message("Could not fit model grid {} to the SED. No overlapping wavelengths.".format(modelgrid.name))

            if mcmc:
                spec.mcmc_fit(modelgrid, name=name, **kwargs)
            else:
                spec.best_fit_model(modelgrid, name=name, **kwargs)

            # Save the best fit
            self.best_fit[name] = spec.best_fit[name]
            setattr(self, name, self.best_fit[name]['label'])

            self.message('Best fit {}: {}'.format(name, self.best_fit[name]['label']))

            # Make the SED in case use_best_fit is True
            self.make_sed()

        else:
            self.message("Could not fit model grid {} to the '{}' SED. No {} to fit.".format(modelgrid.name, fit_to, 'photometry' if fit_to == 'phot' else 'spectrum'))

    def fit_spectral_type(self):
        """
        Fit the spectral SED to a catalog of spectral standards
        """
        # Choose the spectral type library
        # TODO: Add a choice of SPT libraries
        spl = mg.SpexPrismLibrary()

        # Run the fit
        self.fit_modelgrid(spl)

    @property
    def flux_units(self):
        """
        A property for flux_units
        """
        return self._flux_units

    @flux_units.setter
    def flux_units(self, flux_units):
        """
        A setter for flux_units

        Parameters
        ----------
        flux_units: astropy.units.quantity.Quantity
            The astropy units of the SED wavelength
        """
        # Make sure it's a flux density
        if not u.equivalent(flux_units, q.erg / q.s / q.cm**2 / q.AA):
            raise TypeError("{}: flux_units must be a unit of flux density, e.g. 'erg/s/cm2/A'".format(flux_units))

        # fnu2flam(f_nu, lam, units=q.erg / q.s / q.cm**2 / q.AA)

        # Set the flux_units!
        self._flux_units = flux_units
        self.units = [self._wave_units, self._flux_units, self._flux_units]

        # Recalibrate the data
        self._calibrate_photometry()
        self._calibrate_spectra()

    def from_database(self, db, rename_bands=u.PHOT_ALIASES, **kwargs):
        """
        Load the data from an astrodbkit.astrodb.Database

        Parameters
        ----------
        db: astrodbkit.astrodb.Database
            The database instance to query
        rename_bands: dict
            A lookup dictionary to map database bandpass
            names to sedkit required bandpass names,
            e.g. {'2MASS_J': '2MASS.J', 'WISE_W1': 'WISE.W1'}

        Example
        -------
        from sedkit import SED
        from astrodbkit.astrodb import Database
        db = Database('/Users/jfilippazzo/Documents/Modules/BDNYCdb/bdnyc_database.db')
        s = SED()
        s.from_database(db, source_id=710, photometry='*', spectra=[1639], parallax=49)
        s.spectral_type = 'M9'
        s.fit_spectral_type()
        print(s.results)
        s.plot(draw=True)
        """
        # Check that astrodbkit is imported
        if not hasattr(db, 'query'):
            raise TypeError("Please provide an astrodbkit.astrodb.Database object to query.")

        # Get the metadata
        if 'source_id' in kwargs:

            if not isinstance(kwargs['source_id'], int):
                raise TypeError("'source_id' must be an integer")

            self.source_id = kwargs['source_id']
            source = db.query("SELECT * FROM sources WHERE id=?", (self.source_id, ), fmt='dict', fetch='one')

            # Set the name
            self.name = source.get('designation', source.get('names', self.name))

            # Set the coordinates
            ra = source.get('ra') * q.deg
            dec = source.get('dec') * q.deg
            self.sky_coords = SkyCoord(ra=ra, dec=dec, frame='icrs')

        # Get the photometry
        if 'photometry' in kwargs:

            if kwargs['photometry'] == '*':
                phot_q = "SELECT * FROM photometry WHERE source_id={}".format(self.source_id)
                phot = db.query(phot_q, fmt='dict')

            elif isinstance(kwargs['photometry'], (list, tuple)):
                phot_ids = tuple(kwargs['photometry'])
                phot_q = "SELECT * FROM photometry WHERE id IN ({})".format(', '.join(['?'] * len(phot_ids)))
                phot = db.query(phot_q, phot_ids, fmt='dict')

            else:
                raise TypeError("'photometry' must be a list of integers or '*'")

            # Add the bands
            for row in phot:

                # Make sure the bandpass name is right
                if row['band'] in rename_bands:
                    row['band'] = rename_bands.get(row['band'])

                self.add_photometry(row['band'], row['magnitude'], row['magnitude_unc'])

        # Get the parallax
        if 'parallax' in kwargs:

            if not isinstance(kwargs['parallax'], int):
                raise TypeError("'parallax' must be an integer")

            plx = db.query("SELECT * FROM parallaxes WHERE id=?", (kwargs['parallax'], ), fmt='dict', fetch='one')

            # Add it to the object
            self.parallax = plx['parallax'] * q.mas, plx['parallax_unc'] * q.mas

        # Get the spectral type
        if 'spectral_type' in kwargs:

            if not isinstance(kwargs['spectral_type'], int):
                raise TypeError("'spectral_type' must be an integer")

            spt_id = kwargs['spectral_type']
            spt = db.query("SELECT * FROM spectral_types WHERE id=?", (spt_id, ), fmt='dict', fetch='one')

            # Add it to the object
            spectral_type = spt.get('spectral_type')
            spectral_type_unc = spt.get('spectral_type_unc', 0.5)
            gravity = spt.get('gravity')
            lum_class = spt.get('lum_class', 'V')
            prefix = spt.get('prefix')

            # Add it to the object
            self.spectral_type = spectral_type, spectral_type_unc, gravity, lum_class, prefix

        # Get the spectra
        if 'spectra' in kwargs:

            if kwargs['spectra'] == '*':
                spec_q = "SELECT * FROM spectra WHERE source_id={}".format(self.source_id)
                spec = db.query(spec_q, fmt='dict')

            elif isinstance(kwargs['spectra'], (list, tuple)):
                spec_ids = tuple(kwargs['spectra'])
                spec_q = "SELECT * FROM spectra WHERE id IN ({})".format(', '.join(['?'] * len(spec_ids)))
                spec = db.query(spec_q, spec_ids, fmt='dict')

            else:
                raise TypeError("'spectra' must be a list of integers or '*'")

            # Add the spectra
            for row in spec:

                # Make the Spectrum object
                name = row['filename']
                dat = row['spectrum'].data
                if len(dat) == 3:
                    wav, flx, unc = dat
                else:
                    wav, flx, *other = dat
                    unc = np.ones_like(wav) * np.nan
                wave_unit = u.str2Q(row['wavelength_units'])

                # Guess the wave unit if missing
                if str(wave_unit) == '':
                    wave_unit = q.AA if np.nanmean(wav) > 100 else q.um

                if row['flux_units'].startswith('norm') or row['flux_units'] == '':
                    flux_unit = self.flux_units
                else:
                    flux_unit = u.str2Q(row['flux_units'])

                # Add the spectrum to the object
                spectrum = [wav * wave_unit, flx * flux_unit]
                if np.all([np.isnan(i) for i in unc]):
                    self.add_spectrum(spectrum, snr=50, name=name)
                else:
                    self.add_spectrum(spectrum + [unc * flux_unit], name=name)

    def fundamental_params(self, **kwargs):
        """
        Calculate the fundamental parameters of the current SED
        """
        # Calculate bolometric luminosity (dependent on fbol and distance)
        self.calculate_Lbol()
        self.calculate_Mbol()

        # Interpolate surface gravity, mass and radius from isochrones
        if self.Lbol_sun is not None:

            if self.Lbol_sun is None:
                self.message('Lbol={0.Lbol}. Uncertainties are needed to estimate Teff, radius, surface gravity, and mass.'.format(self))

            else:
                if self.radius is None and self.evo_model is not None:
                    self.radius_from_age()
                if self.logg is None and self.evo_model is not None:
                    self.logg_from_age()
                if self.mass is None and self.evo_model is not None:
                    self.mass_from_age()
                if self.evo_model is not None:
                    self.teff_from_age()

        # Calculate Teff (dependent on Lbol, distance, and radius)
        self.calculate_Teff()

    def get_reddening(self, version='bayestar2019'):
        """
        Calculate the reddening from the Bayestar17 dust map
        """
        # Check for distance and coordinates
        if self.sky_coords is not None:

            # Get galactic coordinates
            if self.distance is not None:
                gal_coords = SkyCoord(self.sky_coords.galactic, frame='galactic', distance=self.distance[0])
            else:
                gal_coords = SkyCoord(self.sky_coords.galactic, frame='galactic')

            # Query web server
            try:
                bayestar = BayestarWebQuery(version=version)
                red = bayestar(gal_coords, mode='random_sample')
                ref = '2018JOSS....3..695M'

                # Set the attribute
                if not np.isinf(red) and not np.isnan(red) and red >= 0:
                    self.reddening = red
                    self._refs['reddening'] = ref
                    self.message("Setting interstellar reddening to {} with reference '{}'".format(red, ref))

            except:
                self.message("There was a problem determining the interstellar reddening. Setting to 0. You can manually set this with the 'reddening' attribute.")
                self._refs.pop('reddening', None)

    @staticmethod
    def group_spectra(spectra):
        """
        Puts a list of *spectra* into groups with overlapping wavelength arrays
        """
        groups, idx = [], []
        for N, S in enumerate(spectra):
            if N not in idx:
                group, idx = [S], idx + [N]
                for n, s in enumerate(spectra):
                    if n not in idx and any(np.where(np.logical_and(S.wave < s.wave[-1], S.wave > s.wave[0]))[0]):
                        group.append(s), idx.append(n)
                groups.append(group)
        return groups

    @property
    def info(self):
        """
        Print all the SED info
        """
        for attr in dir(self):
            if not attr.startswith('_') and attr not in ['info', 'results'] and not callable(getattr(self, attr)):
                val = getattr(self, attr)
                self.message('{0: <25}= {1}{2}'.format(attr, '\n' if isinstance(val, at.QTable) else '', val))

    @property
    def logg(self):
        """Getter for surface gravity"""
        return self._logg

    @logg.setter
    def logg(self, logg):
        """
        A setter for surface gravity

        Parameters
        ----------
        logg: sequence
            The logg and uncertainty in cgs units
        """
        if logg is None:
            self._logg = None
            self._refs.pop('logg', None)

        else:

            # If the last value is string, it's the reference
            if isinstance(logg[-1], str):
                ref = logg[-1]
                logg = logg[:-1]
            else:
                ref = None

            # Make sure it's a sequence
            if not u.issequence(logg, length=[2, 3]):
                raise TypeError("log(g) must be a sequence of (value, error) or (value, lower_error, upper_error).")

            # Make sure the values are unitless
            if any([hasattr(l, 'unit') for l in logg]):
                raise TypeError("log(g) values must be unitless magnitudes")

            # Set the logg!
            self._logg = logg

            # Set reference
            self._refs['logg'] = ref

            self.message("Setting log(g) to {} with reference '{}'".format(self.logg, ref))

        # Set SED as uncalculated
        self.calculated = False

    def logg_from_age(self, plot=False):
        """
        Estimate the surface gravity from model isochrones given an age and Lbol
        """
        if self.age is not None and self.Lbol_sun is not None:

            # Default
            logg = None

            # Check for uncertainties
            if self.Lbol_sun[1] is None:
                self.message('Lbol={0.Lbol}. Uncertainties are needed to calculate the surface gravity.'.format(self))
            else:
                logg = self.evo_model.evaluate(self.Lbol_sun, self.age, 'Lbol', 'logg', plot=plot)

            # Print a message if None
            if logg is None:
                self.message("Could not calculate surface gravity.")

            # Store the value
            self.logg = [i.round(2) for i in logg] if logg is not None else logg

        else:
            self.message('Lbol={0.Lbol} and age={0.age}. Both are needed to calculate the surface gravity.'.format(self))

    def make_rj_tail(self, teff=3000 * q.K):
        """
        Generate a Rayleigh Jeans tail for the SED

        Parameters
        ----------
        teff: astropy.units.quantity.Quantity
            The effective temperature of the source
        """
        # Make the blackbody from 0.1 to 1000um
        rj_wave = np.linspace(0.1, 1000, 2000) * q.um
        rj = sp.Blackbody(rj_wave, (teff, 100 * q.K), name='RJ Tail')

        # Convert to native units
        rj.wave_units = self.wave_units
        rj.flux_units = self.flux_units

        # Normalize to longest wavelength data
        if self.max_spec > self.max_phot:

            # If sufficient RJ tail, ignore OPT and NIR
            if self.max_spec > 5 * q.um:
                rj = rj.trim(include=[(5 * q.um, 1000 * q.um)])[0]
            rj = rj.norm_to_spec(self.app_spec_SED)
            max_wave = self.max_spec.to(q.um)

        else:

            # If sufficient RJ tail, ignore OPT and NIR...
            if self.max_phot > 3 * q.um:
                rj = rj.norm_to_mags(self.photometry[self.photometry['eff'] > 3 * q.um])

            # ...or just ignore OPT...
            elif self.max_phot > 3 * q.um:
                rj = rj.norm_to_mags(self.photometry[self.photometry['eff'] > 3 * q.um])

            # ...or just normalize to whatever you have
            else:
                rj = rj.norm_to_mags(self.photometry)
            max_wave = self.max_phot.to(q.um)

        # Trim so there is no data overlap
        rj = rj.trim(include=[(max_wave, 1000 * q.um)])[0]

        self.rj = rj

    def make_sed(self):
        """
        Construct the SED
        """
        # Make sure there is data
        if len(self.spectra) == 0 and len(self.photometry) == 0:
            self.message('Cannot make the SED without spectra or photometry!')
            return

        # Calculate flux and calibrate
        self._calibrate_photometry()

        # Combine spectra and flux calibrate
        self._calibrate_spectra()

        # Turn off print statements
        verb = self.verbose
        self.verbose = False

        if len(self.stitched_spectra) > 0:

            # Make list of spectrum segment limits
            seg_limits = [(spec.wave_min, spec.wave_max) for spec in self.stitched_spectra]

            if self.use_best_fit:

                # Check that a best fit exists
                if len(self.best_fit) == 0:
                    self.message("Please run a fitting routine to include best fit in calculations")

                else:

                    # Get the model
                    if isinstance(self.use_best_fit, str):
                        model = self.best_fit[self.use_best_fit]
                    else:
                        # Get name of first best fit
                        bf_name = list(self.best_fit.keys())[0]
                        model = self.best_fit[bf_name]

                    # Get the full best fit model
                    const = model['const']
                    model = model['full_model']
                    model.wave_units = self.wave_units

                    # Add the segments to the list of spectra (to be removed later)
                    seg_models = model.trim(exclude=seg_limits, concat=False)
                    n_models = len(seg_models)
                    for n, mod in enumerate(seg_models):
                        data = mod.spectrum
                        data[1] *= const
                        self.add_spectrum(data, name='model')
                    self._calibrate_spectra()

            # If photometry and spectra, exclude photometric points with spectrum coverage
            if len(self.photometry) > 0:

                # Check photometry for spectral coverage
                uncovered = self.app_phot_SED.trim(exclude=seg_limits, concat=False)

                # If all the photometry is covered, just use spectra
                if len(uncovered) == 0:
                    self.app_specphot_SED = np.sum(self.stitched_spectra)

                # Otherwise make spectra + photometry curve from stitched spectra and uncovered photometry
                else:

                    # Concatenate points from the apparent spectrum SED and the uncovered apparent photometry
                    concat = np.concatenate([uncov.data for uncov in uncovered] + [spec.data for spec in self.stitched_spectra], axis=1).T

                    # Reorder by wavelength and turn into Spectrum object
                    ord = concat[np.argsort(concat[:, 0])].T
                    self.app_specphot_SED = sp.Spectrum(ord[0] * self.wave_units, ord[1] * self.flux_units, ord[2] * self.flux_units)

            # If no photometry, just use spectra
            else:
                self.app_specphot_SED = np.sum(self.stitched_spectra)

        # If no spectra, just use photometry
        else:
            self.app_specphot_SED = self.app_phot_SED

        # Make Wein and Rayleigh Jeans tails
        self.make_wein_tail()
        self.make_rj_tail()

        # Remove model spectra and trim Wein and RJ tails
        if self.use_best_fit and len(self.best_fit) > 0:
            for _ in range(n_models):
                self.drop_spectrum(-1)
            self._calibrate_spectra()

        # Restore print statements
        self.verbose = verb

        # Run the calculation
        self._calculate_sed()

        # Set SED as calculated
        self.calculated = True

    def make_wein_tail(self, teff=None):
        """
        Generate a Wein tail for the SED

        Parameters
        ----------
        teff: astropy.units.quantity.Quantity (optional)
            The effective temperature of the source
        """
        if teff is not None:

            # Make the blackbody from ~0 to 1um
            wein_wave = np.linspace(0.0001, 1.1, 500) * q.um
            wein = sp.Blackbody(wein_wave, (teff, 100 * q.K), name='Wein Tail')

            # Convert to native units
            wein.wave_units = self.wave_units
            wein.flux_units = self.flux_units

            # Normalize to shortest wavelength data
            if self.min_spec < self.min_phot:
                wein = wein.norm_to_spec(self.app_spec_SED, exclude=[(1.1 * q.um, 1E30 * q.um)])
            else:
                # Ignore NIR photometry
                wein = wein.norm_to_mags(self.photometry[self.photometry['eff'] < 1 * q.um])

        else:

            # Otherwise just use ~0 flux at ~0 wavelength
            wein = sp.Spectrum((np.array([0.0001]) * q.um).to(self.wave_units), np.array([1E-30]) * self.flux_units, np.array([1E-30]) * self.flux_units, name='Wein Tail')

        # Trim so there is no data overlap
        min_wave = min(self.min_spec, self.min_phot)
        wein = wein.trim(include=[(0 * q.um, min_wave)])[0]

        self.wein = wein
        
    @property
    def mass(self):
        """Getter for mass"""
        return self._mass

    @mass.setter
    def mass(self, mass):
        """
        A setter for mass

        Parameters
        ----------
        mass: sequence
            The mass and uncertainty in mass units
        """
        if mass is None:
            self._mass = None
            self._refs.pop('mass', None)

        else:

            # If the last value is string, it's the reference
            if isinstance(mass[-1], str):
                ref = mass[-1]
                mass = mass[:-1]
            else:
                ref = None

            # Make sure it's a sequence
            if not u.issequence(mass, length=[2, 3]):
                raise TypeError("mass must be a sequence of (value, error) or (value, lower_error, upper_error).")

            # Make sure it's in mass units
            if not all([u.equivalent(ms, q.M_sun) for ms in mass]):
                raise TypeError("Mass values must be mass units of astropy.units.quantity.Quantity, e.g. 'M_sun'")

            # Set the mass!
            self._mass = mass

            # Set reference
            self._refs['mass'] = ref

            self.message("Setting mass to {} with reference '{}'".format(self.mass, ref))

        # Set SED as uncalculated
        self.calculated = False

    def mass_from_age(self, mass_units=q.Msun, plot=False):
        """
        Estimate the mass from model isochrones given an age and Lbol

        Parameters
        ----------
        mass_units: astropy.units.quantity.Quantity
            The units for the mass
        """
        if self.age is not None and self.Lbol_sun is not None:

            # Default
            mass = None

            # Check for uncertainties
            if self.Lbol_sun[1] is None:
                self.message('Lbol={0.Lbol}. Uncertainties are needed to calculate the mass.'.format(self))
            else:
                mass = self.evo_model.evaluate(self.Lbol_sun, self.age, 'Lbol', 'mass', plot=plot)

            # Print a message if None
            if mass is None:
                self.message("Could not calculate mass.")

            # Store the value
            self.mass = [i.round(3) for i in mass] if mass is not None else mass

        else:
            self.message('Lbol={0.Lbol} and age={0.age}. Both are needed to calculate the mass.'.format(self))

    @property
    def membership(self):
        """
        A property for membership
        """
        return self._membership

    @membership.setter
    def membership(self, membership):
        """
        A setter for membership

        Parameters
        ----------
        membership: str
            The name of the moving group to assign membership to
        """
        if membership is None:

            self._membership = None

        elif membership in iso.NYMG_AGES:

            # Set the membership!
            self._membership = membership

            self.message('Setting membership to', self.membership)

            # Set the age
            self.age = iso.NYMG_AGES.get(membership)

        else:
            self.message('{} not valid. Supported memberships include {}.'.format(membership, ', '.join(iso.NYMG_AGES.keys())))

    def message(self, msg, pre='[sedkit]'):
        """
        Only print message if verbose=True

        Parameters
        ----------
        msg: str
            The message to print
        pre: str
            The stuff to print before
        """
        if self.verbose:
            if pre is None:
                print(msg)
            else:
                print("{} {}".format(pre, msg))

    @property
    def name(self):
        """
        A property for name
        """
        return self._name

    @name.setter
    def name(self, new_name):
        """
        A setter for the source name, which looks up metadata given a Simbad name

        Parameters
        ----------
        new_name: str
            The name
        """
        # Convert from bytes if necessary
        if isinstance(new_name, bytes):
            new_name = new_name.decode('UTF-8')

        # Make sure it's a string
        if not isinstance(new_name, str):
            raise TypeError("{}: Name must be a string, not {}".format(new_name, type(new_name)))

        # Set the attribute
        self._name = new_name
        self.message("Setting name to {}".format(self.name))

        # Check for Simbad record (repeat if ConnectionError)
        try:
            self.find_Simbad()
        except:
            time.sleep(4)
            self.find_Simbad()

    @property
    def parallax(self):
        """
        A property for parallax
        """
        return self._parallax

    @parallax.setter
    def parallax(self, parallax):
        """
        A setter for parallax

        Parameters
        ----------
        parallax: sequence
            The (parallax, err) or (parallax, lower_err, upper_err)
        """
        if parallax is None:
            self._parallax = None
            self._distance = None
            self._refs.pop('parallax', None)
            self._refs.pop('distance', None)

        else:

            # If the last value is string, it's the reference
            if isinstance(parallax[-1], str):
                ref = parallax[-1]
                parallax = parallax[:-1]
            else:
                ref = None

            # Make sure it's a sequence
            if not u.issequence(parallax, length=[2, 3]):
                raise TypeError("Parallax must be a sequence of (value, error) or (value, lower_error, upper_error).")

            # Make sure the values are in distance units
            if not all([u.equivalent(plx, q.mas) for plx in parallax]):
                raise TypeError("Parallax values must be angular units of astropy.units.quantity.Quantity, e.g. 'mas'")

            # Set the parallax!
            self._parallax = parallax

            # Update the distance
            self._distance = u.pi2pc(*self.parallax)

            # Set reference
            self._refs['parallax'] = ref
            self._refs['distance'] = ref

            self.message("Setting parallax to {} and distance to {} with reference '{}'".format(self.parallax, self.distance, ref))

        # Try to calculate reddening
        self.get_reddening()

        # Update the absolute photometry
        self._calibrate_photometry()

        # Update the flux calibrated spectra
        self._calibrate_spectra()

        # Set SED as uncalculated
        self.calculated = False

    @property
    def photometry(self):
        """
        A property for photometry
        """
        self._photometry.sort('eff')
        return self._photometry

    def plot(self, app=True, photometry=True, spectra=True, integral=False,
             synthetic_photometry=False, blackbody=False, best_fit=False, normalize=None,
             scale=['log', 'log'], output=False, fig=None, color='#1f77b4', one_color=True,
             **kwargs):
        """
        Plot the SED

        Parameters
        ----------
        app: bool
            Plot the apparent SED instead of absolute
        photometry: bool
            Plot the photometry
        spectra: bool
            Plot the spectra
        integrals: bool
            Plot the curve used to calculate fbol
        synthetic_photometry: bool
            Plot the synthetic photometry
        blackbody: bool
            Plot the blackbody fit
        best_fit: bool
            Plot the best fit model
        normalize: sequence
            The wavelength ranges to normalize to 1
        scale: array-like
            The (x, y) scales to plot, 'linear' or 'log'
        bokeh: bool
            Plot in Bokeh
        output: bool
            Just return figure, don't draw plot
        fig: bokeh.plotting.figure (optional)
            The Boheh plot to add the SED to
        color: str
            The color for the plot points and lines

        Returns
        -------
        bokeh.models.figure
            The SED plot
        """
        if not self.calculated:
            self.make_sed()

        # Distinguish between apparent and absolute magnitude
        pre = 'app_' if app else 'abs_'

        # Calculate reasonable axis limits
        full_SED = getattr(self, pre + 'SED')
        spec_SED = getattr(self, pre + 'spec_SED')
        phot_cols = ['eff', pre + 'flux', pre + 'flux_unc']
        phot_SED = np.array([np.array([np.nanmean(self.photometry.loc[b][col].value) for b in list(set(self.photometry['band']))]) for col in phot_cols])

        # Calculate normalization constant
        const = 1.
        if isinstance(normalize, (list, tuple)):
            idx = u.idx_include(full_SED.wave, normalize)
            const = u.minimize_norm(np.ones_like(idx), full_SED.flux[idx])[0]

        # Check for min and max phot data
        try:
            mn_xp = np.nanmin(phot_SED[0])
            mx_xp = np.nanmax(phot_SED[0])
            mn_yp = np.nanmin(phot_SED[1])
            mx_yp = np.nanmax(phot_SED[1])
        except:
            mn_xp, mx_xp, mn_yp, mx_yp = 0.3, 18, 0, 1

        # Check for min and max spec data
        try:
            mn_xs = np.nanmin(spec_SED.wave)
            mx_xs = np.nanmax(spec_SED.wave)
            mn_ys = np.nanmin(spec_SED.flux[spec_SED.flux > 0])
            mx_ys = np.nanmax(spec_SED.flux[spec_SED.flux > 0])
        except:
            mn_xs, mx_xs, mn_ys, mx_ys = 0.3, 18, 999, -999

        mn_x = np.nanmin([mn_xp, mn_xs])
        mx_x = np.nanmax([mx_xp, mx_xs])
        mn_y = np.nanmin([mn_yp, mn_ys])
        mx_y = np.nanmax([mx_yp, mx_ys])

        # Use input figure...
        if hasattr(fig, 'legend'):
            self.fig = fig

        # ...or make a new plot
        else:
            TOOLS = ['pan', 'reset', 'box_zoom', 'wheel_zoom', 'save']
            xlab = 'Wavelength [{}]'.format(self.wave_units)
            ylab = 'Flux Density [{}]'.format(str(self.flux_units))
            self.fig = figure(plot_width=800, plot_height=500, title=self.name,
                              y_axis_type=scale[1], x_axis_type=scale[0],
                              x_axis_label=xlab, y_axis_label=ylab,
                              tools=TOOLS)

        # Plot spectra
        if spectra and len(self.spectra) > 0:

            if spectra == 'all':
                for n, spec in enumerate(self.spectra['spectrum']):
                    self.fig = spec.plot(fig=self.fig, components=True, const=const)

            else:
                self.fig.line(spec_SED.wave, spec_SED.flux * const, color=color, alpha=0.8, legend_label='Spectrum')

        # Plot photometry
        if photometry and len(self.photometry) > 0:

            # Set up hover tool
            phot_tips = [('Band', '@desc'), ('Wave', '@x'), ('Flux', '@y'), ('Unc', '@z')]
            hover = HoverTool(names=['photometry', 'nondetection'], tooltips=phot_tips, mode='vline')
            self.fig.add_tools(hover)

            # Plot points with errors
            pts = np.array([(bnd, wav, flx * const, err * const) for bnd, wav, flx, err in np.array(self.photometry['band', 'eff', pre + 'flux', pre + 'flux_unc']) if not any([(np.isnan(i) or i <= 0) for i in [wav, flx, err]])], dtype=[('desc', 'S20'), ('x', float), ('y', float), ('z', float)])
            if len(pts) > 0:
                source = ColumnDataSource(data=dict(x=pts['x'], y=pts['y'], z=pts['z'], desc=[b.decode("utf-8") for b in pts['desc']]))
                self.fig.circle('x', 'y', source=source, legend_label='Photometry', name='photometry', color=color, fill_alpha=0.7, size=8)
                y_err_x = []
                y_err_y = []
                for name, px, py, err in pts:
                    y_err_x.append((px, px))
                    y_err_y.append((py - err, py + err))
                self.fig.multi_line(y_err_x, y_err_y, color=color)

            # Plot points without errors
            pts = np.array([(bnd, wav, flx * const, err * const) for bnd, wav, flx, err in np.array(self.photometry['band', 'eff', pre + 'flux', pre + 'flux_unc']) if (np.isnan(err) or err <= 0) and not np.isnan(flx)], dtype=[('desc', 'S20'), ('x', float), ('y', float), ('z', float)])
            if len(pts) > 0:
                source = ColumnDataSource(data=dict(x=pts['x'], y=pts['y'], z=pts['z'], desc=[b.decode("utf-8") for b in pts['desc']]))
                self.fig.circle('x', 'y', source=source, legend_label='Limit', name='nondetection', color=color, fill_alpha=0, size=8)

        # Plot photometry
        if synthetic_photometry and len(self.synthetic_photometry) > 0:

            # Set up hover tool
            phot_tips = [('Band', '@desc'), ('Wave', '@x'), ('Flux', '@y'), ('Unc', '@z')]
            hover = HoverTool(names=['synthetic photometry'], tooltips=phot_tips, mode='vline')
            self.fig.add_tools(hover)

            # Plot points with errors
            pts = np.array([(bnd, wav, flx * const, err * const) for bnd, wav, flx, err in np.array(self.synthetic_photometry['band', 'eff', pre + 'flux', pre + 'flux_unc']) if not any([np.isnan(i) for i in [wav, flx, err]])], dtype=[('desc', 'S20'), ('x', float), ('y', float), ('z', float)])
            if len(pts) > 0:
                source = ColumnDataSource(data=dict(x=pts['x'], y=pts['y'], z=pts['z'], desc=[b.decode("utf-8") for b in pts['desc']]))
                self.fig.square('x', 'y', source=source, legend_label='Synthetic Photometry', name='synthetic photometry', color=color, fill_alpha=0.7, size=8)
                y_err_x = []
                y_err_y = []
                for name, px, py, err in pts:
                    y_err_x.append((px, px))
                    y_err_y.append((py - err, py + err))
                self.fig.multi_line(y_err_x, y_err_y, color=color)

        # Plot the SED with linear interpolation completion
        if integral:
            label = str(self.Teff[0]) if self.Teff is not None else 'Integral'
            self.fig.line(full_SED.wave, full_SED.flux * const, line_color=color if one_color else 'black', alpha=0.3, legend_label=label)

        # Plot the blackbody fit
        if blackbody and self.blackbody:
            bb_wav, bb_flx = self.blackbody.data[:2]
            self.fig.line(bb_wav, bb_flx * const, line_color=color if one_color else 'red', legend_label='{} K'.format(self.Teff_bb))

        if best_fit and len(self.best_fit) > 0:
            col_list = u.color_gen('Category10', n=len(self.best_fit) + 1)
            _ = next(col_list)
            for bf, mod_fit in self.best_fit.items():
                mod = mod_fit['full_model']
                mod.wave_units = self.wave_units
                self.fig.line(mod.wave, mod.flux * mod_fit['const'], alpha=0.3, color=color if one_color else next(col_list), legend_label=mod_fit['label'], line_width=2)

        self.fig.legend.location = "top_right"
        self.fig.legend.click_policy = "hide"
        self.fig.x_range = Range1d(mn_x * 0.8, mx_x * 1.2)
        self.fig.y_range = Range1d(mn_y * 0.5 * const, mx_y * 2 * const)

        if not output:
            show(self.fig)

        return self.fig

    @property
    def ra(self):
        """
        A property for right ascension
        """
        return self._ra

    @ra.setter
    def ra(self, ra, ra_unc=None, frame='icrs'):
        """
        Set the right ascension of the source

        Parameters
        ----------
        ra: astropy.units.quantity.Quantity
            The right ascension
        ra_unc: astropy.units.quantity.Quantity (optional)
            The uncertainty
        frame: str
            The reference frame
        """
        if not isinstance(ra, (q.quantity.Quantity, str)):
            raise TypeError("{}: Cannot interpret ra".format(ra))

        # Make sure it's decimal degrees
        self._ra = Angle(ra)
        if self.dec is not None:
            sky_coords = SkyCoord(ra=self.ra, dec=self.dec, unit=q.degree, frame='icrs')
            self._set_sky_coords(sky_coords, simbad=False)

    @property
    def radius(self):
        """
        A property for radius
        """
        return self._radius

    @radius.setter
    def radius(self, radius):
        """
        A setter for radius

        Parameters
        ----------
        radius: sequence
            The radius and uncertainty in distance units
        """
        if radius is None:
            self._radius = None
            self._refs.pop('radius', None)

        else:

            # If the last value is string, it's the reference
            if isinstance(radius[-1], str):
                ref = radius[-1]
                radius = radius[:-1]
            else:
                ref = None

            # Make sure it's a sequence
            if not u.issequence(radius, length=[2, 3]):
                raise TypeError("Radius must be a sequence of (value, error) or (value, lower_error, upper_error).")

            # Make sure the values are in distance units
            if not all([u.equivalent(rad, q.R_sun) for rad in radius]):
                raise TypeError("Radius values must be length units of astropy.units.quantity.Quantity, e.g. 'R_sun'")

            # Set the radius!
            self._radius = radius

            # Set reference
            self._refs['radius'] = ref

            self.message("Setting radius to {} with reference '{}'".format(self.radius, ref))

        # Set SED as uncalculated
        self.calculated = False

    def radius_from_spectral_type(self, spt=None):
        """
        Estimate the radius from CMD plot

        Parameters
        ----------
        spt: float
            The spectral type float, where 0-99 correspond to types O0-Y9
        """
        spt = spt or self.spectral_type[0]
        try:
            self.radius = SptRadius.get_radius(spt)

        except:
            self.message("Could not estimate radius from spectral type {}".format(spt))

    def radius_from_age(self, radius_units=q.Rsun, plot=False):
        """
        Estimate the radius from model isochrones given an age and Lbol

        Parameters
        ----------
        radius_units: astropy.units.quantity.Quantity
            The radius units
        """
        if self.age is not None and self.Lbol_sun is not None:

            # Default
            radius = None

            # Check for uncertainties
            if self.Lbol_sun[1] is None:
                self.message('Lbol={0.Lbol}. Uncertainties are needed to calculate the radius.'.format(self))
            else:
                self.evo_model.radius_units = radius_units
                radius = self.evo_model.evaluate(self.Lbol_sun, self.age, 'Lbol', 'radius', plot=plot)

            # Print a message if None
            if radius is None:
                self.message("Could not calculate radius.")

            # Store the value
            self.radius = [i.round(3) for i in radius] if radius is not None else radius
            if radius is not None:
                self.isochrone_radius = True
            self.message("{}: Lbol={}, age={} ==> radius={}".format(self.evo_model.name, self.Lbol_sun, self.age, self.radius ))


        else:
            self.message('Lbol={0.Lbol} and age={0.age}. Both are needed to calculate the radius.'.format(self))

    @property
    def refs(self):
        """
        Getter for the references
        """
        # Get the singular references
        all_refs = self._refs

        # Collect multiple references
        if len(self.photometry) > 0:
            all_refs['photometry'] = {row['band']: row['ref'] for row in self.photometry}
        if len(self.spectra) > 0:
            all_refs['spectra'] = {row['name']: row['ref'] for row in self.spectra}
        if len(self.best_fit) > 0:
            all_refs['best_fit'] = {name: val['full_model'].ref for name, val in self.best_fit.items()}

        return all_refs

    @property
    def results(self):
        """
        A property for displaying the results
        """
        # Make the SED to get the most recent results
        if not self.calculated:
            self.make_sed()

        # Get the params to display
        params = copy(self.params)

        # Add best fits
        for name, fit in self.best_fit.items():
            params.append(name)

        # Get the params
        rows = []
        for param in params:

            # Get the values and format
            attr = getattr(self, param, None)

            if attr is None:
                attr = '--'

            if isinstance(attr, (tuple, list)):
                val, unc = attr[:2]
                unit = val.unit if hasattr(val, 'unit') else '--'
                val = val.value if hasattr(val, 'unit') else val
                unc = unc.value if hasattr(unc, 'unit') else unc
                if val < 1E-3 or val > 1e5:
                    val = float('{:.2e}'.format(val))
                    if unc is None:
                        unc = '--'
                    else:
                        unc = float('{:.2e}'.format(unc))
                rows.append([param, val, unc, unit])

            elif isinstance(attr, (str, float, bytes, int)):
                rows.append([param, attr, '--', '--'])

            else:
                pass

        return at.Table(np.asarray(rows), names=('param', 'value', 'unc', 'units'))

    def run_methods(self, method_list):
        """
        Run the methods listed in order

        Parameters
        ----------
        method_list: list
            A list of methods to run with arguments
        """
        # Make into list of lists
        method_list = [[meth, {}] if isinstance(meth, str) else meth for meth in method_list]

        # Iterate over list
        for method, args in method_list:

            # Check for valid method
            if method in dir(self):

                # Check if args are a None, list, or dict
                args = args or {}

                # Make sure args are a dictionary
                if not isinstance(args, dict):
                    raise TypeError("{} arguments must be a dictionary".format(method))

                # Run the method
                getattr(self, method)(**args)

    @property
    def sky_coords(self):
        """
        A property for sky coordinates
        """
        return self._sky_coords

    @sky_coords.setter
    def sky_coords(self, sky_coords, frame='icrs'):
        """
        A setter for sky coordinates

        Parameters
        ----------
        sky_coords: astropy.coordinates.SkyCoord, tuple
            The sky coordinates
        frame: str
            The coordinate frame
        """
        # Make sure it's a sky coordinate
        if not isinstance(sky_coords, (SkyCoord, tuple)):
            raise TypeError('Sky coordinates must be astropy.coordinates.SkyCoord or (ra, dec) tuple.')

        if isinstance(sky_coords, tuple) and len(sky_coords) == 2:

            if isinstance(sky_coords[0], str):
                sky_coords = SkyCoord(ra=sky_coords[0], dec=sky_coords[1], unit=(q.degree, q.degree), frame=frame)

            elif isinstance(sky_coords[0], (float, Angle, q.quantity.Quantity)):
                sky_coords = SkyCoord(ra=sky_coords[0], dec=sky_coords[1], unit=q.degree, frame=frame)

            else:
                raise TypeError("Cannot convert type {} to coordinates.".format(type(sky_coords[0])))

        self._set_sky_coords(sky_coords)

    def _set_sky_coords(self, sky_coords, simbad=True):
        """
        Calculate and set attributes from sky coords

        Parameters
        ----------
        sky_coords: astropy.coordinates.SkyCoord
            The sky coordinates
        simbad: bool
            Search Simbad by the coordinates
        """
        # Set the sky coordinates
        self._sky_coords = sky_coords
        self._ra = sky_coords.ra.degree
        self._dec = sky_coords.dec.degree
        self.message("Setting sky_coords to {}".format(self.sky_coords))

        # Try to calculate reddening
        self.get_reddening()

        # Try to find the source in Simbad
        if simbad:
            self.find_Simbad()

    def spectrum_from_modelgrid(self, model_grid, snr=10, **kwargs):
        """
        Load a spectrum of the given parameters from the given ModelGrid

        Parameters
        ----------
        model_grid: sedkit.modelgrid.ModelGrid
            A model grid to get the spectrum from
        snr: int, float
            The signal to noise to apply
        """
        # Get the model from the model grid
        model = model_grid.get_spectrum(snr=snr, **kwargs)

        # Save the model as a spectrum
        self.add_spectrum(model)

    @property
    def spectra(self):
        """
        A property for spectra
        """
        return self._spectra

    @property
    def spectral_type(self):
        """
        A property for spectral_type
        """
        return self._spectral_type

    @spectral_type.setter
    def spectral_type(self, spectral_type):
        """
        A setter for spectral_type

        Parameters
        ----------
        spec_type: sequence, str
            The nominal spectral type value
        spectral_type_unc: float
            The uncertainty in the spectral type
        gravity: str (optional)
            The low surface gravity suffix, ['b', 'beta', 'g', 'gamma']
        lum_class: str (optional)
            The luminosity class, ['I', 'II', 'III', 'IV', 'V']
        prefix: str
            The spectral type prefix, ['sd', 'esd']
        """
        if spectral_type is None:
            self._spectral_type = None
            self._refs.pop('spectral_type', None)

        else:

            # No reference by default
            ref = None

            # If there are two items, it's the SpT and the reference or the SpT and the unc
            if len(spectral_type) == 2:
                if isinstance(spectral_type[1], str):
                    spectral_type, ref = spectral_type

            # If the spectral type is a float or integer, assume it's the numeric spectral type
            if isinstance(spectral_type, (int, float)):
                spectral_type = spectral_type, 0.5

            # Just a spectral type
            if isinstance(spectral_type, str):
                spec_type = u.specType(spectral_type)
                spectral_type, spectral_type_unc, prefix, gravity, lum_class = spec_type

            elif u.issequence(spectral_type, length=[1, 2, 3, 4, 5]):
                spectral_type, spectral_type_unc, *other = spectral_type
                gravity = lum_class = prefix = ''
                if other:
                    gravity, *other = other
                if other:
                    lum_class, *other = other
                if other:
                    prefix = other[0]

            else:
                raise TypeError('{}: Please provide a string or sequence to set the spectral type.'.format(spectral_type))

            # Set the spectral_type
            self._spectral_type = spectral_type, spectral_type_unc or 0.5
            self.luminosity_class = lum_class or 'V'
            self.gravity = gravity or None
            self.prefix = prefix or None
            self.SpT = u.specType([self.spectral_type[0], self.spectral_type[1], self.prefix, self.gravity, self.luminosity_class])

            # Set the age if not explicitly set
            if self.age is None and self.gravity is not None:
                if gravity in ['b', 'beta', 'g', 'gamma']:
                    self.age = 225 * q.Myr, 75 * q.Myr

                else:
                    self.message("{} is an invalid gravity. Please use 'beta' or 'gamma' instead.".format(gravity))

            # If radius not explicitly set, estimate it from spectral type
            if self.spectral_type is not None and self.radius is None:
                self.radius_from_spectral_type()

            # Update the reference
            self._refs['spectral_type'] = ref

            self.message("Setting spectral_type to {} with reference '{}'".format((self.spectral_type[0], self.spectral_type[1], self.luminosity_class, self.gravity, self.prefix), ref))

        # Set SED as uncalculated
        self.calculated = False

    @property
    def synthetic_photometry(self):
        """
        A property for synthetic photometry
        """
        if len(self._synthetic_photometry) == 0:
            self.message("No synthetic photometry. Run `calculate_synthetic_photometry()` to generate.")

        # Sort by wavelength
        # self._synthetic_photometry.sort('eff')

        return self._synthetic_photometry

    def teff_from_age(self, teff_units=q.K, plot=False):
        """
        Estimate the radius from model isochrones given an age and Lbol

        Parameters
        ----------
        teff_units: astropy.units.quantity.Quantity
            The temperature units to use
        """
        if self.age is not None and self.Lbol_sun is not None:

            # Default
            teff = None

            # Check for uncertainties
            if self.Lbol_sun[1] is None:
                self.message('Lbol={0.Lbol}. Uncertainties are needed to calculate the teff.'.format(self))
            else:
                self.evo_model.teff_units = teff_units
                teff = self.evo_model.evaluate(self.Lbol_sun, self.age, 'Lbol', 'teff', plot=plot)

            # Print a message if None
            if teff is None:
                self.message("Could not calculate teff.")

            # Store the value
            self.Teff_evo = [i.round(0) for i in teff] if teff is not None else teff

        else:
            self.message('Lbol={0.Lbol} and age={0.age}. Both are needed to calculate the teff.'.format(self))

    @property
    def wave_units(self):
        """
        A property for wave_units
        """
        return self._wave_units

    @wave_units.setter
    def wave_units(self, wave_units):
        """
        A setter for wave_units

        Parameters
        ----------
        wave_units: astropy.units.quantity.Quantity
            The astropy units of the SED wavelength
        """
        # Make sure the values are in length units
        if not u.equivalent(wave_units, q.um):
            raise TypeError("{}: wave_units must be length units of astropy.units.quantity.Quantity, e.g. 'um'".format(wave_units))

        # Set the wave_units!
        self._wave_units = wave_units
        self.units = [self._wave_units, self._flux_units, self._flux_units]

        # Recalibrate the data
        self._calibrate_photometry()
        self._calibrate_spectra()


class VegaSED(SED):
    """
    A precomputed SED of Vega
    """
    def __init__(self, **kwargs):
        """Initialize the SED of Vega"""
        # Make the Spectrum object
        super().__init__(**kwargs)

        self.name = 'Vega'
        self.find_SDSS()
        self.find_2MASS()
        self.find_WISE()
        self.radius = 2.818 * q.Rsun, 0.008 * q.Rsun, '2010ApJ...708...71Y'
        self.age = 455 * q.Myr, 13 * q.Myr, '2010ApJ...708...71Y'
        self.logg = 4.1, 0.1, '2006ApJ...645..664A'

        # Get the spectrum
        self.add_spectrum(sp.Vega(snr=100))

        # Calculate
        self.make_sed()
