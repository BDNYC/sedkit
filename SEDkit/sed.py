#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
SEDkit rewritten with astropy and astrodbkit
"""
import numpy as np
import astropy.table as at
import astropy.units as q
import astropy.io.ascii as ii
import astropy.constants as ac
import pysynphot as ps
from astropy.modeling.models import custom_model
from astropy.modeling import models, fitting
from astropy.modeling.blackbody import blackbody_lambda
from astropy.constants import b_wien
from . import utilities as u
from . import synphot as s
from . import spectrum as sp
from . import isochrone as iso
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import HoverTool, Label, Range1d, BoxZoomTool, ColumnDataSource


PHOT_ALIASES = {'2MASS_J':'2MASS.J', '2MASS_H':'2MASS.H', '2MASS_Ks':'2MASS.Ks', 'WISE_W1':'WISE.W1', 'WISE_W2':'WISE.W2', 'WISE_W3':'WISE.W3', 'WISE_W4':'WISE.W4', 'IRAC_ch1':'IRAC.I1', 'IRAC_ch2':'IRAC.I2', 'IRAC_ch3':'IRAC.I3', 'IRAC_ch4':'IRAC.I4', 'SDSS_u':'SDSS.u', 'SDSS_g':'SDSS.g', 'SDSS_r':'SDSS.r', 'SDSS_i':'SDSS.i', 'SDSS_z':'SDSS.z', 'MKO_J':'NSFCam.J', 'MKO_Y':'Wircam.Y', 'MKO_H':'NSFCam.H', 'MKO_K':'NSFCam.K', "MKO_L'":'NSFCam.Lp', "MKO_M'":'NSFCam.Mp', 'Johnson_V':'Johnson.V', 'Cousins_R':'Cousins.R', 'Cousins_I':'Cousins.I', 'FourStar_J':'FourStar.J', 'FourStar_J1':'FourStar.J1', 'FourStar_J2':'FourStar.J2', 'FourStar_J3':'FourStar.J3', 'HST_F125W':'WFC3_IR.F125W'}

def from_ids(db, **kwargs):
    """
    Create dictionary of data tables from record id values or lists
    
    Example
    -------
    data = sed.from_ids(db, sources=2, photometry=[1096,1097,12511,12512], spectra=[3176,3773], parallaxes=575)
    
    Parameters
    ----------
    db: astrodbkit.astrodb.Database
        The database to draw the records from
    """
    # Make an empty dict
    data = {}.fromkeys(kwargs.keys())
    
    # Generate each table
    for k,v in kwargs.items():
        try:
            # Option to get all records
            if v=='*':
                v = db.query("SELECT id from {} WHERE source_id={}".format(k,kwargs['sources']))['id']
                
            # Make sure it's a list
            if isinstance(v, int):
                v = [v]
                
            # Build the query with the provided ids
            id_str = ','.join(list(map(str,v)))
            qry = "SELECT * FROM {} WHERE id IN ({})".format(k,id_str)
            data[k] = db.query(qry, fmt='table')
            
        except IOError:
            print('Could not generate',k,'table.')
            
    return data


class SED(object):
    """
    A class to construct spectral energy distributions and calculate fundamental paramaters of stars
    
    Attributes
    ==========
    Lbol: astropy.units.quantity.Quantity
        The bolometric luminosity [erg/s]
    Lbol_sun: astropy.units.quantity.Quantity
        The bolometric luminosity [L_sun]
    Mbol: float
        The absolute bolometric magnitude
    SpT: float
        The string spectral type
    Teff: astropy.units.quantity.Quantity
        The effective temperature calculated from the SED
    Teff_bb: astropy.units.quantity.Quantity
        The effective temperature calculated from the blackbody fit
    abs_SED: sequence
        The [W,F,E] of the calculate absolute SED
    abs_phot_SED: sequence
        The [W,F,E] of the calculate absolute photometric SED
    abs_spec_SED: sequence
        The [W,F,E] of the calculate absolute spectroscopic SED
    age_max: astropy.units.quantity.Quantity
        The upper limit on the age of the target
    age_min: astropy.units.quantity.Quantity
        The lower limit on the age of the target
    app_SED: sequence
        The [W,F,E] of the calculate apparent SED
    app_phot_SED: sequence
        The [W,F,E] of the calculate apparent photometric SED
    app_spec_SED: sequence
        The [W,F,E] of the calculate apparent spectroscopic SED
    bb_source: str
        The [W,F,E] fit to calculate Teff_bb
    blackbody: astropy.modeling.core.blackbody
        The best fit blackbody function
    distance: astropy.units.quantity.Quantity
        The target distance
    fbol: astropy.units.quantity.Quantity
        The apparent bolometric flux [erg/s/cm2]
    flux_units: astropy.units.quantity.Quantity
        The desired flux density units
    gravity: str
        The surface gravity suffix
    mbol: float
        The apparent bolometric magnitude
    name: str
        The name of the target
    parallaxes: astropy.table.QTable
        The table of parallaxes
    photometry: astropy.table.QTable
        The table of photometry
    piecewise: sequence
        The list of all piecewise combined spectra for normalization
    radius: astropy.units.quantity.Quantity
        The target radius
    sources: astropy.table.QTable
        The table of sources (with only one row of cource)
    spectra: astropy.table.QTable
        The table of spectra
    spectral_type: float
        The numeric spectral type, where 0-99 corresponds to spectral types O0-Y9
    spectral_types: astropy.table.QTable
        The table of spectral types
    suffix: str
        The spectral type suffix
    syn_photometry: astropy.table.QTable
        The table of calcuated synthetic photometry
    wave_units: astropy.units.quantity.Quantity
        The desired wavelength units
    """
    def __init__(self, name='My Target', verbose=True, **kwargs):
        """
        Pulls all available data from the BDNYC Data Archive, 
        constructs an SED, and stores all calculations at *pickle_path*
        
        Parameters
        ----------
        name: str (optional)
            A name for the target
        verbose: bool
            Print some diagnostic stuff
        """
        # Single valued attributes
        self.name = name
        self.verbose = verbose
        self._age = None
        self._distance = None
        self._parallax = None
        self._radius = None
        self._spectral_type = None
        self._membership = None
        
        # Set the default wavelength and flux units
        self._wave_units = q.um 
        self._flux_units = q.erg/q.s/q.cm**2/q.AA
        self.units = [self._wave_units, self._flux_units, self._flux_units]
        
        # Attributes of arbitrary length
        self._spectra = []
        self.stitched_spectra = []
        self.app_spec_SED = None
        self.abs_spec_SED = None
        self.app_phot_SED = None
        self.abs_phot_SED = None
        phot_cols = ('band', 'eff', 'app_magnitude', 'app_magnitude_unc', 'app_flux', 'app_flux_unc', 'abs_magnitude', 'abs_magnitude_unc', 'abs_flux', 'abs_flux_unc', 'bandpass')
        phot_typs = ('U16', float, float, float, float, float, float, float, float, float, 'O')
        self._photometry = at.QTable(names=phot_cols, dtype=phot_typs)
        self._photometry['eff'].unit = self._wave_units
        self._photometry['app_flux'].unit = self._flux_units
        self._photometry['app_flux_unc'].unit = self._flux_units
        self._photometry['abs_flux'].unit = self._flux_units
        self._photometry['abs_flux_unc'].unit = self._flux_units
        self._photometry.add_index('band')
        
        # Try to set attributes from kwargs
        for k,v in kwargs.items():
            setattr(self, k, v)
            
        # Make a plot
        self.fig = figure()
        
        # Empty result attributes
        self.Teff = None
        self.Lbol = None
        self.Mbol = None
        
        # Default parameters
        self.age = 6*q.Gyr, 4*q.Gyr
        self.mass = None
        self.logg = None
    
    
    def add_photometry(self, band, mag, mag_unc, **kwargs):
        """Add a photometric measurement to the photometry table
        
        Parameters
        ----------
        band: name, SEDkit.synphot.Bandpass
            The bandpass name or instance
        mag: float
            The magnitude
        mag_unc: float
            The magnitude uncertainty
        """
        # Make sure the magnitudes are floats
        if not isinstance(mag, float) and not isinstance(mag_unc, float):
            raise TypeError("Magnitude and uncertainty must be floats.")
            
        # Get the bandpass
        if isinstance(band, str):
            bp = s.Bandpass(band)
        elif isinstance(band, SEDkit.synphot.Bandpass):
            bp, band = band, band.name
        else:
            print('Not a recognized bandpass:',band)
            
        # Convert bandpass to desired units
        bp.wave_units = self.wave_units
        
        # Drop the current band if it exists
        if band in self.photometry['band']:
            self.drop_photometry(band)
        
        # Make a dict for the new point
        new_photometry = {'band':band, 'eff':bp.eff, 'app_magnitude':mag, 'app_magnitude_unc':mag_unc, 'bandpass':bp}
        
        # Add the kwargs
        new_photometry.update(kwargs)
            
        # Add it to the table
        self._photometry.add_row(new_photometry)
        
        
    def add_photometry_file(self, file):
        """Add a table of photometry from an ASCII file that 
        contains the columns 'band', 'magnitude', and 'uncertainty'
        
        Parameters
        ----------
        file: str
            The path to the ascii file
        """
        # Read the data
        table = ii.read(file)
        
        # Test to see if columns are present
        cols = ['band','magnitude','uncertainty']
        if not all([i in table.colnames for i in cols]):
            raise TableError('File must contain columns', cols)
        
        # Keep relevant cols
        table = table[cols]
        
        # Add the data to the SED object
        for row in table:
            self.add_photometry(*row)
        
        
    def add_spectrum(self, wave, flux, unc=None, **kwargs):
        """Add a new Spectrum object to the SED

        Parameters
        ----------
        wave: astropy.units.quantity.Quantity
            The wavelength array
        flux: astropy.units.quantity.Quantity
            The flux array
        unc: astropy.units.quantity.Quantity (optional)
            The uncertainty array
        """
        # Create the Spectrum object
        spec = sp.Spectrum(wave, flux, unc, **kwargs)
        
        # Check to see if it is a duplicate spectrum
        if len(self.spectra)>0 and any([all(spec.wave==i.wave) for i in self.spectra]):
            print('Looks like you already added this spectrum. If you want to overwrite, run drop_spectrum() first.')
            
        else:
            # Convert to SED units
            spec.wave_units = self.wave_units
            spec.flux_units = self.flux_units
        
            # Add the spectrum object to the list of spectra
            self._spectra.append(spec)
        
        
    def add_spectrum_file(self, file, wave_units, flux_units):
        """Add a 2 or 3 column spectrum from an ASCII file
        
        Parameters
        ----------
        file: str
            The path to the ascii file
        wave_units: astropy.units.quantity.Quantity
            The wavelength units
        flux_units: astropy.units.quantity.Quantity
            The flux units
        """
        # Read the data
        data = np.genfromtxt(file, unpack=True)
        
        # Apply units
        wave = data[0]*wave_units
        flux = data[1]*flux_units
        if len(data)>2:
            unc = data[2]*flux_units
        else:
            unc = None
        
        # Add the data to the SED object
        self.add_spectrum(wave, flux, unc=unc)
            
            
    @property
    def age(self):
        """A property for age"""
        return self._age
    
    
    @age.setter
    def age(self, age):
        """A setter for age"""
        # Make sure it's a sequence
        if not isinstance(age, (tuple, list, np.ndarray)) or len(age) not in [2,3]:
            raise TypeError('Age must be a sequence of (value, error) or (value, lower_error, upper_error).')
            
        # Make sure the values are in time units
        try:
            _ = [i.to(q.Myr) for i in age]
        except:
            raise TypeError("Age values must be time units of astropy.units.quantity.Quantity, e.g. 'Myr'")
        
        # Set the age!
        self._age = age
        
        if self.verbose:
            print('Setting age to',self.age)
        
        
    def _calibrate_photometry(self):
        """Calculate the absolute magnitudes and flux values of all rows in the photometry table
        """
        if self.photometry is not None and len(self.photometry)>0:
            
            # Update the photometry
            self._photometry['eff'] = self._photometry['eff'].to(self.wave_units)
            self._photometry['app_flux'] = self._photometry['app_flux'].to(self.flux_units)
            self._photometry['app_flux_unc'] = self._photometry['app_flux_unc'].to(self.flux_units)
            self._photometry['abs_flux'] = self._photometry['abs_flux'].to(self.flux_units)
            self._photometry['abs_flux_unc'] = self._photometry['abs_flux_unc'].to(self.flux_units)
        
            # Get the app_mags
            m = np.array(self._photometry)['app_magnitude']
            m_unc = np.array(self._photometry)['app_magnitude_unc']
        
            # Calculate app_flux values
            for n,row in enumerate(self._photometry):
                # app_flux, app_flux_unc = u.mag2flux(row['band'], row['app_magnitude'], sig_m=row['app_magnitude_unc'])
                app_flux, app_flux_unc = u.mag2flux(row['bandpass'], row['app_magnitude'], sig_m=row['app_magnitude_unc'])
                self._photometry['app_flux'][n] = app_flux.to(self.flux_units)
                self._photometry['app_flux_unc'][n] = app_flux_unc.to(self.flux_units)
            
            # Calculate absolute mags
            if self._distance is not None:
            
                # Calculate abs_mags
                M, M_unc = u.flux_calibrate(m, self._distance[0], m_unc, self._distance[1])
                self._photometry['abs_magnitude'] = M
                self._photometry['abs_magnitude_unc'] = M_unc
            
                # Calculate abs_flux values
                for n,row in enumerate(self._photometry):
                    # abs_flux, abs_flux_unc = u.mag2flux(row['band'], row['abs_magnitude'], sig_m=row['abs_magnitude_unc'])
                    abs_flux, abs_flux_unc = u.mag2flux(row['bandpass'], row['abs_magnitude'], sig_m=row['abs_magnitude_unc'])
                    self._photometry['abs_flux'][n] = abs_flux.to(self.flux_units)
                    self._photometry['abs_flux_unc'][n] = abs_flux_unc.to(self.flux_units)

            # Make apparent photometric SED with photometry
            app_cols = ['eff','app_flux','app_flux_unc']
            phot_array = np.array(self.photometry[app_cols])
            phot_array = phot_array[(self.photometry['app_flux']>0)&(self.photometry['app_flux_unc']>0)]
            self.app_phot_SED = sp.Spectrum(*[phot_array[i]*Q for i,Q in zip(app_cols,self.units)])

            # Make absolute photometric SED with photometry
            if self.distance is not None:
                self.abs_phot_SED = self.app_phot_SED.flux_calibrate(self.distance)
    
    
    def _calibrate_spectra(self):
        """Create composite spectra and flux calibrate
        """
        if self.spectra is not None and len(self.spectra)>0:
            
            # Update the spectra
            for spectrum in self.spectra:
                spectrum.flux_units = self.flux_units
            
            # Group overlapping spectra and stitch together where possible
            # to form peacewise spectrum for flux calibration
            self.stitched_spectra = []
            if len(self.spectra) > 1:
                groups = self.group_spectra(self.spectra)
                self.stitched_spectra = [np.sum(group) if len(group)>1 else group for group in groups]
                
            # If one or none, no need to make composite
            elif len(self.spectra) == 1:
                self.stitched_spectra = self.spectra
            
            # If no spectra, forget it
            else:
                self.stitched_spectra = []
                print('No spectra available for SED.')
            
            # Renormalize the stitched spectra
            self.stitched_spectra = [i.norm_to_mags(self.photometry) for i in self.stitched_spectra]
                
            # Make apparent spectral SED
            if len(self.stitched_spectra)>1:
                self.app_spec_SED = sum(self.stitched_spectra)
            elif len(self.stitched_spectra)==1:
                self.app_spec_SED = self.stitched_spectra[0]
            else:
                self.app_spec_SED = None
            
            # Make absolute spectral SED
            if self.app_spec_SED is not None and self.distance is not None:
                self.abs_spec_SED = self.app_spec_SED.flux_calibrate(self.distance)
    
    
    @property
    def distance(self):
        """A property for distance"""
        return self._distance
    
    
    @distance.setter
    def distance(self, distance):
        """A setter for distance
        
        Parameters
        ----------
        distance: sequence
            The (distance, err) or (distance, lower_err, upper_err)
        """
        # Make sure it's a sequence
        if not isinstance(distance, (tuple, list, np.ndarray)) or len(distance) not in [2,3]:
            raise TypeError('Distance must be a sequence of (value, error) or (value, lower_error, upper_error).')
            
        # Make sure the values are in time units
        try:
            _ = [i.to(q.pc) for i in distance]
        except:
            raise TypeError("Distance values must be length units of astropy.units.quantity.Quantity, e.g. 'pc'")
        
        # Set the distance
        self._distance = distance
        
        if self.verbose:
            print('Setting distance to',self.distance)
        
        # Update the parallax
        self._parallax = u.pi2pc(*self.distance, pc2pi=True)
        
        # Update the absolute photometry
        self._calibrate_photometry()

        # Update the flux calibrated spectra
        self._calibrate_spectra()
        
        
    def drop_photometry(self, band):
        """Drop a photometry by its index or name in the photometry list
        """
        if isinstance(band, str) and band in self.photometry['band']:
            band = self._photometry.remove_row(np.where(self._photometry['band']==band)[0][0])

        if isinstance(band, int) and band<=len(self._photometry):
            self._photometry.remove_row(band)
        
        
    def drop_spectrum(self, idx):
        """Drop a spectrum by its index in the spectra list
        """
        self._spectra = [i for n,i in enumerate(self._spectra) if n!=idx]
        
        
    @property
    def flux_units(self):
        """A property for flux_units"""
        return self._flux_units
    
    
    @flux_units.setter
    def flux_units(self, flux_units):
        """A setter for flux_units
        
        Parameters
        ----------
        flux_units: astropy.units.quantity.Quantity
            The astropy units of the SED wavelength
        """
        # Make sure it's a quantity
        if not isinstance(flux_units, (q.core.PrefixUnit, q.core.Unit, q.core.CompositeUnit)):
            raise TypeError('flux_units must be astropy.units.quantity.Quantity')
            
        # Make sure the values are in length units
        try:
            flux_units.to(q.erg/q.s/q.cm**2/q.AA)
        except:
            raise TypeError("flux_units must be a unit of length, e.g. 'um'")
        
        # fnu2flam(f_nu, lam, units=q.erg/q.s/q.cm**2/q.AA)
            
        # Set the flux_units!
        self._flux_units = flux_units
        self.units = [self._wave_units, self._flux_units, self._flux_units]
        
        # Recalibrate the data
        self._calibrate_photometry()
        self._calibrate_spectra()
        
        
    def from_database(self, db, from_dict=None, **kwargs):
        """
        Load the data from a SQL database

        if 'photometry' in kwargs:
            # Get phot from database

        """
        # Get the data for the source from the dictionary of ids
        if isinstance(from_dict, dict):
            if not 'sources' in from_dict:
                from_dict['sources'] = source_id
            all_data = from_ids(db, **from_dict)

        # Or get the inventory from the database
        else:
            all_data = db.inventory(source_id, fetch=True)

        # Store the tables as attributes
        for table in ['sources','spectra','photometry','spectral_types','parallaxes']:

            # Get data from the dictionary
            if table in all_data:
                setattr(self, table, at.QTable(all_data[table]))

            # If no data, generate dummy
            else:
                qry = "SELECT * FROM {} LIMIT 1".format(table)
                dummy = db.query(qry, fmt='table')
                dummy.remove_row(0)
                setattr(self, table, at.QTable(dummy))


    def fundamental_params(self, **kwargs):
        """
        Calculate the fundamental parameters of the current SED
        """
        # Calculate bolometric luminosity (dependent on fbol and distance)
        self.get_Lbol()
        self.get_Mbol()
        
        # Interpolate surface gravity, mass and radius from isochrones
        if self.radius is None:
            self.radius_from_age()
        self.logg_from_age()
        self.mass_from_age()
        
        # Calculate Teff (dependent on Lbol, distance, and radius)
        self.get_Teff()
        
        
    def get_fbol(self, units=q.erg/q.s/q.cm**2):
        """Calculate the bolometric flux of the SED
        
        Parameters
        ----------
        units: astropy.units.quantity.Quantity
            The target untis for fbol
        """
        # Integrate the SED to get fbol
        self.fbol = self.app_SED.integral(units=units)
        
        
    def get_Lbol(self):
        """Calculate the bolometric luminosity of the SED
        """
        # Caluclate fbol if not present
        if not hasattr(self, 'fbol'):
            self.get_fbol()
            
        # Calculate Lbol
        if self.distance is not None:
            Lbol = (4*np.pi*self.fbol[0]*self.distance[0]**2).to(q.erg/q.s)
            Lbol_sun = round(np.log10((Lbol/ac.L_sun).decompose().value), 3)
            
            # Calculate Lbol_unc
            Lbol_unc = Lbol*np.sqrt((self.fbol[1]/self.fbol[0]).value**2+(2*self.distance[1]/self.distance[0]).value**2)
            Lbol_sun_unc = round(abs(Lbol_unc/(Lbol*np.log(10))).value, 3)
            
            # Update the attributes
            self.Lbol = Lbol, Lbol_unc
            self.Lbol_sun = Lbol_sun, Lbol_sun_unc
            
            
    def get_mbol(self, L_sun=3.86E26*q.W, Mbol_sun=4.74):
        """Calculate the apparent bolometric magnitude of the SED
        
        Parameters
        ==========
        L_sun: astropy.units.quantity.Quantity
            The bolometric luminosity of the Sun
        Mbol_sun: float
            The absolute bolometric magnitude of the sun
        """
        # Calculate fbol if not present
        if not hasattr(self, 'fbol'):
            self.get_fbol()
            
        # Calculate mbol
        mbol = round(-2.5*np.log10(self.fbol[0].value)-11.482, 3)
            
        # Calculate mbol_unc
        mbol_unc = round((2.5/np.log(10))*(self.fbol[1]/self.fbol[0]).value, 3)
        
        # Update the attribute
        self.mbol = mbol, mbol_unc
        
        
    def get_Mbol(self):
        """Calculate the absolute bolometric magnitude of the SED
        """
        # Calculate mbol if not present
        if not hasattr(self, 'mbol'):
            self.get_mbol()
           
        # Calculate Mbol
        if self.distance is not None:
            Mbol = round(self.mbol[0]-5*np.log10((self.distance[0]/10*q.pc).value), 3)
            
            # Calculate Mbol_unc
            Mbol_unc = round(np.sqrt(self.mbol[1]**2+((2.5/np.log(10))*(self.distance[1]/self.distance[0]).value)**2), 3)
            
            # Update the attribute
            self.Mbol = Mbol, Mbol_unc
            
            
    def get_Teff(self):
        """Calculate the effective temperature
        """
        # Calculate Teff
        if self.distance is not None and self.radius is not None:
            Teff = np.sqrt(np.sqrt((self.Lbol[0]/(4*np.pi*ac.sigma_sb*self.radius[0]**2)).to(q.K**4))).astype(int)
            
            # Calculate Teff_unc
            Teff_unc = (Teff*np.sqrt((self.Lbol[1]/self.Lbol[0]).value**2 + (2*self.radius[1]/self.radius[0]).value**2)/4.).astype(int)
            
            # Update the attribute
            self.Teff = Teff, Teff_unc
    
    
    @staticmethod
    def group_spectra(spectra):
        """Puts a list of *spectra* into groups with overlapping wavelength arrays
        """
        groups, idx = [], []
        for N, S in enumerate(spectra):
            if N not in idx:
                group, idx = [S], idx + [N]
                for n, s in enumerate(spectra):
                    if n not in idx and any(np.where(np.logical_and(S.wave<s.wave[-1], S.wave>s.wave[0]))[0]):
                        group.append(s), idx.append(n)
                groups.append(group)
        return groups
        
        
    def logg_from_age(self):
        """Estimate the surface gravity from model isochrones given an age and Lbol
        """
        if self.age is not None and self.Lbol is not None:
            
            self.logg = tuple(iso.isochrone_interp(self.Lbol, self.age, yparam='logg'))
            
        else:
            print('Lbol={0.Lbol} and age={0.age}. Both are needed to calculate the surface gravity.'.format(self))
        
        
    def make_sed(self):
        """Construct the SED
        """
        # Calculate flux and calibrate
        self._calibrate_photometry()
        
        # Combine spectra and flux calibrate
        self._calibrate_spectra()
        
        # Make a Wein tail that goes to (almost) zero flux at (almost) zero wavelength
        self.wein = sp.Spectrum(np.array([0.00001])*self.wave_units, np.array([1E-30])*self.flux_units, np.array([1E-30])*self.flux_units)
        
        # Create Rayleigh Jeans Tail
        rj_wave = np.arange(np.min([self.app_phot_SED.wave[-1],12.]), 500, 0.1)*q.um
        rj_flux, rj_unc = u.blackbody(rj_wave, 1500*q.K, 100*q.K)

        # Normalize Rayleigh-Jeans tail to the longest wavelength photometric point
        rj_flux = (rj_flux*self.app_phot_SED.flux[-1]/rj_flux[0])*self.flux_units
        self.rj = sp.Spectrum(rj_wave, rj_flux, rj_unc)
        
        # Exclude photometric points with spectrum coverage
        if self.stitched_spectra is not None:
            covered = []
            for idx, i in enumerate(self.app_phot_SED.wave):
                for N,spec in enumerate(self.stitched_spectra):
                    if i<spec.wave[-1] and i>spec.wave[0]:
                        covered.append(idx)
            WP, FP, EP = [[i for n,i in enumerate(A) if n not in covered]*Q for A,Q in zip(self.app_phot_SED.spectrum, self.units)]
            
            if len(WP)==0:
                self.app_specphot_SED = None
            else:
                self.app_specphot_SED = sp.Spectrum(WP, FP, EP)
        else:
            self.app_specphot_SED = self.app_phot_SED

        # Construct full app_SED
        self.app_SED = np.sum([self.wein, self.app_specphot_SED, self.rj]+self.stitched_spectra)

        # Flux calibrate SEDs
        if self.distance is not None:
            self.abs_SED = self.app_SED.flux_calibrate(self.distance)

        # Calculate Fundamental Params
        self.fundamental_params()
        
        
    def mass_from_age(self, mass_units=q.Msun):
        """Estimate the surface gravity from model isochrones given an age and Lbol
        """
        if self.age is not None and self.Lbol is not None:
            
            mass = iso.isochrone_interp(self.Lbol, self.age, yparam='mass')
            
            self.mass = (mass[0]*q.Mjup).to(mass_units), (mass[1]*q.Mjup).to(mass_units)
            
        else:
            print('Lbol={0.Lbol} and age={0.age}. Both are needed to calculate the mass.'.format(self))
        
        
    @property
    def membership(self):
        """A property for membership"""
        return self._membership
    
    
    @membership.setter
    def membership(self, membership):
        """A setter for membership"""
        if membership is None:
            
            self._membership = None
            
        elif membership in AGES:
            
            # Set the membership!
            self._membership = membership
            
            if self.verbose:
                print('Setting membership to',self.membership)
            
            # Set the age
            self.age = AGES.get(membership)
            
        else:
            print('{} not valid. Supported memberships include {}.'.format(membership, ', '.join(AGES.keys())))
            
            
    @property
    def parallax(self):
        """A property for parallax"""
        return self._parallax
    
    
    @parallax.setter
    def parallax(self, parallax, parallax_units=q.mas):
        """A setter for parallax
        
        Parameters
        ----------
        parallax: sequence
            The (parallax, err) or (parallax, lower_err, upper_err)
        """
        # Make sure it's a sequence
        if not isinstance(parallax, (tuple, list, np.ndarray)) or len(parallax) not in [2,3]:
            raise TypeError('parallax must be a sequence of (value, error) or (value, lower_error, upper_error).')
            
        # Make sure the values are in time units
        try:
            _ = [i.to(q.mas) for i in parallax]
        except:
            raise TypeError("parallax values must be parallax units of astropy.units.quantity.Quantity, e.g. 'mas'")
        
        # Set the parallax
        self._parallax = parallax
        
        # Update the distance
        self._distance = u.pi2pc(*self.parallax)
        
        # Update the absolute photometry
        self._calibrate_photometry()
        
        # Update the flux calibrated spectra
        self._calibrate_spectra()
        
        
    @property
    def photometry(self):
        """A property for photometry"""
        self._photometry.sort('eff')
        return self._photometry
        
        
    def plot(self, app=True, photometry=True, spectra=True, integral=False, syn_photometry=True, blackbody=True, scale=['log','log'], output=False, **kwargs):
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
        syn_photometry: bool
            Plot the synthetic photometry
        blackbody: bool
            Polot the blackbody fit
        scale: array-like
            The (x,y) scales to plot, 'linear' or 'log'
        bokeh: bool
            Plot in Bokeh
        output: bool
            Just return figure, don't draw plot

        Returns
        =======
        bokeh.models.figure
            The SED plot
        """
        # Distinguish between apparent and absolute magnitude
        pre = 'app_' if app else 'abs_'

        # Calculate reasonable axis limits
        spec_SED = getattr(self, pre+'spec_SED')
        phot_SED = np.array([np.array([np.nanmean(self.photometry.loc[b][col].value) for b in list(set(self.photometry['band']))]) for col in ['eff',pre+'flux',pre+'flux_unc']])

        # Check for min and max phot data
        try:
            mn_xp, mx_xp, mn_yp, mx_yp = np.nanmin(phot_SED[0]), np.nanmax(phot_SED[0]), np.nanmin(phot_SED[1]), np.nanmax(phot_SED[1])
        except:
            mn_xp, mx_xp, mn_yp, mx_yp = 0.3, 18, 0, 1

        # Check for min and max spec data
        try:
            mn_xs, mx_xs = np.nanmin(spec_SED.wave), np.nanmax(spec_SED.wave)
            mn_ys, mx_ys = np.nanmin(spec_SED.flux[spec_SED.flux>0]), np.nanmax(spec_SED.flux[spec_SED.flux>0])
        except:
            mn_xs, mx_xs, mn_ys, mx_ys = 0.3, 18, 999, -999

        mn_x, mx_x, mn_y, mx_y = np.nanmin([mn_xp,mn_xs]), np.nanmax([mx_xp,mx_xs]), np.nanmin([mn_yp,mn_ys]), np.nanmax([mx_yp,mx_ys])

        # TOOLS = 'crosshair,resize,reset,hover,box,save'
        self.fig = figure(plot_width=800, plot_height=500, title=self.name, 
                     y_axis_type=scale[1], x_axis_type=scale[0], 
                     x_axis_label='Wavelength [{}]'.format(self.wave_units), 
                     y_axis_label='Flux Density [{}]'.format(str(self.flux_units)))

        # Plot spectra
        if spectra and len(self.spectra)>0:
            spec_SED = getattr(self, pre+'spec_SED')
            # source = ColumnDataSource(data=dict(x=spec_SED.wave, y=spec_SED.flux, z=spec_SED.unc))
            # hover = HoverTool(tooltips=[( 'wave', '$x'),( 'flux', '$y'),('unc','$z')], mode='vline')
            # self.fig.add_tools(hover)
            # self.fig.line('x', 'y', source=source, legend='Spectra')
            self.fig.line(spec_SED.wave, spec_SED.flux, legend='Spectra')

        # Plot photometry
        if photometry and self.photometry is not None:

            # Plot points with errors
            pts = np.array([(x,y,z) for x,y,z in np.array(self.photometry['eff',pre+'flux',pre+'flux_unc']) if not any([np.isnan(i) for i in [x,y,z]])]).T
            try:
                errorbar(self.fig, pts[0], pts[1], yerr=pts[2], point_kwargs={'fill_alpha':0.7, 'size':8}, legend='Photometry')
            except:
                pass

            # Plot saturated photometry
            pts = np.array([(x,y,z) for x,y,z in np.array(self.photometry['eff','app_flux','app_flux_unc']) if np.isnan(z) and not np.isnan(y)]).T
            try:
                errorbar(self.fig, pts[0], pts[1], point_kwargs={'fill_alpha':0, 'size':8}, legend='Nondetection')
            except:
                pass

        # # Plot synthetic photometry
        # if syn_photometry and self.syn_photometry is not None:
        #
        #     # Plot points with errors
        #     pts = np.array([(x,y,z) for x,y,z in np.array(self.syn_photometry['eff',pre+'flux',pre+'flux_unc']) if not np.isnan(z)]).T
        #     try:
        #         errorbar(self.fig, pts[0], pts[1], yerr=pts[2], point_kwargs={'fill_color':'red', 'fill_alpha':0.7, 'size':8}, legend='Synthetic Photometry')
        #     except:
        #         pass
        #
        # # Plot the SED with linear interpolation completion
        # if integral:
        #     full_SED = getattr(self, pre+'SED')
        #     self.fig.line(full_SED[0].value, full_SED[1].value, line_color='black', alpha=0.3, legend='Integral Surface')
        #     # plt.fill_between(full_SED[0].value, full_SED[1].value-full_SED[2].value, full_SED[1].value+full_SED[2].value, color='k', alpha=0.1)
        #
        # if blackbody and self.blackbody:
        #     fit_sed = getattr(self, self.bb_source)
        #     fit_sed = [i[fit_sed[0]<10] for i in fit_sed]
        #     bb_wav = np.linspace(np.nanmin(fit_sed[0]), np.nanmax(fit_sed[0]), 500)*q.um
        #     bb_flx, bb_unc = u.blackbody(bb_wav, self.Teff_bb*q.K, 100*q.K)
        #     bb_norm = np.trapz(fit_sed[1], x=fit_sed[0])/np.trapz(bb_flx.value, x=bb_wav.value)
        #     bb_wav = np.linspace(0.2, 30, 1000)*q.um
        #     bb_flx, bb_unc = u.blackbody(bb_wav, self.Teff_bb*q.K, 100*q.K)
        #     # print(bb_norm,bb_flx)
        #     fig.line(bb_wav.value, bb_flx.value*bb_norm, line_color='red', legend='{} K'.format(self.Teff_bb))

        self.fig.legend.location = "top_right"
        self.fig.legend.click_policy = "hide"
        self.fig.x_range = Range1d(mn_x*0.8, mx_x*1.2)
        self.fig.y_range = Range1d(mn_y*0.5, mx_y*2)

        if not output:
            show(self.fig)

        return self.fig
        
    
    def plot_photometry(self, pre='app', **kwargs):
        """Plot the photometry"""
        # Plot the photometry with uncertainties
        data = self.photometry[self.photometry[pre+'_flux_unc']>0]
        errorbar(self.fig, data['eff'], data[pre+'_flux'], yerr=data[pre+'_flux_unc'], color='navy', **kwargs)
        
        # Plot the photometry without uncertainties
        data = self.photometry[self.photometry[pre+'_flux_unc']==np.nan]
        errorbar(self.fig, data['eff'], data[pre+'_flux'], point_kwargs={'fill_color':'white', 'line_color':'navy'}, **kwargs)
        
        
    def plot_spectra(self, stitched=True, **kwargs):
        """Plot the spectra"""
        # Stitched or not
        specs = self.stitched_spectra if stitched else self.spectra
        
        # Plot each spectrum
        for spec in specs:
            spec.plot(fig=self.fig)
            
        
    @property
    def radius(self):
        """A property for radius"""
        return self._radius
    
    
    @radius.setter
    def radius(self, radius):
        """A setter for radius"""
        # Make sure it's a sequence
        if not isinstance(radius, (tuple, list, np.ndarray)) or len(radius) not in [2,3]:
            raise TypeError('Radius must be a sequence of (value, error) or (value, lower_error, upper_error).')
            
        # Make sure the values are in length units
        try:
            _ = [i.to(q.m) for i in radius]
        except:
            raise TypeError("Radius values must be length units of astropy.units.quantity.Quantity, e.g. 'Rjup'")
        
        # Set the radius!
        self._radius = tuple(radius)
        
        if self.verbose:
            print('Setting radius to',self.radius)
        
        # Update the things that depend on radius!
        
        
    def radius_from_spectral_type(self):
        """Estimate the radius from CMD plot
        """
        pass
        
        
    def radius_from_age(self, radius_units=q.Rsun):
        """Estimate the radius from model isochrones given an age and Lbol
        """
        if self.age is not None and self.Lbol is not None:
            
            radius = iso.isochrone_interp(self.Lbol, self.age)
            
            self.radius = (radius[0]*q.Rjup).to(radius_units), (radius[1]*q.Rjup).to(radius_units)
            
        else:
            print('Lbol={0.Lbol} and age={0.age}. Both are needed to calculate the radius.'.format(self))
        
        
    @property
    def results(self):
        """A property for displaying the results"""
        # Make the SED to get the most recent results
        self.make_sed()
        
        # Get the results
        rows = []
        for param in ['name', 'age', 'distance', 'parallax', 'radius',\
                      'spectral_type', 'membership', 'fbol', 'mbol', \
                      'Lbol', 'Lbol_sun', 'Mbol', 'logg', 'mass', 'Teff']:
            
            # Get the values and format
            attr = getattr(self, param, None)
            
            if attr is None:
                attr = '--'
            
            if isinstance(attr, (tuple,list)):
                val, unc = attr[:2]
                unit = val.unit if hasattr(val, 'unit') else '--'
                val = val.value if hasattr(val, 'unit') else val
                unc = unc.value if hasattr(unc, 'unit') else unc
                if val<1E-4 or val>1e5:
                    val = float('{:.2e}'.format(val))
                    unc = float('{:.2e}'.format(unc))
                rows.append([param, val, unc, unit])
                
            elif isinstance(attr, str):
                rows.append([param, attr, '--', '--'])
                
            else:
                pass
        
        return at.Table(np.asarray(rows), names=('param','value','unc','units'))
        
        
    @property
    def spectra(self):
        """A property for spectra"""
        return self._spectra
        
        
    @property
    def spectral_type(self):
        """A property for spectral_type"""
        return self._spectral_type
    
    
    @spectral_type.setter
    def spectral_type(self, spectral_type, spectral_type_unc=None, gravity=None, lum_class='V', prefix=None):
        """A setter for spectral_type"""
        # Make sure it's a sequence
        if isinstance(spectral_type, str):
            self.SpT = spectral_type
            spec_type = u.specType(spectral_type)
            spectral_type, spectral_type_unc, prefix, gravity, lum_class = spec_type
            
        # Set the spectral_type!
        self._spectral_type = spectral_type, spectral_type_unc or 0.5
        self.luminosity_class = lum_class
        self.gravity = gravity or None
        self.prefix = prefix or None
        
        # Set the age if not explicitly set
        if self.age is None and self.gravity is not None:
            if gravity in ['b','beta','g','gamma']:
                self.age = 225*q.Myr, 75*q.Myr
                
            else:
                print("{} is an invalid gravity. Please use 'beta' or 'gamma' instead.".format(gravity))
        
        # Update the radius
        if self.radius is None:
            # TODO self.radius = get_radius()
            pass
    
    
    @property
    def wave_units(self):
        """A property for wave_units"""
        return self._wave_units
    
    
    @wave_units.setter
    def wave_units(self, wave_units):
        """A setter for wave_units
        
        Parameters
        ----------
        wave_units: astropy.units.quantity.Quantity
            The astropy units of the SED wavelength
        """
        # Make sure it's a quantity
        if not isinstance(wave_units, (q.core.PrefixUnit, q.core.Unit, q.core.CompositeUnit)):
            raise TypeError('wave_units must be astropy.units.quantity.Quantity')
            
        # Make sure the values are in length units
        try:
            wave_units.to(q.um)
        except:
            raise TypeError("wave_units must be a unit of length, e.g. 'um'")
        
        # Set the wave_units!
        self._wave_units = wave_units
        self.units = [self._wave_units, self._flux_units, self._flux_units]
        
        # Recalibrate the data
        self._calibrate_photometry()
        self._calibrate_spectra()
    
    
    # def write(self, dirpath, app=False, spec=True, phot=False):
    #     """
    #     Exports a file of photometry and a file of the composite spectra with minimal data headers
    #
    #     Parameters
    #     ----------
    #     dirpath: str
    #       The directory path to place the file
    #     app: bool
    #       Write apparent SED data
    #     spec: bool
    #       Write a file for the spectra with wavelength, flux and uncertainty columns
    #     phot: bool
    #       Write a file for the photometry with
    #     """
    #     if spec:
    #         try:
    #             spec_data = self.app_spec_SED.spectrum if app else self.abs_spec_SED.spectrum
    #             if dirpath.endswith('.txt'):
    #                 specpath = dirpath
    #             else:
    #                 specpath = dirpath + '{} SED.txt'.format(self.name)
    #
    #             header = '{} {} spectrum (erg/s/cm2/A) as a function of wavelength (um)'.format(self.name, 'apparent' if app else 'flux calibrated')
    #
    #             np.savetxt(specpath, np.asarray(spec_data).T, header=header)
    #
    #         except IOError:
    #             print("Couldn't print spectra.")
    #
    #     if phot:
    #         try:
    #             phot = self.photometry
    #
    #             if dirpath.endswith('.txt'):
    #                 photpath = dirpath
    #             else:
    #                 photpath = dirpath + '{} phot.txt'.format(self.name)
    #
    #             phot.write(photpath, format='ipac')
    #
    #         except IOError:
    #             print("Couldn't print photometry.")

    # =========================================================================
    # =========================================================================
    # =========================================================================
        
        
    # def get_syn_photometry(self, bands=[], plot=False):
    #     """
    #     Calculate the synthetic magnitudes
    #
    #     Parameters
    #     ----------
    #     bands: sequence
    #         The list of bands to calculate
    #     plot: bool
    #         Plot the synthetic mags
    #     """
    #     try:
    #         if not any(bands):
    #             bands = FILTERS['Band']
    #
    #         # Only get mags in regions with spectral coverage
    #         syn_mags = []
    #         for spec in [i.as_void() for i in self.piecewise]:
    #             spec = [Q*(i.value if hasattr(i,'unit') else i) for i,Q in zip(spec,[self.wave_units,self.flux_units,self.flux_units])]
    #             syn_mags.append(s.all_mags(spec, bands=bands, plot=plot))
    #
    #         # Stack the tables
    #         self.syn_photometry = at.vstack(syn_mags)
    #
    #     except:
    #         print('No spectral coverage to calculate synthetic photometry.')
    #
    # def fit_blackbody(self, fit_to='app_phot_SED', epsilon=0.1, acc=5):
    #     """
    #     Fit a blackbody curve to the data
    #
    #     Parameters
    #     ==========
    #     fit_to: str
    #         The attribute name of the [W,F,E] to fit
    #     epsilon: float
    #         The step size
    #     acc: float
    #         The acceptible error
    #     """
    #     # Get the data
    #     data = getattr(self, fit_to)
    #
    #     # Remove NaNs
    #     print(data)
    #     data = np.array([(x,y,z) for x,y,z in zip(*data) if not any([np.isnan(i) for i in [x,y,z]]) and x<10]).T
    #     print(data)
    #     # Initial guess
    #     try:
    #         teff = self.Teff.value
    #     except:
    #         teff = 3000
    #     init = blackbody(temperature=teff)
    #
    #     # Fit the blackbody
    #     fit = fitting.LevMarLSQFitter()
    #     bb = fit(init, data[0], data[1]/np.nanmax(data[1]), epsilon=epsilon, acc=acc)
    #
    #     # Store the results
    #     try:
    #         self.Teff_bb = int(bb.temperature.value)
    #         self.bb_source = fit_to
    #         self.blackbody = bb
    #         print('\nBlackbody fit: {} K'.format(self.Teff_bb))
    #     except:
    #         print('\nNo blackbody fit.')
        
            
        
def errorbar(fig, x, y, xerr='', yerr='', color='black', point_kwargs={}, error_kwargs={}, legend=''):
    """
    Hack to make errorbar plots in bokeh
    
    Parameters
    ==========
    x: sequence
        The x axis data
    y: sequence
        The y axis data
    xerr: sequence (optional)
        The x axis errors
    yerr: sequence (optional)
        The y axis errors
    color: str
        The marker and error bar color
    point_kwargs: dict
        kwargs for the point styling
    error_kwargs: dict
        kwargs for the error bar styling
    legend: str
        The text for the legend
    """
    fig.circle(x, y, color=color, legend=legend, **point_kwargs)

    if xerr!='':
        x_err_x = []
        x_err_y = []
        for px, py, err in zip(x, y, xerr):
            x_err_x.append((px - err, px + err))
            x_err_y.append((py, py))
        fig.multi_line(x_err_x, x_err_y, color=color, **error_kwargs)

    if yerr!='':
        y_err_x = []
        y_err_y = []
        for px, py, err in zip(x, y, yerr):
            y_err_x.append((px, px))
            y_err_y.append((py - err, py + err))
        fig.multi_line(y_err_x, y_err_y, color=color, **error_kwargs)

def test(n=1):
    """
    Run a test target
    """
    from astrodbkit import astrodb
    from SEDkit import sed
    db = astrodb.Database('/Users/jfilippazzo/Documents/Modules/BDNYCdevdb/bdnycdev.db')
    
    if n==1:
        source_id = 2
        from_dict = {'spectra':3176, 'photometry':'*', 'parallaxes':575, 'sources':source_id}
    if n==2:
        source_id = 86
        from_dict = {'spectra':[379,1580,2726], 'photometry':'*', 'parallaxes':247, 'spectral_types':277, 'sources':86}
    if n==3:
        source_id = 2051
        from_dict = {}
    
    x = sed.MakeSED(source_id, db, from_dict=from_dict)
    x.get_syn_photometry()
    x.plot()
    
    return x

@custom_model
def blackbody(wavelength, temperature=2000):
    """
    Generate a blackbody of the given temperature at the given wavelengths
    
    Parameters
    ----------
    wavelength: array-like
        The wavelength array [um]
    temperature: float
        The temperature of the star [K]
    
    Returns
    -------
    astropy.quantity.Quantity
        The blackbody curve
    """
    wavelength = q.Quantity(wavelength, "um")
    temperature = q.Quantity(temperature, "K")
    max_val = blackbody_lambda((b_wien/temperature).to(q.um),temperature).value
    return blackbody_lambda(wavelength, temperature).value/max_val

AGES = {'TW Hya': (14*q.Myr, 6*q.Myr),
       'beta Pic': (17*q.Myr, 5*q.Myr),
       'Tuc-Hor': (25*q.Myr, 15*q.Myr),
       'Columba': (25*q.Myr, 15*q.Myr),
       'Carina': (25*q.Myr, 15*q.Myr),
       'Argus': (40*q.Myr, 10*q.Myr),
       'AB Dor': (85*q.Myr, 35*q.Myr),
       'Pleiades': (120*q.Myr, 10*q.Myr)}