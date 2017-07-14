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
import matplotlib.pyplot as plt
from . import utilities as u
from svo_filters import svo

FILTERS = svo.filters()
FILTERS.add_index('Band')

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

class MakeSED(object):
    def __init__(self, source_id, db, from_dict='', pi='', dist='', pop=[], \
        age='', radius='', membership='', flux_units=q.erg/q.s/q.cm**2/q.AA, wave_units=q.um):
        """
        Pulls all available data from the BDNYC Data Archive, 
        constructs an SED, and stores all calculations at *pickle_path*
        
        Parameters
        ----------
        source_id: int, str
            The *source_id*, *unum*, *shortname* or *designation* for any 
            source in the database.
        db: astrodbkit.astrodb.Database, dict
            The database instance to retreive data from or a dictionary
            of astropy tables to mimick the db query
        spec_ids: list, tuple (optional)
            A sequence of the ids from the SPECTRA table to plot. 
            Uses any available spectra if no list is given. Uses no spectra 
            if 'None' is given
        
        pop: sequence (optional)
            Photometric bands to exclude from the SED
        
        """
        # TODO: resolve source_id in database given id, (ra,dec), name, etc.
        # source_id = db._resolve_source_id()
        
        # Get the data for the source from the dictionary of ids
        if isinstance(from_dict, dict):
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
                
        # =====================================================================
        # Metadata
        # =====================================================================
        
        # Set some attributes
        self.flux_units = flux_units
        self.wave_units = wave_units
        
        # =====================================================================
        # Distance
        # =====================================================================
        
        # Index and add units
        fill = np.zeros(len(self.parallaxes))
        self.parallaxes['parallax'].unit = q.mas
        self.parallaxes['parallax_unc'].unit = q.mas
        
        # Add distance columns to the parallaxes table
        self.parallaxes.add_column(at.Column(fill, 'distance', unit=q.pc))
        self.parallaxes.add_column(at.Column(fill, 'distance_unc', unit=q.pc))
        
        # Check for input parallax or distance and set adopted
        self.parallaxes['adopted'] = fill
        if pi:
            self.parallaxes.add_row({'parallax':pi[0], 'parallax_unc':pi[1], \
                'adopted':1, 'publication_shortname':'Input'})
        elif dist:
            self.parallaxes.add_row({'distance':dist[0], 'distance_unc':dist[1],\
                'adopted':1, 'publication_shortname':'Input'})
        else:
            self.parallaxes[0]['adopted'] = 1
            
        # Calculate missing distance or parallax
        for row in self.parallaxes:
            if row['parallax'].value and not row['distance'].value:
                distance = u.pi2pc(row['parallax'], row['parallax_unc'])
                row['distance'] = distance[0]
                row['distance_unc'] = distance[1]
                
            elif row['distance'].value and not row['parallax'].value:
                parallax = u.pi2pc(row['distance'], row['distance_unc'], pc2pi=True)
                row['parallax'] = parallax[0]
                row['parallax_unc'] = parallax[1]
                
            else:
                pass
                
        self.parallaxes.add_index('adopted')
        
        # =====================================================================
        # Spectral Type
        # =====================================================================
        # TODO
        self.gravity_suffix = ''
        
        # =====================================================================
        # Age
        # =====================================================================
        
        # Retreive age data from input NYMG membership, input age range, or age estimate
        if isinstance(age, tuple):
            self.age_min, self.age_max = age
        elif membership in NYMG:
            self.age_min, self.age_max = (NYMG[membership]['age_min'], NYMG[membership]['age_min'])*q.Myr
        elif self.gravity_suffix:
            self.age_min, self.age_max = (0.01, 0.15)*q.Gyr
        else:
            self.age_min, self.age_max = (0.5, 10)*q.Gyr
            
        # =====================================================================
        # Radius
        # =====================================================================
        
        # Use radius if given
        self.radius, self.radius_unc = radius or ['', '']
        
        # =====================================================================
        # Photometry
        # =====================================================================
        
        # Index and add units
        fill = np.zeros(len(self.photometry))
        self.photometry['band'] = at.Column([b.replace('_','.') for b in list(self.photometry['band'])])
        self.photometry.add_index('band')
        self.photometry.rename_column('magnitude','app_magnitude')
        self.photometry.rename_column('magnitude_unc','app_magnitude_unc')
        self.photometry['app_magnitude'].unit = q.mag
        self.photometry['app_magnitude_unc'].unit = q.mag
        
        # Add effective wavelengths to the photometry table
        self.photometry.add_column(at.Column(fill, 'eff', unit=wave_units))
        for row in self.photometry:
            try:
                band = FILTERS.loc[row['band']]
                row['eff'] = band['WavelengthEff']*q.Unit(band['WavelengthUnit'])
            except IOError:
                row['eff'] = np.nan
            
        # Add absolute magnitude columns to the photometry table
        self.photometry.add_column(at.Column(fill, 'abs_magnitude', unit=q.mag))
        self.photometry.add_column(at.Column(fill, 'abs_magnitude_unc', unit=q.mag))
        
        # Calculate absolute mags and add to the photometry table
        d = self.parallaxes.loc[1]
        for row in self.photometry:
            M, M_unc = u.flux_calibrate(row['app_magnitude'], d['distance'], \
                row['app_magnitude_unc'], d['distance_unc'])
            row['abs_magnitude'] = M
            row['abs_magnitude_unc'] = M_unc
            
        # Add flux density columns to the photometry table
        for colname in ['app_flux','app_flux_unc','abs_flux','abs_flux_unc']:
            self.photometry.add_column(at.Column(fill, colname, unit=self.flux_units))
            
        # Calculate fluxes and add to the photometry table
        for i in ['app_','abs_']:
            for row in self.photometry:
                ph_flux = u.mag2flux(row['band'], row[i+'magnitude'],\
                    sig_m=row[i+'magnitude_unc'])
                row[i+'flux'] = ph_flux[0]
                row[i+'flux_unc'] = ph_flux[1]
                
        # Make relative and absolute photometric SEDs
        self.app_phot_SED = np.array([self.photometry['eff'], self.photometry['app_flux'], self.photometry['app_flux_unc']])
        self.abs_phot_SED = np.array([self.photometry['eff'], self.photometry['abs_flux'], self.photometry['abs_flux_unc']])
                
        # =====================================================================
        # Spectra
        # =====================================================================
        
        # Index and add units
        fill = np.zeros(len(self.spectra))
        self.spectra.add_index('id')
        
        # Prepare apparent spectra
        wav_list, flx_list, unc_list = [], [], []
        for row in self.spectra:
            
            # Unpack the spectrum
            w, f, e = row['spectrum'].data[:3]
            
            # Convert log units to linear
            if row['flux_units'].startswith('log '):
                f, e = 10**f, 10**e
                row['flux_units'] = row['flux_units'].replace('log ', '')
            if row['wavelength_units'].startswith('log '):
                w = 10**w
                row['wavelength_units'] = row['wavelength_units'].replace('log ', '')
            
            # Make sure the wavelength units are right
            w = w*u.str2Q(row['wavelength_units']).to(self.wave_units).value
            
            # Convert F_nu to F_lam if necessary
            if row['flux_units']=='Jy':
                f = u.fnu2flam(f*q.Jy, w*self.wave_units, units=self.flux_units).value
                e = u.fnu2flam(e*q.Jy, w*self.wave_units, units=self.flux_units).value
            
            # Normalize the spectra to the available photometry
            w, f, e = s.norm_to_mags([w,f,e], self.photometry)
            
            # Apply desired units
            w = w*self.wave_units
            f = f*self.flux_units
            e = e*self.flux_units
            
            # Add the data to the lists
            wav_list.append(u.ArrayWrapper(w))
            flx_list.append(u.ArrayWrapper(f))
            unc_list.append(u.ArrayWrapper(e))
        
        # Add the lists to the table
        self.spectra['wavelength'] = wav_list
        self.spectra['app_flux'] = flx_list
        self.spectra['app_flux_unc'] = unc_list
        
        # Group overlapping spectra and make composites where possible to form peacewise spectrum for flux calibration
        all_spectra = [[i.data for i in j] for j in self.spectra[['wavelength','app_flux','app_flux_unc']]]
        if len(all_spectra) > 1:
            groups, piecewise = u.group_spectra(all_spectra), []
            for group in groups:
                composite = u.make_composite([[spec[0]*self.wave_units, 
                                               spec[1]*self.flux_units, 
                                               spec[2]*self.flux_units] for spec in group])
                piecewise.append(composite)
                
        # If only one spectrum, no need to make composite
        elif len(all_spectra) == 1:
            piecewise = all_spectra
            
        # If no spectra, forget it
        else:
            piecewise = []
            print('No spectra available for SED.')
            
        # Add piecewise spectra to table
        self.piecewise = at.Table([[spec[i] for spec in piecewise] for i in [0,1,2]], 
                                  names=['wavelength','app_flux','app_flux_unc'])
                                  
        # Normalize the self.spectra and self.piecewise spectra to all covered photometric bands
        # TODO
        
        # =====================================================================
        # Construct SED
        # =====================================================================
        # TODO
        
        # =====================================================================
        # Calculate Fundamental Params
        # =====================================================================
        # TODO
        # self.fundamental_params(**kwargs)
        
        # =====================================================================
        # Print some stuff
        # =====================================================================
        # TODO
        
        # =====================================================================
        # Save the data to file for cmd.py to read
        # =====================================================================
        # TODO
        
    
    def fundamental_params(self, age='', nymg='', radius='', evo_model='hybrid_solar_age'):
        """
        Calculate the fundamental parameters of the current SED
        
        Parameters
        ----------
        age: tuple, list (optional)
            The lower and upper age limits of the source in astropy.units
        nymg: str (optional)
            The nearby young moving group name
        radius: tuple, list (optional)
            The lower and upper age limits of the source in astropy.units
        evo_model: str
            The evolutionary model to use
        """
        self.get_Lbol()
        self.get_Mbol()
        self.get_Teff()
    
    def get_mbol(self, L_sun=3.86E26*q.W, Mbol_sun=4.74):
        """
        Calculate the apparent bolometric magnitude of the SED
        """
        self.mbol = round(-2.5*np.log10(self.fbol.value)-11.482, 3)
        self.mbol_unc = round((2.5/np.log(10))*(self.fbol_unc/self.fbol).value, 3)
        
    def get_Mbol(self):
        """
        Calculate the absolute bolometric magnitude of the SED
        """
        self.Mbol = round(self.mbol-5*np.log10((self.distance/10*q.pc).value), 3)
        self.Mbol_unc = round(np.sqrt(self.mbol_unc**2+((2.5/np.log(10))*(self.distance_unc/self.distance).value)**2), 3)
        
    def get_fbol(self):
        """
        Calculate the bolometric flux of the SED
        """
        self.fbol = (np.trapz(self.app_SED.flux, x=self.app_SED.wavelength)).to(self.flux_units*self.wave_units)
        self.fbol_unc = np.sqrt(np.sum((self.app_SED.unc*np.gradient(self.app_SED.wavelength)).to(self.flux_units*self.wave_units).value**2))
        
    def get_Lbol(self):
        """
        Calculate the bolometric luminosity of the SED
        """
        self.Lbol = (4*np.pi*self.fbol*self.distance**2).to(q.erg/q.s)
        self.Lbol_unc = self.Lbol*np.sqrt((self.fbol_unc/self.fbol).value**2+(2*self.distance_unc/self.distance).value**2)

        self.Lbol_sun = round(np.log10((self.Lbol/ac.L_sun).decompose().value), 3)
        self.Lbol_sun_unc = round(abs(self.Lbol_unc/(self.Lbol*np.log(10))).value, 3)
        
    def get_Teff(self, radius, radius_unc):
        """
        Calculate the effective temperature of the SED

        Parameters
        ----------
        r: astropy.quantity
            The radius of the source in units of R_Jup
        sig_r: astropy.quantity
            The uncertainty in the radius
        """
        self.Teff = np.sqrt(np.sqrt((self.Lbol/(4*np.pi*ac.sigma_sb*radius**2)).to(q.K**4))).round(0)
        self.Teff_unc = (self.Teff*np.sqrt((self.Lbol_unc/self.Lbol).value**2 + (2*radius_unc/radius).value**2)/4.).round(0)
    
    def plot(self, photometry=True, spectra=True, app=False, scale=['log','log'], **kwargs):
        """
        Plot the SED
        
        Parameters
        ----------
        photometry: bool
            Plot the photometry
        spectra: bool
            Plot the spectra
        app: bool
            Plot the apparent SED instead of absolute
        scale: array-like
            The (x,y) scales to plot, 'linear' or 'log'
        """
        # Make the figure
        plt.figure(**kwargs)
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        
        # Set the x and y scales
        plt.xscale(scale[0], nonposx='clip')
        plt.yscale(scale[1], nonposy='clip')
        
        # Distinguish between apparent and absolute magnitude
        pre = 'app_' if app else 'abs_'
        
        # Plot photometry
        if photometry:
            phot_SED = self.app_phot_SED if app else self.abs_phot_SED
            plt.errorbar(phot_SED[0], phot_SED[1], yerr=phot_SED[2], marker='o', ls='None', **kwargs)
                
        # Plot spectra
        if spectra:
            spec_SED = self.app_spec_SED if app else self.abs_spec_SED
            plt.step(spec_SED[0], spec_SED[1], **kwargs)
        
NYMG = {'TW Hya': {'age_min': 8, 'age_max': 20, 'age_ref': 0},
         'beta Pic': {'age_min': 12, 'age_max': 22, 'age_ref': 0},
         'Tuc-Hor': {'age_min': 10, 'age_max': 40, 'age_ref': 0},
         'Columba': {'age_min': 10, 'age_max': 40, 'age_ref': 0},
         'Carina': {'age_min': 10, 'age_max': 40, 'age_ref': 0},
         'Argus': {'age_min': 30, 'age_max': 50, 'age_ref': 0},
         'AB Dor': {'age_min': 50, 'age_max': 120, 'age_ref': 0},
         'Pleiades': {'age_min': 110, 'age_max': 130, 'age_ref': 0}}