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
import astropy.constants as ac
from astropy.modeling.models import custom_model
from astropy.modeling import models, fitting
from astropy.analytic_functions import blackbody_lambda
from astropy.constants import b_wien
from . import utilities as u
from . import syn_phot as s
from svo_filters import svo
from bokeh.models import HoverTool, Label, Range1d, BoxZoomTool, ColumnDataSource

from bokeh.plotting import figure, output_file, show, save

FILTERS = svo.filters()
FILTERS.add_index('Band')

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

class MakeSED(object):
    """
    A class to construct spectral energy distributions and calculate fundamental paramaters of stars
    
    Attributes
    ==========
    Lbol: astropy.units.quantity.Quantity
        The bolometric luminosity [erg/s]
    Lbol_sun: astropy.units.quantity.Quantity
        The bolometric luminosity [L_sun]
    Lbol_sun_unc: astropy.units.quantity.Quantity
        The bolometric luminosity [L_sun] uncertainty
    Lbol_unc: astropy.units.quantity.Quantity
        The bolometric luminosity [erg/s] uncertainty
    Mbol: float
        The absolute bolometric magnitude
    Mbol_unc: float
        The absolute bolometric magnitude uncertainty
    SpT: float
        The string spectral type
    Teff: astropy.units.quantity.Quantity
        The effective temperature calculated from the SED
    Teff_bb: astropy.units.quantity.Quantity
        The effective temperature calculated from the blackbody fit
    Teff_unc: astropy.units.quantity.Quantity
        The effective temperature calculated from the SED uncertainty
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
    distance_unc: astropy.units.quantity.Quantity
        The target distance uncertainty
    fbol: astropy.units.quantity.Quantity
        The apparent bolometric flux [erg/s/cm2]
    fbol_unc: astropy.units.quantity.Quantity
        The apparent bolometric flux [erg/s/cm2] uncertainty
    flux_units: astropy.units.quantity.Quantity
        The desired flux density units
    gravity: str
        The surface gravity suffix
    mbol: float
        The apparent bolometric magnitude
    mbol_unc: float
        The apparent bolometric magnitude uncertainty
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
    radius_unc: astropy.units.quantity.Quantity
        The target radius uncertainty
    sources: astropy.table.QTable
        The table of sources (with only one row of cource)
    spectra: astropy.table.QTable
        The table of spectra
    spectral_type: float
        The numeric spectral type, where 0-99 corresponds to spectral types O0-Y9
    spectral_type_unc: float
        The numeric spectral type uncertainty
    spectral_types: astropy.table.QTable
        The table of spectral types
    suffix: str
        The spectral type suffix
    syn_photometry: astropy.table.QTable
        The table of calcuated synthetic photometry
    wave_units: astropy.units.quantity.Quantity
        The desired wavelength units
    """
    def __init__(self, source_id, db, from_dict='', wave_units=q.um, flux_units=q.erg/q.s/q.cm**2/q.AA, SED_trim=[], SED_split=[], name='', verbose=True, **kwargs):
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
        from_dict: dict (optional)
            A dictionary of the {table_name:[ids], ...} to construct the SED
        wave_units: astropy.units.quantity.Quantity
            The wavelength units to use
        flux_units: astropy.units.quantity.Quantity
            The flux density units to use
        SED_trim: sequence (optional)
            A sequence of (wave_min, wave_max) sequences to trim the SED by
        SED_split: sequence (optional)
            Wavelength positions to split spectra at so the pieces are independently normalized
        name: str (optional)
            A name for the target
        verbose: bool
            Print some diagnostic stuff
        
        Example 1
        ---------
        from astrodbkit import astrodb
        from SEDkit import sed
        db = astrodb.Database('/Users/jfilippazzo/Documents/Modules/BDNYCdevdb/bdnycdev.db')
        x = sed.MakeSED(2051, db)
        x.plot()
        
        Example 2
        ---------
        from astrodbkit import astrodb
        from SEDkit import sed
        db = astrodb.Database('/Users/jfilippazzo/Documents/Modules/BDNYCdevdb/bdnycdev.db')
        from_dict = {'spectra':3176, 'photometry':'*', 'parallaxes':575, 'sources':2}
        x = sed.MakeSED(2, db, from_dict=from_dict)
        x.plot()
        
        """
        # TODO: resolve source_id in database given id, (ra,dec), name, etc.
        # source_id = db._resolve_source_id()
        
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
                
        # =====================================================================
        # Metadata
        # =====================================================================
        
        # Print source data
        try:
            self.name = name or self.sources['names'][0].split(',')[0].strip()
        except:
            self.name = 'Source {}'.format(source_id)
        print('='*100)
        print(self.name,'='*(99-len(self.name)))
        print('='*100,'\n')
        self.sources[['names','ra','dec','publication_shortname']].pprint()
        
        # Set some attributes
        self.flux_units = flux_units
        self.wave_units = wave_units
        units = [self.wave_units,self.flux_units,self.flux_units]
        
        # =====================================================================
        # Distance
        # =====================================================================
        
        # Punt if no distance info
        if len(self.parallaxes)==0 and kwargs.get('pi')=='' and kwargs.get('dist')=='':
            
            print("\nNo distance for this source")
            self.distance = self.distance_unc = ''
            
        else:
            
            self.process_parallaxes(**kwargs)
        
        # =====================================================================
        # Spectral Type
        # =====================================================================
        
        # Punt if no SpT info
        if len(self.spectral_types)==0 and kwargs.get('spt')=='':
            
            print("\nNo spectral type for this source")
            self.spectral_type = self.spectral_type_unc = self.gravity = self.suffix = self.SpT = ''
            
        else:
            
            self.process_spectral_types(**kwargs)
        
        # =====================================================================
        # Age
        # =====================================================================
        self.process_age(**kwargs)
            
        # =====================================================================
        # Radius
        # =====================================================================
        self.process_radius(**kwargs)
        
        # =====================================================================
        # Photometry
        # =====================================================================
        
        # Punt if no photometry
        if len(self.photometry)==0:
            
            print('\nNo photometry for this source.')
            
        else:
            self.process_photometry(**kwargs)
            
        # Make apparent photometric SED from photometry with uncertainties
        with_unc = self.photometry[(self.photometry['app_flux']>0)&(self.photometry['app_flux_unc']>0)]
        self.app_phot_SED = np.array([np.array([np.nanmean(with_unc.loc[b][col].value) for b in list(set(with_unc['band']))]) for col in ['eff','app_flux','app_flux_unc']])
        WP0, FP0, EP0 = u.finalize_spec(self.app_phot_SED, wave_units=self.wave_units, flux_units=self.flux_units)
        
        # =====================================================================
        # Blackbody fit
        # =====================================================================
        
        # Set up empty blackbody fit
        self.blackbody = None
        self.Teff_bb = None
        self.bb_source = None
        
        # Fit blackbody to the photometry
        self.fit_blackbody()
        
        # =====================================================================
        # Spectra
        # =====================================================================
        
        if len(self.spectra)==0:
            self.processed_spectra = []
            print('\nNo spectra available for this source')
            
        else:
            self.process_spectra()
        
        # =====================================================================
        # Construct SED
        # =====================================================================
        
        # Group overlapping spectra and make composites where possible
        # to form peacewise spectrum for flux calibration
        if len(self.processed_spectra) > 1:
            groups, piecewise = u.group_spectra(self.processed_spectra), []
            for group in groups:
                composite = u.make_composite([[spec[0]*self.wave_units, spec[1]*self.flux_units, spec[2]*self.flux_units] for spec in group])
                piecewise.append(composite)
                
        # If only one spectrum, no need to make composite
        elif len(self.processed_spectra) == 1:
            piecewise = np.copy(self.processed_spectra)
            
        # If no spectra, forget it
        else:
            piecewise = []
            print('No spectra available for SED.')
            
        # Splitting
        keepers = []
        if SED_split:
            for pw in piecewise:
                wavs = list(filter(None, [np.where(pw[0]<i)[0][-1] if pw[0][0]<i and pw[0][-1]>i else None for i in SED_split]))
                keepers += map(list, zip(*[np.split(i, list(wavs)) for i in pw]))
                
            piecewise = np.copy(keepers)
            
        # Create Rayleigh Jeans Tail
        RJ_wav = np.arange(np.min([self.app_phot_SED[0][-1],12.]), 500, 0.1)*q.um
        RJ_flx, RJ_unc = u.blackbody(RJ_wav, self.Teff_bb*q.K, 100*q.K)

        # Normalize Rayleigh-Jeans tail to the longest wavelength photometric point
        RJ_flx *= self.app_phot_SED[1][-1]/RJ_flx[0].value
        RJ = u.finalize_spec([RJ_wav, RJ_flx, RJ_unc], wave_units=self.wave_units, flux_units=self.flux_units)
        
        # Normalize the composite spectra to the available photometry
        for n,spec in enumerate(piecewise):
            pw = s.norm_to_mags(spec, self.photometry, extend=RJ)
            
            # Add NaN to gaps
            pw[1][0] *= np.nan
            pw[1][-1] *= np.nan
                
            piecewise[n] = pw
            
        # Add piecewise spectra to table
        self.piecewise = at.Table([[spec[i] for spec in piecewise] for i in [0,1,2]], names=['wavelength','app_flux','app_flux_unc'])
        self.piecewise['wavelength'].unit = self.wave_units
        self.piecewise['app_flux'].unit = self.flux_units
        self.piecewise['app_flux_unc'].unit = self.flux_units
        
        # Concatenate pieces and finalize composite spectrum with units
        if self.piecewise:
            self.app_spec_SED = (W, F, E) = [np.asarray(i)*Q for i,Q in zip(u.trim_spectrum([np.concatenate(j) for j in [list(self.piecewise[col]) for col in ['wavelength', 'app_flux', 'app_flux_unc']]], SED_trim), units)]
        else:
            W, F, E = W0, F0, E0 = self.app_spec_SED = [Q*np.array([]) for Q in units]
        
        # Exclude photometric points with spectrum coverage
        if self.piecewise:
            covered = []
            for n, i in enumerate(WP0):
                for N,spec in enumerate(self.piecewise):
                    wav_mx = spec['wavelength'][-1]*q.um if isinstance(spec['wavelength'][-1],float) else spec['wavelength'][-1]
                    wav_mn = spec['wavelength'][0]*q.um if isinstance(spec['wavelength'][0],float) else spec['wavelength'][0]
                    if i<wav_mx and i>wav_mn:
                        covered.append(n)
            WP, FP, EP = [[i for n,i in enumerate(A) if n not in covered]*Q for A,Q in zip(self.app_phot_SED, units)]
        else:
            WP, FP, EP = WP0, FP0, EP0
            
        # Use zero flux at zero wavelength from bluest data point for Wein tail approximation
        Wein = [np.array([0.00001])*self.wave_units, np.array([1E-30])*self.flux_units, np.array([1E-30])*self.flux_units]
        
        # Create spectra + photometry SED for model fitting
        if self.spectra or self.photometry:
            specPhot = u.finalize_spec([i*Q for i,Q in zip([j.value for j in [np.concatenate(i) for i in [[pp, ss] for pp, ss in zip([WP, FP, EP], [W, F, E])]]], units)])
        else:
            specPhot = [[999*q.um], None, None]
        
        # Create full SED from Wien tail, spectra, linear interpolation between photometry, and Rayleigh-Jeans tail
        try:
            self.app_SED = [np.concatenate(i).value for i in [[ww[Wein[0] < min([min(i) for i in [WP, specPhot[0] or [999 * q.um]] if any(i)])], sp, bb[RJ[0] > max([max(i) for i in [WP, specPhot[0] or [-999 * q.um]] if any(i)])]] for ww, bb, sp in zip(Wein, RJ, specPhot)]]
            self.app_SED = [self.app_SED[0]*self.wave_units, self.app_SED[1]*self.flux_units, self.app_SED[2]*self.flux_units]
        except IOError:
            self.app_SED = ''
            
        # =====================================================================
        # Calculate synthetic photometry
        # =====================================================================
        
        # Set up empty synthetic photometry table
        self.syn_photometry = None
        
        # Find synthetic mags
        self.get_syn_photometry()
            
        # =====================================================================
        # Flux calibrate everything
        # =====================================================================
        
        # Calibrate using self.distance, self.distance_unc
        self.abs_SED = u.flux_calibrate(self.app_SED[1], self.distance, self.app_SED[2], self.distance_unc)
        self.abs_phot_SED = u.flux_calibrate(self.app_phot_SED[1], self.distance, self.app_phot_SED[2], self.distance_unc)
        self.abs_spec_SED = u.flux_calibrate(self.app_spec_SED[1], self.distance, self.app_spec_SED[2], self.distance_unc)
        
        # =====================================================================
        # Calculate Fundamental Params
        # =====================================================================
        self.fundamental_params(**kwargs)
        
        # =====================================================================
        # Save the data to file for cmd.py to read
        # =====================================================================
        # TODO
        
        print('\n'+'='*100)
        
    def process_radius(self, radius='', **kwargs):
        """
        Process the radius
        
        Parameters
        ==========
        radius: sequence (optional)
            The radius and uncertainty of the target
        """
        # Input radius
        if isinstance(radius, tuple):
            
            # Make sure it is a time unit
            try:
                _, _ = radius[0].to(q.m), radius[1].to(q.m)
                self.radius, self.radius_unc = radius
            except:
                print('Radius {} is not in units of length.'.format(radius))
                
        # Jupiter radius
        else:
            self.radius, self.radius_unc = 1.*ac.R_jup, ac.R_jup/100.
    
    def process_age(self, age='', membership='', **kwargs):
        """
        Process tha age
        
        Parameters
        ==========
        age: sequence (optional)
            The age minimum and maximum of the target
        membership: str (optional)
            The name of the parent NYMG
        """
        # Input age
        if isinstance(age, tuple):
            
            # Make sure it is a time unit
            try:
                _, _ = age[0].to(q.Myr), age[1].to(q.Myr)
                self.age_min, self.age_max = age
            except:
                print('Age {} is not in units of time.'.format(age))
                
        # NYMG age
        elif membership in NYMG:
            self.age_min, self.age_max = (NYMG[membership]['age_min'], NYMG[membership]['age_min'])*q.Myr
            
        # Low-g age
        elif self.gravity:
            self.age_min, self.age_max = (0.01, 0.15)*q.Gyr
            
        # Field age
        else:
            self.age_min, self.age_max = (0.5, 10)*q.Gyr
        
    def process_spectra(self, SNR=[], SNR_trim=5, trim=[], **kwargs):
        """
        Process the spectra
        
        Parameters
        ==========
        SNR: sequence (optional)
            A sequence of (spectrum_id, signal-to-noise) sequences to override spectrum SNR
        SNR_trim: float (optional)
            The SNR value to trim spectra edges up to
        trim: sequence (optional)
            A sequence of (spectrum_id, wave_min, wave_max) sequences to override spectrum trimming
        """
        # Index and add units
        fill = np.zeros(len(self.spectra))
        self.spectra.add_index('id')
        
        # Prepare apparent spectra
        self.processed_spectra = []
        for n,row in enumerate(self.spectra):
            
            # Unpack the spectrum
            w, f = row['spectrum'].data[:2]
            try:
                e = row['spectrum'].data[2]
            except IndexError:
                e = ''
            
            # Convert log units to linear
            if row['flux_units'].startswith('log '):
                f = 10**f, 
                try:
                    e = 10**e
                except:
                    pass
                row['flux_units'] = row['flux_units'].replace('log ', '')
            if row['wavelength_units'].startswith('log '):
                w = 10**w
                row['wavelength_units'] = row['wavelength_units'].replace('log ', '')
                
            # Make sure the wavelength units are right
            w = w*u.str2Q(row['wavelength_units']).to(self.wave_units).value
            
            # Convert F_nu to F_lam if necessary
            if row['flux_units']=='Jy':
                f = u.fnu2flam(f*q.Jy, w*self.wave_units, units=self.flux_units).value
                try:
                    e = u.fnu2flam(e*q.Jy, w*self.wave_units, units=self.flux_units).value
                except:
                    pass
                    
            # Force uncertainty array if none
            if not any(e) or e=='':
                e = f/10.
                print('No uncertainty array for spectrum {}. Using SNR=10.'.format(row['id']))
                
            # Insert uncertainty array of set SNR to force plotting
            for snr in SNR:
                if snr[0]==row['id']:
                    e = f/(1.*snr[1])
                    
            # Trim spectra frist up to first point with SNR>SNR_trim then manually
            if isinstance(SNR_trim, (float, int)):
                snr_trim = SNR_trim
            elif SNR_trim and any([i[0]==row['id'] for i in SNR_trim]):
                snr_trim = [i[1] for i in SNR_trim if i[0]==row['id']][0]
            else:
                snr_trim = 10
                
            if not SNR or not any([i[0]==row['id'] for i in SNR]):
                keep, = np.where(f/e>=snr_trim)
                if any(keep):
                    w, f, e = [i[np.nanmin(keep):np.nanmax(keep)+1] for i in [w, f, e]]
            if trim and any([i[0]==row['id'] for i in trim]):
                w, f, e = u.trim_spectrum([w, f, e], [i[1:] for i in trim if i[0]==row['id']])
                
            self.processed_spectra.append([w,f,e])
            
        # Print
        print('\nSPECTRA')
        self.spectra[['id','instrument_id','telescope_id','mode_id','publication_shortname']].pprint()
        
    def process_photometry(self, aliases='', **kwargs):
        """
        Process the photometry
        """
        # Index and add units
        fill = np.zeros(len(self.photometry))
        
        # Fill in empty columns
        for col in ['magnitude','magnitude_unc']:
            self.photometry[col][self.photometry[col]==None] = np.nan
            
        # Rename bands. What a pain in the ass.
        if isinstance(aliases,dict):
            self.photometry['band'] = [aliases.get(i) for i in self.photometry['band']]
        elif aliases=='guess':
            self.photometry['band'] = [min(list(FILTERS['Band']), key=lambda v: len(set(b)^set(v))) for b in self.photometry['band']]
        else:
            pass
        
        self.photometry.add_index('band')
        self.photometry.rename_column('magnitude','app_magnitude')
        self.photometry.rename_column('magnitude_unc','app_magnitude_unc')
        self.photometry['app_magnitude'].unit = q.mag
        self.photometry['app_magnitude_unc'].unit = q.mag
        
        # Add effective wavelengths to the photometry table
        self.photometry.add_column(at.Column(fill, 'eff', unit=self.wave_units))
        for row in self.photometry:
            try:
                band = FILTERS.loc[row['band']]
                row['eff'] = band['WavelengthEff']*q.Unit(band['WavelengthUnit'])
            except:
                row['eff'] = np.nan
            
        # Add absolute magnitude columns to the photometry table
        self.photometry.add_column(at.Column(fill, 'abs_magnitude', unit=q.mag))
        self.photometry.add_column(at.Column(fill, 'abs_magnitude_unc', unit=q.mag))
        
        # Calculate absolute mags and add to the photometry table
        if self.distance:
            for row in self.photometry:
                M, M_unc = u.flux_calibrate(row['app_magnitude'], self.distance, row['app_magnitude_unc'], self.distance_unc)
                row['abs_magnitude'] = M
                row['abs_magnitude_unc'] = M_unc
            
        # Add flux density columns to the photometry table
        for colname in ['app_flux','app_flux_unc','abs_flux','abs_flux_unc']:
            self.photometry.add_column(at.Column(fill, colname, unit=self.flux_units))
            
        # Calculate fluxes and add to the photometry table
        for i in ['app_','abs_']:
            for row in self.photometry:
                ph_flux = u.mag2flux(row['band'], row[i+'magnitude'], sig_m=row[i+'magnitude_unc'])
                row[i+'flux'] = ph_flux[0]
                row[i+'flux_unc'] = ph_flux[1]
                
        # Print
        print('\nPHOTOMETRY')
        self.photometry[['id','band','eff','app_magnitude','app_magnitude_unc','publication_shortname']].pprint()
        
    def process_parallaxes(self, pi='', dist='', **kwargs):
        """
        Process the parallax data
        """
        # Index and add units
        fill = np.zeros(len(self.parallaxes))
        
        # Add distance columns to the parallaxes table
        self.parallaxes.add_column(at.Column(fill, 'distance'))
        self.parallaxes.add_column(at.Column(fill, 'distance_unc'))
        
        # Add units
        self.parallaxes.add_row(np.zeros(len(self.parallaxes.colnames)))
        self.parallaxes['parallax'].unit = q.mas
        self.parallaxes['parallax_unc'].unit = q.mas
        self.parallaxes['distance'].unit = q.pc
        self.parallaxes['distance_unc'].unit = q.pc
        self.parallaxes = self.parallaxes[:-1]
        
        # Check for input parallax or distance
        if pi or dist:
            self.parallaxes['adopted'] = fill
            if pi:
                self.parallaxes.add_row({'parallax':pi[0], 'parallax_unc':pi[1], 'adopted':1, 'publication_shortname':'Input'})
            elif dist:
                self.parallaxes.add_row({'distance':dist[0], 'distance_unc':dist[1], 'adopted':1, 'publication_shortname':'Input'})
                    
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
                
        # Set adopted distance
        if len(self.parallaxes)>0 and not any(self.parallaxes['adopted']==1):
            self.parallaxes['adopted'][0] = 1
            
        # Sort by adopted distance
        self.parallaxes.add_index('adopted')
        
        # Get the adopted distance
        try:
            self.distance = self.parallaxes.loc[1]['distance']
            self.distance_unc = self.parallaxes.loc[1]['distance_unc']
        except KeyError:
            self.distance = self.distance_unc = ''
            
        # Print
        print('\nPARALLAXES')
        self.parallaxes[['id','distance','distance_unc','publication_shortname']].pprint()
        
    def process_spectral_types(self, spt='', **kwargs):
        """
        Process the spectral type data
        """
        # Sort by adopted spectral types
        fill = np.zeros(len(self.spectral_types))
        
        # Check for input parallax or distance
        if spt:
            self.spectral_types['adopted'] = fill
            sp, sp_unc, sp_pre, sp_grv, sp_lc = u.specType(spt)
            self.spectral_types.add_row({'spectral_type':sp, 'spectral_type_unc':sp_unc, 'gravity':sp_grv, 'suffix':sp_pre, 'adopted':1, 'publication_shortname':'Input'})
            
        # Set adopted spectral type
        if len(self.spectral_types)>0 and not any(self.spectral_types['adopted']==1):
            self.spectral_types['adopted'][0] = 1
            
        # Sort by adopted spectral type
        self.spectral_types.add_index('adopted')
        
        # Get the adopted spectral type
        try:
            self.spectral_type = self.spectral_types.loc[1]['spectral_type']
            self.spectral_type_unc = self.spectral_types.loc[1]['spectral_type_unc']
            self.gravity = self.spectral_types.loc[1]['gravity']
            self.suffix = self.spectral_types.loc[1]['suffix']
            self.SpT = u.specType([self.spectral_type, self.spectral_type_unc, self.suffix, self.gravity, ''])
            
        except:
            self.spectral_type = self.spectral_type_unc = self.gravity = self.suffix = self.SpT = ''
            
        # Print
        print('\nSPECTRAL TYPES')
        self.spectral_types[['id','spectral_type','spectral_type_unc','regime','suffix','gravity','publication_shortname']].pprint()
        
    def fundamental_params(self, **kwargs):
        """
        Calculate the fundamental parameters of the current SED
        """
        self.get_Lbol()
        self.get_Mbol()
        self.get_Teff()
        
        params = ['-','Lbol','Mbol','Teff']
        teff = self.Teff.value if hasattr(self.Teff, 'unit') else self.Teff
        teff_unc = self.Teff_unc.value if hasattr(self.Teff_unc, 'unit') else self.Teff_unc
        ptable = at.QTable(np.array([['Value',self.Lbol_sun,self.Mbol,teff],['Error',self.Lbol_sun_unc,self.Mbol_unc,teff_unc]]), names=params)
        print('\nRESULTS')
        ptable.pprint()
    
    def get_mbol(self, L_sun=3.86E26*q.W, Mbol_sun=4.74):
        """
        Calculate the apparent bolometric magnitude of the SED
        
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
        try:
            self.mbol = round(-2.5*np.log10(self.fbol.value)-11.482, 3)
            
            # Calculate mbol_unc
            try:
                self.mbol_unc = round((2.5/np.log(10))*(self.fbol_unc/self.fbol).value, 3)
            except:
                self.mbol_unc = ''
                
        # No dice
        except:
            self.mbol = self.mbol_unc = ''
        
        
    def get_Mbol(self):
        """
        Calculate the absolute bolometric magnitude of the SED
        """
        # Calculate mbol if not present
        if not hasattr(self, 'mbol'):
            self.get_mbol()
           
        # Calculate Mbol
        try:
            self.Mbol = round(self.mbol-5*np.log10((self.distance/10*q.pc).value), 3)
            
            # Calculate Mbol_unc
            try:
                self.Mbol_unc = round(np.sqrt(self.mbol_unc**2+((2.5/np.log(10))*(self.distance_unc/self.distance).value)**2), 3)
            except:
                self.Mbol_unc = ''
                
        # No dice
        except:
            self.Mbol = self.Mbol_unc = ''
        
    def get_fbol(self, units='erg/s/cm2'):
        """
        Calculate the bolometric flux of the SED
        """
        # Calculate fbol
        try:
            # Scrub negatives and NaNs
            app_sed = u.scrub(self.app_SED)
            
            self.fbol = np.trapz(app_sed[1], x=app_sed[0]).to(units)
            
            # Calculate fbol_unc
            try:
                self.fbol_unc = (np.sqrt(np.nansum(self.app_SED[2].value*np.gradient(self.app_SED[0].value))**2)*self.flux_units*self.wave_units).to(units)
            except:
                self.fbol_unc = ''
                
        # No dice
        except:
            self.fbol = self.fbol_unc = ''
        
    def get_Lbol(self):
        """
        Calculate the bolometric luminosity of the SED
        """
        # Caluclate fbol if not present
        if not hasattr(self, 'fbol'):
            self.get_fbol()
            
        # Calculate Lbol
        try:
            self.Lbol = (4*np.pi*self.fbol*self.distance**2).to(q.erg/q.s)
            self.Lbol_sun = round(np.log10((self.Lbol/ac.L_sun).decompose().value), 3)
            
            # Calculate Lbol_unc
            try:
                self.Lbol_unc = self.Lbol*np.sqrt((self.fbol_unc/self.fbol).value**2+(2*self.distance_unc/self.distance).value**2)
                self.Lbol_sun_unc = round(abs(self.Lbol_unc/(self.Lbol*np.log(10))).value, 3)
            except IOError:
                self.Lbol_unc = self.Lbol_sun_unc = ''
                
        # No dice
        except:
            self.Lbol = self.Lbol_sun = self.Lbol_unc = self.Lbol_sun_unc =''
                
    def get_Teff(self):
        """
        Calculate the effective temperature
        """
        # Calculate Teff
        try:
            self.Teff = np.sqrt(np.sqrt((self.Lbol/(4*np.pi*ac.sigma_sb*self.radius**2)).to(q.K**4))).round(0)
            
            # Calculate Teff_unc
            try:
                self.Teff_unc = (self.Teff*np.sqrt((self.Lbol_unc/self.Lbol).value**2 + (2*self.radius_unc/self.radius).value**2)/4.).round(0)
            except:
                self.Teff_unc = ''
                
        # No dice
        except:
            self.Teff = self.Teff_unc = ''
    
    def get_syn_photometry(self, bands=[], plot=False):
        """
        Calculate the synthetic magnitudes
        
        Parameters
        ----------
        bands: sequence
            The list of bands to calculate
        plot: bool
            Plot the synthetic mags
        """
        try:
            if not any(bands):
                bands = FILTERS['Band']
            
            # Only get mags in regions with spectral coverage
            syn_mags = []
            for spec in [i.as_void() for i in self.piecewise]:
                spec = [Q*(i.value if hasattr(i,'unit') else i) for i,Q in zip(spec,[self.wave_units,self.flux_units,self.flux_units])]
                syn_mags.append(s.all_mags(spec, bands=bands, plot=plot))
            
            # Stack the tables
            self.syn_photometry = at.vstack(syn_mags)
        
        except:
            print('No spectral coverage to calculate synthetic photometry.')
    
    def fit_blackbody(self, fit_to='app_phot_SED', epsilon=0.1, acc=5):
        """
        Fit a blackbody curve to the data
        
        Parameters
        ==========
        fit_to: str
            The attribute name of the [W,F,E] to fit
        epsilon: float
            The step size
        acc: float
            The acceptible error
        """
        # Get the data
        data = getattr(self, fit_to)
        
        # Remove NaNs
        print(data)
        data = np.array([(x,y,z) for x,y,z in zip(*data) if not any([np.isnan(i) for i in [x,y,z]]) and x<10]).T
        print(data)
        # Initial guess
        try:
            teff = self.Teff.value
        except:
            teff = 3000
        init = blackbody(temperature=teff)
        
        # Fit the blackbody
        fit = fitting.LevMarLSQFitter()
        bb = fit(init, data[0], data[1]/np.nanmax(data[1]), epsilon=epsilon, acc=acc)
        
        # Store the results
        try:
            self.Teff_bb = int(bb.temperature.value)
            self.bb_source = fit_to
            self.blackbody = bb
            print('\nBlackbody fit: {} K'.format(self.Teff_bb))
        except:
            print('\nNo blackbody fit.')
        
    
    def plot(self, app=True, photometry=True, spectra=True, integrals=False, syn_photometry=True, blackbody=True, scale=['log','log'], bokeh=True, output=False, **kwargs):
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
            mn_xs, mx_xs = np.nanmin(spec_SED[0].value), np.nanmax(spec_SED[0].value)
            mn_ys, mx_ys = np.nanmin(spec_SED[1].value[spec_SED[1].value>0]), np.nanmax(spec_SED[1].value[spec_SED[1].value>0])
        except:
            mn_xs, mx_xs, mn_ys, mx_ys = 0.3, 18, 999, -999
            
        mn_x, mx_x, mn_y, mx_y = np.nanmin([mn_xp,mn_xs]), np.nanmax([mx_xp,mx_xs]), np.nanmin([mn_yp,mn_ys]), np.nanmax([mx_yp,mx_ys])
        
        if not bokeh:
            
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Make the figure
            plt.figure(**kwargs)
            plt.title(self.name)
            plt.xlabel('Wavelength [{}]'.format(str(self.wave_units)))
            plt.ylabel('Flux [{}]'.format(str(self.flux_units)))
            
            # Plot spectra
            if spectra:
                spec_SED = self.app_spec_SED if app else self.abs_spec_SED
                plt.step(spec_SED[0], spec_SED[1], label='Spectra', **kwargs)
            
            # Plot photometry
            if photometry:
                phot_SED = self.app_phot_SED if app else self.abs_phot_SED
                plt.errorbar(phot_SED[0], phot_SED[1], yerr=phot_SED[2], marker='o', ls='None', label='Photometry', **kwargs)
            
            # Plot synthetic photometry
            if syn_photometry and self.syn_photometry:
                plt.errorbar(self.syn_photometry['eff'], self.syn_photometry['app_flux'], yerr=self.syn_photometry['app_flux_unc'], marker='o', ls='none', label='Synthetic Photometry')
            
            # Plot the SED with linear interpolation completion
            if integrals:
                full_SED = self.app_SED if app else self.abs_SED
                plt.plot(full_SED[0].value, full_SED[1].value, color='k', alpha=0.5, ls='--', label='Integral Surface')
                plt.fill_between(full_SED[0].value, full_SED[1].value-full_SED[2].value, full_SED[1].value+full_SED[2].value, color='k', alpha=0.1)
            
            # Set the x andx  y scales
            plt.xscale(scale[0], nonposx='clip')
            plt.yscale(scale[1], nonposy='clip')
            
        else:
            
            # TOOLS = 'crosshair,resize,reset,hover,box,save'
            fig = figure(plot_width=1000, plot_height=600, title=self.name, y_axis_type=scale[1], x_axis_type=scale[0], x_axis_label='Wavelength [{}]'.format(self.wave_units), y_axis_label='Flux Density [{}]'.format(str(self.flux_units)))
            
            # Plot spectra
            if spectra:
                spec_SED = getattr(self, pre+'spec_SED')
                source = ColumnDataSource(data=dict(x=spec_SED[0], y=spec_SED[1], z=spec_SED[2]))
                hover = HoverTool(tooltips=[( 'wave', '$x'),( 'flux', '$y'),('unc','$z')], mode='vline')
                fig.add_tools(hover)
                fig.line('x', 'y', source=source, legend='Spectra')
                
            # Plot photometry
            if photometry:
                
                # Plot points with errors
                pts = np.array([(x,y,z) for x,y,z in np.array(self.photometry['eff',pre+'flux',pre+'flux_unc']) if not any([np.isnan(i) for i in [x,y,z]])]).T
                try:
                    errorbar(fig, pts[0], pts[1], yerr=pts[2], point_kwargs={'fill_alpha':0.7, 'size':8}, legend='Photometry')
                except:
                    pass
                    
                # Plot saturated photometry
                pts = np.array([(x,y,z) for x,y,z in np.array(self.photometry['eff','app_flux','app_flux_unc']) if np.isnan(z) and not np.isnan(y)]).T
                try:
                    errorbar(fig, pts[0], pts[1], point_kwargs={'fill_alpha':0, 'size':8}, legend='Nondetection')
                except:
                    pass
                    
            # Plot synthetic photometry
            if syn_photometry and self.syn_photometry:
                
                # Plot points with errors
                pts = np.array([(x,y,z) for x,y,z in np.array(self.syn_photometry['eff',pre+'flux',pre+'flux_unc']) if not np.isnan(z)]).T
                try:
                    errorbar(fig, pts[0], pts[1], yerr=pts[2], point_kwargs={'fill_color':'red', 'fill_alpha':0.7, 'size':8}, legend='Synthetic Photometry')
                except:
                    pass
            
            # Plot the SED with linear interpolation completion
            if integrals:
                full_SED = getattr(self, pre+'SED')
                fig.line(full_SED[0].value, full_SED[1].value, line_color='black', alpha=0.3, legend='Integral Surface')
                # plt.fill_between(full_SED[0].value, full_SED[1].value-full_SED[2].value, full_SED[1].value+full_SED[2].value, color='k', alpha=0.1)
                
            if blackbody and self.blackbody:
                fit_sed = getattr(self, self.bb_source)
                fit_sed = [i[fit_sed[0]<10] for i in fit_sed]
                bb_wav = np.linspace(np.nanmin(fit_sed[0]), np.nanmax(fit_sed[0]), 500)*q.um
                bb_flx, bb_unc = u.blackbody(bb_wav, self.Teff_bb*q.K, 100*q.K)
                bb_norm = np.trapz(fit_sed[1], x=fit_sed[0])/np.trapz(bb_flx.value, x=bb_wav.value)
                bb_wav = np.linspace(0.2, 30, 1000)*q.um
                bb_flx, bb_unc = u.blackbody(bb_wav, self.Teff_bb*q.K, 100*q.K)
                print(bb_norm,bb_flx)
                fig.line(bb_wav.value, bb_flx.value*bb_norm, line_color='red', legend='{} K'.format(self.Teff_bb))
                
            fig.legend.location = "top_right"
            fig.legend.click_policy = "hide"
            fig.x_range = Range1d(mn_x*0.8, mx_x*1.2)
            fig.y_range = Range1d(mn_y*0.5, mx_y*2)
                
            if not output:
                show(fig)
            
            return fig
            
    def write(self, dirpath, app=False, spec=True, phot=False):
        """
        Exports a file of photometry and a file of the composite spectra with minimal data headers

        Parameters
        ----------
        dirpath: str
          The directory path to place the file
        app: bool
          Write apparent SED data
        spec: bool
          Write a file for the spectra with wavelength, flux and uncertainty columns
        phot: bool
          Write a file for the photometry with
        """
        if spec:
            try:
                spec_data = self.app_spec_SED if app else self.abs_spec_SED
                if dirpath.endswith('.txt'):
                    specpath = dirpath
                else:
                    specpath = dirpath + '{} SED.txt'.format(self.name)
                    
                header = '{} {} spectrum (erg/s/cm2/A) as a function of wavelength (um)'.format(self.name, 'apparent' if app else 'flux calibrated')
                
                np.savetxt(specpath, np.asarray(spec_data).T, header=header)
                
            except IOError:
                print("Couldn't print spectra.")
                
        if phot:
            try:
                phot = self.photometry
                
                if dirpath.endswith('.txt'):
                    photpath = dirpath
                else:
                    photpath = dirpath + '{} phot.txt'.format(self.name)
                    
                phot.write(photpath, format='ipac')
                
            except IOError:
                print("Couldn't print photometry.")
        
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

NYMG = {'TW Hya': {'age_min': 8, 'age_max': 20, 'age_ref': 0},
         'beta Pic': {'age_min': 12, 'age_max': 22, 'age_ref': 0},
         'Tuc-Hor': {'age_min': 10, 'age_max': 40, 'age_ref': 0},
         'Columba': {'age_min': 10, 'age_max': 40, 'age_ref': 0},
         'Carina': {'age_min': 10, 'age_max': 40, 'age_ref': 0},
         'Argus': {'age_min': 30, 'age_max': 50, 'age_ref': 0},
         'AB Dor': {'age_min': 50, 'age_max': 120, 'age_ref': 0},
         'Pleiades': {'age_min': 110, 'age_max': 130, 'age_ref': 0}}