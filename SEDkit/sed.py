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
from . import utilities as u
from . import syn_phot as s
from svo_filters import svo
from bokeh.models import HoverTool, Label, Range1d, BoxZoomTool
from bokeh.plotting import figure, output_file, show, save

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
    def __init__(self, source_id, db, from_dict='', pi='', dist='', pop=[], SNR=[], SNR_trim=5, SED_trim=[], split=[], trim=[], \
        age='', radius='', membership='', spt='', flux_units=q.erg/q.s/q.cm**2/q.AA, wave_units=q.um, name='', phot_aliases='guess'):
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
        if len(self.parallaxes)==0 and pi=='' and dist=='':
            print("\nNo distance for this source")
            
            self.distance = self.distance_unc = ''
            
        else:
            
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
                    self.parallaxes.add_row({'parallax':pi[0], 'parallax_unc':pi[1], \
                        'adopted':1, 'publication_shortname':'Input'})
                elif dist:
                    self.parallaxes.add_row({'distance':dist[0], 'distance_unc':dist[1],\
                        'adopted':1, 'publication_shortname':'Input'})
                        
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
        
        # =====================================================================
        # Spectral Type
        # =====================================================================
        
        # Punt if no SpT info
        if len(self.spectral_types)==0 and spt=='':
            print("\nNo spectral type for this source")
            
            self.spectral_type = self.spectral_type_unc = self.gravity = self.suffix = self.SpT = ''
            
        else:
            
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
        
        # =====================================================================
        # Age
        # =====================================================================
        
        # Retreive age data from input NYMG membership, input age range, or age estimate
        if isinstance(age, tuple):
            self.age_min, self.age_max = age
        elif membership in NYMG:
            self.age_min, self.age_max = (NYMG[membership]['age_min'], NYMG[membership]['age_min'])*q.Myr
        elif self.gravity:
            self.age_min, self.age_max = (0.01, 0.15)*q.Gyr
        else:
            self.age_min, self.age_max = (0.5, 10)*q.Gyr
            
        # =====================================================================
        # Radius
        # =====================================================================
        
        # Use radius if given
        self.radius, self.radius_unc = radius or [1.*ac.R_jup, ac.R_jup/100.]
        
        # =====================================================================
        # Photometry
        # =====================================================================
        
        # Index and add units
        fill = np.zeros(len(self.photometry))
        
        # Fill in empty columns
        for col in ['magnitude','magnitude_unc']:
            self.photometry[col][self.photometry[col]==None] = np.nan
            
        # Rename bands. What a pain in the ass.
        if isinstance(phot_aliases,dict):
            self.photometry['band'] = [phot_aliases.get(i) for i in self.photometry['band']]
        elif phot_aliases=='guess':
            self.photometry['band'] = [min(list(FILTERS['Band']), key=lambda v: len(set(b) ^ set(v))) for b in self.photometry['band']]
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
                
        # Make apparent photometric SED from photometry with uncertainties
        with_unc = self.photometry[(self.photometry['app_flux']>0)&(self.photometry['app_flux_unc']>0)]
        self.app_phot_SED = np.array([with_unc['eff'], with_unc['app_flux'], with_unc['app_flux_unc']])
        WP0, FP0, EP0 = [Q*i for Q,i in zip(units,self.app_phot_SED)]
        
        # Print
        print('\nPHOTOMETRY')
        self.photometry[['id','band','app_magnitude','app_magnitude_unc','publication_shortname']].pprint()
        
        # =====================================================================
        # Spectra
        # =====================================================================
        
        # Index and add units
        fill = np.zeros(len(self.spectra))
        self.spectra.add_index('id')
        
        # Prepare apparent spectra
        all_spectra = []
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
                
            all_spectra.append([w,f,e])
            
        # Print
        print('\nSPECTRA')
        self.spectra[['id','instrument_id','telescope_id','mode_id','publication_shortname']].pprint()
        
        # =====================================================================
        # Construct SED
        # =====================================================================
        
        # Group overlapping spectra and make composites where possible
        # to form peacewise spectrum for flux calibration
        if len(all_spectra) > 1:
            groups, piecewise = u.group_spectra(all_spectra), []
            for group in groups:
                composite = u.make_composite([[spec[0]*self.wave_units, spec[1]*self.flux_units, spec[2]*self.flux_units] for spec in group])
                piecewise.append(composite)
                
        # If only one spectrum, no need to make composite
        elif len(all_spectra) == 1:
            piecewise = np.copy(all_spectra)
            
        # If no spectra, forget it
        else:
            piecewise = []
            print('No spectra available for SED.')
            
        # Splitting
        keepers = []
        if split:
            for pw in piecewise:
                wavs = list(filter(None, [np.where(pw[0]<i)[0][-1] if pw[0][0]<i and pw[0][-1]>i else None for i in split]))
                keepers += map(list, zip(*[np.split(i, list(wavs)) for i in pw]))
                
            piecewise = np.copy(keepers)
            
        # Create Rayleigh Jeans Tail
        RJ_wav = np.arange(np.min([self.app_phot_SED[0][-1],12.]), 500, 0.1)*q.um
        RJ_flx, RJ_unc = u.blackbody(RJ_wav, 3000*q.K, 100*q.K)
        
        # Normalize Rayleigh-Jeans tail to the longest wavelength photometric point
        RJ_flx *= self.app_phot_SED[1][-1]/RJ_flx[0].value
        RJ = [RJ_wav, RJ_flx, RJ_unc]
        
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
            self.app_spec_SED = [np.array([])]*3
            W, F, E = W0, F0, E0 = [Q*np.array([]) for Q in units]
        
        # Exclude photometric points with spectrum coverage
        if self.piecewise:
            covered = []
            for n, i in enumerate(WP0):
                for N,spec in enumerate(self.piecewise):
                    if i<spec['wavelength'][-1] and i>spec['wavelength'][0]:
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
            self.app_SED = [np.concatenate(i) for i in [[ww[Wein[0] < min([min(i) for i in [WP, specPhot[0] or [999 * q.um]] if any(i)])], sp, bb[RJ[0] > max([max(i) for i in [WP, specPhot[0] or [-999 * q.um]] if any(i)])]] for ww, bb, sp in zip(Wein, RJ, specPhot)]]
        except IOError:
            self.app_SED = ''
            
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
        # TODO
        self.fundamental_params()
        
        # =====================================================================
        # Save the data to file for cmd.py to read
        # =====================================================================
        # TODO
        
        print('\n'+'='*100)
        
        
    def fundamental_params(self, age='', nymg='', evo_model='hybrid_solar_age', verbose=True):
        """
        Calculate the fundamental parameters of the current SED
        
        Parameters
        ----------
        age: tuple, list (optional)
            The lower and upper age limits of the source in astropy.units
        nymg: str (optional)
            The nearby young moving group name
        evo_model: str
            The evolutionary model to use
        """
        self.get_Lbol()
        self.get_Mbol()
        self.get_Teff()
        
        if verbose:
            
            params = ['-','Lbol','Mbol','Teff']
            ptable = at.QTable(np.array([['Value',self.Lbol_sun,self.Mbol,self.Teff.value],['Error',self.Lbol_sun_unc,self.Mbol_unc,self.Teff_unc.value]]), names=params)
            print('\nRESULTS')
            ptable.pprint()
    
    def get_mbol(self, L_sun=3.86E26*q.W, Mbol_sun=4.74):
        """
        Calculate the apparent bolometric magnitude of the SED
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
            self.fbol = (np.trapz(self.app_SED[1], x=self.app_SED[0])*self.flux_units*self.wave_units).to(units)
            
            # Calculate fbol_unc
            try:
                self.fbol_unc = (np.sqrt(np.sum(self.app_SED[2]*np.gradient(self.app_SED[0]))**2)*self.flux_units*self.wave_units).to(units)
            except:
                self.fbol_unc = ''
                
        # No dice
        except IOError:
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
            except:
                self.Lbol_unc = self.Lbol_sun_unc = ''
                
        # No dice
        except IOError:
            self.Lbol = self.Lbol_sun = self.Lbol_unc = self.Lbol_sun_unc =''
                
    def get_Teff(self):
        """
        Calculate the effective temperature of the SED
        
        Parameters
        ----------
        r: astropy.quantity
            The radius of the source in units of R_Jup
        sig_r: astropy.quantity
            The uncertainty in the radius
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
    
    def plot(self, photometry=True, spectra=True, integrals=True, app=True, scale=['log','log'], bokeh=True, **kwargs):
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
        bokeh: bool
            Plot in Bokeh
        """
        if not bokeh:
            
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Make the figure
            plt.figure(**kwargs)
            plt.title(self.name)
            plt.xlabel('Wavelength [{}]'.format(str(self.wave_units)))
            plt.ylabel('Flux [{}]'.format(str(self.flux_units)))
        
            # Distinguish between apparent and absolute magnitude
            pre = 'app_' if app else 'abs_'
        
            # Plot spectra
            if spectra:
                spec_SED = self.app_spec_SED if app else self.abs_spec_SED
                plt.step(spec_SED[0], spec_SED[1], **kwargs)
            
            # Plot photometry
            if photometry:
                phot_SED = self.app_phot_SED if app else self.abs_phot_SED
                plt.errorbar(phot_SED[0], phot_SED[1], yerr=phot_SED[2], marker='o', ls='None', **kwargs)
            
            # Plot the SED with linear interpolation completion
            if integrals:
                full_SED = self.app_SED if app else self.abs_SED
                plt.plot(full_SED[0].value, full_SED[1].value, color='k', alpha=0.5, ls='--')
                plt.fill_between(full_SED[0].value, full_SED[1].value-full_SED[2].value, full_SED[1].value+full_SED[2].value, color='k', alpha=0.1)
            
            # Set the x andx  y scales
            plt.xscale(scale[0], nonposx='clip')
            plt.yscale(scale[1], nonposy='clip')
            
        else:
            
            # TOOLS = 'crosshair,resize,reset,hover,box,save'
            fig = figure(plot_width=1000, plot_height=600, title=self.name, y_axis_type='log', x_axis_type='log', x_axis_label='Wavelength [{}]'.format(self.wave_units), y_axis_label='Flux Density [{}]'.format(str(self.flux_units)))
            
            # Plot spectra
            if spectra:
                spec_SED = self.app_spec_SED if app else self.abs_spec_SED
                fig.line(spec_SED[0], spec_SED[1], **kwargs)
                
            # Plot photometry
            if photometry:
                phot_SED = self.app_phot_SED if app else self.abs_phot_SED
                errorbar(fig, phot_SED[0], phot_SED[1], yerr=phot_SED[2])
                
            # Plot the SED with linear interpolation completion
            if integrals:
                full_SED = self.app_SED if app else self.abs_SED
                # fig.line(full_SED[0].value, full_SED[1].value, color='k', alpha=0.5, ls='--')
                # plt.fill_between(full_SED[0].value, full_SED[1].value-full_SED[2].value, full_SED[1].value+full_SED[2].value, color='k', alpha=0.1)
                
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
        
def errorbar(fig, x, y, xerr='', yerr='', color='black', point_kwargs={}, error_kwargs={}):
    """
    Hack to make errorbar plots in bokeh
    """
    fig.circle(x, y, color=color, **point_kwargs)

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
        
NYMG = {'TW Hya': {'age_min': 8, 'age_max': 20, 'age_ref': 0},
         'beta Pic': {'age_min': 12, 'age_max': 22, 'age_ref': 0},
         'Tuc-Hor': {'age_min': 10, 'age_max': 40, 'age_ref': 0},
         'Columba': {'age_min': 10, 'age_max': 40, 'age_ref': 0},
         'Carina': {'age_min': 10, 'age_max': 40, 'age_ref': 0},
         'Argus': {'age_min': 30, 'age_max': 50, 'age_ref': 0},
         'AB Dor': {'age_min': 50, 'age_max': 120, 'age_ref': 0},
         'Pleiades': {'age_min': 110, 'age_max': 130, 'age_ref': 0}}