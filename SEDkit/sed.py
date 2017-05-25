#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
SEDkit rewritten with astropy, pysynphot, and astrodbkit
"""
import numpy as np
import astropy.table as at
import astropy.units as q
import matplotlib.pyplot as plt
from . import utilities as u
from svo_filters import svo

FILTERS = svo.filters()
FILTERS['Band'] = [i.replace('.','_').replace('_I','_ch') for i in FILTERS['Band']]
FILTERS.add_index('Band')

class MakeSED(object):
    def __init__(self, source_id, db, spec_ids=[], pi='', dist='', pop=[], \
        flux_units=q.erg/q.s/q.cm**2/q.AA, wave_units=q.um, ):
        """
        Pulls all available data from the BDNYC Data Archive, 
        constructs an SED, and stores all calculations at *pickle_path*
        
        Parameters
        ----------
        source_id: int, str
            The *source_id*, *unum*, *shortname* or *designation* for any 
            source in the database.
        db: database instance
            The database instance to retreive data from
        spec_ids: list, tuple (optional)
            A sequence of the ids from the SPECTRA table to plot. 
            Uses any available spectra if no list is given. Uses no spectra 
            if 'None' is given
        
        pop: sequence (optional)
            Photometric bands to exclude from the SED
        
        """
        # TODO: resolve source_id in database given id, (ra,dec), name, etc.
        # source_id = db._resolve_source_id()
        
        # Get the data for the source from the database
        all_data = db.inventory(source_id, fetch=True)
        
        # Store the tables as attributes
        for key,table in all_data.items():
            setattr(self, key, at.QTable(table))
            
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
        
        # Check for input parallax or distance
        if pi:
            self.parallaxes['adopted'] = fill
            self.parallaxes.add_row({'parallax':pi[0], 'parallax_unc':pi[1], \
                'adopted':1, 'publication_shortname':'Input'})
        elif dist:
            self.parallaxes['adopted'] = fill
            self.parallaxes.add_row({'distance':dist[0], 'distance_unc':dist[1],\
                'adopted':1, 'publication_shortname':'Input'})
        else:
            pass
            
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
        # Photometry
        # =====================================================================
        
        # Index and add units
        fill = np.zeros(len(self.photometry))
        self.photometry.add_index('band')
        self.photometry.rename_column('magnitude','app_magnitude')
        self.photometry.rename_column('magnitude_unc','app_magnitude_unc')
        self.photometry['app_magnitude'].unit = q.mag
        self.photometry['app_magnitude_unc'].unit = q.mag
        
        # Pop unwanted mags
        self.photometry = self.photometry[[self.photometry.loc[band].index \
            for band in self.photometry['band'] if band not in pop]]
            
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
            self.photometry.add_column(at.Column(fill, colname, unit=flux_units))
            
        # Calculate fluxes and add to the photometry table
        for i in ['app_','abs_']:
            for row in self.photometry:
                ph_flux = u.mag2flux(row['band'], row[i+'magnitude'],\
                    sig_m=row[i+'magnitude_unc'])
                row[i+'flux'] = ph_flux[0]
                row[i+'flux_unc'] = ph_flux[1]
                
        # Make relative and absolute photometric SEDs
        # self.photometry.sort('eff')
        self.app_phot_SED = np.array([self.photometry['eff'], \
            self.photometry['app_flux'], self.photometry['app_flux_unc']])
        self.abs_phot_SED = np.array([self.photometry['eff'], \
            self.photometry['abs_flux'], self.photometry['abs_flux_unc']])
                
        # =====================================================================
        # Spectra
        # =====================================================================
        
        # Index and add units
        fill = np.zeros(len(self.spectra))
        self.spectra.add_index('id')
        self.photometry.rename_column('spectrum','app_spectrum')
        
        # Pop unwanted spectra
        if spec_ids:
            self.spectra = self.spectra[[self.spectra.loc[spec_id].index \
                for spec_id in self.spectra['id'] if spec_id in spec_ids]]
            
        # Combine all spectra into apparent SED
        
        #
        
    def plot(self, phot=True, spec=True, app=False, scale=['log','log'], **kwargs):
        """
        Plot the SED
        
        Parameters
        ----------
        phot: bool
            Plot the photometry
        spec: bool
            Plot the spectra
        app: bool
            Plot the apparent SED instead of absolute
        scale: array-like
            The (x,y) scales to plot, 'linear' or 'log'
        """
        # Distinguish between apparent and absolute magnitude
        pre = 'app_' if app else 'abs_'
        
        # Plot photometry
        if phot:
            phot_SED = self.app_phot_SED if app else self.abs_phot_SED
            plt.errorbar(phot_SED[0], phot_SED[1], yerr=phot_SED[2], \
                marker='o', ls='None', **kwargs)
                
        # # Plot spectra
        # if spec:
        #     spec_SED = self.app_spec_SED if app else self.abs_spec_SED
        #     plt.errorbar(spec_SED[0], spec_SED[1], yerr=spec_SED[2], \
        #         marker='o', ls='None', **kwargs)
        
        # Set the x and y scales
        plt.yscale(scale[0], nonposy='clip')
        plt.yscale(scale[1], nonposy='clip')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        