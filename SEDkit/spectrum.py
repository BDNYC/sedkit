#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
Make nicer spectrum objects to pass around SED class
"""
import numpy as np
import astropy.units as q
import pysynphot as ps
import copy
import itertools
from . import synphot as syn
import astropy.constants as ac
from uncertainties import unumpy as unp
from bokeh.plotting import figure, output_file, show, save
from bokeh.palettes import Category10
from pkg_resources import resource_filename

def color_gen():
    """Color generator for Bokeh plots"""
    yield from itertools.cycle(Category10[10])
COLORS = color_gen()


class Spectrum(ps.ArraySpectrum):
    """A spectrum object to add uncertainty handling and spectrum stitching to ps.ArraySpectrum
    """
    def __init__(self, wave, flux, unc=None, snr=10, snr_trim=5, trim=[]):
        """Store the spectrum and units separately
        
        Parameters
        ----------
        wave: np.ndarray
            The wavelength array
        flux: np.ndarray
            The flux array
        unc: np.ndarray
            The uncertainty array
        snr: float (optional)
            A value to override spectrum SNR
        snr_trim: float (optional)
            The SNR value to trim spectra edges up to
        trim: sequence (optional)
            A sequence of (wave_min, wave_max) sequences to override spectrum trimming
        """
        # Make sure the arrays are the same shape
        if not wave.shape==flux.shape and ((unc is None) or not (unc.shape==flux.shape)):
            raise TypeError("Wavelength, flux and uncertainty arrays must be the same shape.")
            
        # Check wave units and convert to Angstroms if necessary to work with pysynphot
        try:
            # Store original units
            wave_units = wave.unit
            wave = wave.to(q.AA)
        except:
            raise TypeError("Wavelength array must be in astropy.units.quantity.Quantity length units, e.g. 'um'")
        
        # Check flux units
        try:
            _ = flux.to(q.erg/q.s/q.cm**2/q.AA)
        except:
            raise TypeError("Flux array must be in astropy.units.quantity.Quantity flux density units, e.g. 'erg/s/cm2/A'")
        
        # Check uncertainty units
        if unc is None:
            try:
                unc = flux/snr
                print("No uncertainty array for this spectrum. Using SNR=",snr)
            except:
                raise TypeError("Not a valid SNR: ",snr)
        
        # Make sure the uncertainty array is in the correct units
        try:
            _ = unc.to(q.erg/q.s/q.cm**2/q.AA)
        except:
            raise TypeError("Uncertainty array must be in astropy.units.quantity.Quantity flux density units, e.g. 'erg/s/cm2/A'")
            
        # Trim spectrum edges by SNR value
        if isinstance(snr_trim, (float, int)):
            idx, = np.where(flux/unc>=snr_trim)
            if any(idx):
                wave, flux, unc = [i[np.nanmin(idx):np.nanmax(idx)+1] for i in [wave, flux, unc]]
        
        # Trim manually
        if isinstance(trim, (list,tuple)):
            for mn,mx in trim:
                try:
                    idx, = np.where((wave<mn)|(wave>mx))
                    if any(idx):
                        wave, flux, unc = [i[idx] for i in [wave, flux, unc]]
                except TypeError:
                    print('Please provide a list of (lower,upper) bounds with units to trim, e.g. [(0*q.um,0.8*q.um)]')
            
        # Strip and store units
        self._wave_units = q.AA
        self._flux_units = flux.unit
        self.units = [self._wave_units, self._flux_units, self._flux_units]
        wave, flux, unc = [i.value for i in [wave,flux,unc]]
        
        # Make negatives and zeros into nans
        idx, = np.where(flux[flux>0])
        wave, flux, unc = [i[idx] for i in [wave, flux, unc]]
        
        # Inherit from ArraySpectrum
        super().__init__(wave, flux)
        
        # Add the uncertainty
        self._unctable = unc
        
        # Store components is added
        self.components = None
        
        # Convert back to input units
        self.wave_units = wave_units
        
        
    def __add__(self, spec2):
        """Add the spectra of this and another Spectrum object
        
        Parameters
        ----------
        spec2: SEDkit.spectrum.Spectrum
            The spectrum object to add
        
        Returns
        -------
        SEDkit.spectrum.Spectrum
            A new spectrum object with the input spectra stitched together
        """
        # If None is added, just return a copy
        if spec2 is None:
            return Spectrum(*self.spectrum)
            
        try:
            
            # Make spec2 the same units
            spec2.wave_units = self.wave_units
            spec2.flux_units = self.flux_units
            
            # Get the two spectra to stitch
            s1 = self.data
            s2 = spec2.data
            
            # Determine if overlapping
            overlap = True
            if s1[0][-1]>s1[0][0]>s2[0][-1] or s2[0][-1]>s2[0][0]>s1[0][-1]:
                overlap = False
                
            # Concatenate and order two segments if no overlap
            if not overlap:
            
                # Concatenate arrays and sort by wavelength
                spec = np.concatenate([s1,s2], axis=1).T
                spec = spec[np.argsort(spec[:, 0])].T
                
            # Otherwise there are three segments, (left, overlap, right)
            else:
            
                # Get the left segemnt
                left = s1[:, s1[0]<=s2[0][0]]
                if not np.any(left):
                    left = s2[:, s2[0]<=s1[0][0]]
                
                # Get the right segment
                right = s1[:, s1[0]>=s2[0][-1]]
                if not np.any(right):
                    right = s2[:, s2[0]>=s1[0][-1]]
                
                # Get the overlapping segements
                o1 = s1[:, np.where((s1[0]<right[0][0])&(s1[0]>left[0][-1]))].squeeze()
                o2 = s2[:, np.where((s2[0]<right[0][0])&(s2[0]>left[0][-1]))].squeeze()
                
                # Get the resolutions
                r1 = s1.shape[1]/(max(s1[0])-min(s1[0]))
                r2 = s2.shape[1]/(max(s2[0])-min(s2[0]))
                
                # Make higher resolution s1
                if r1<r2:
                    o1, o2 = o2, o1
                    
                # Interpolate s2 to s1
                o2_flux = np.interp(o1[0], s2[0], s2[1])
                o2_unc = np.interp(o1[0], s2[2], s2[2])
                
                # Get the average
                o_flux = np.nanmean([o1[1], o2_flux], axis=0)
                o_unc = np.sqrt(o1[2]**2 + o2_unc**2)
                overlap = np.array([o1[0], o_flux, o_unc])
                
                # Concatenate the segments
                spec = np.concatenate([left, overlap, right], axis=1)
                
            # Add units
            spec = [i*Q for i,Q in zip(spec, self.units)]
            
            # Make the new spectrum object
            new_spec = Spectrum(*spec)
            
            # Store the components
            new_spec.components = Spectrum(*self.spectrum), spec2
            
            return new_spec
            
        except IOError:
            raise TypeError('Only another SEDkit.spectrum.Spectrum object can be added. Input is of type {}'.format(type(spec2)))
            
            
    @property
    def data(self):
        """Store the spectrum without units
        """
        return np.array([self.wave, self.flux, self.unc])
        
        
    def flux_calibrate(self, distance, target_distance=10*q.pc, flux_units=None):
        """Flux calibrate the spectrum from the given distance to the target distance
        
        Parameters
        ----------
        distance: astropy.unit.quantity.Quantity, sequence
            The current distance or (distance, uncertainty) of the spectrum
        target_distance: astropy.unit.quantity.Quantity
            The distance to flux calibrate the spectrum to
        flux_units: astropy.unit.quantity.Quantity
            The desired flux units of the output
        """
        # Set target flux units
        if flux_units is None:
            flux_units = self.flux_units
            
        # Calculate the scaled flux
        flux = (self.spectrum[1]*(distance[0]/target_distance)**2).to(flux_units)
        
        # Calculate the scaled uncertainty
        term1 = (self.spectrum[2]*distance[0]/target_distance).to(flux_units)
        term2 = (2*self.spectrum[1]*(distance[1]*distance[0]/target_distance**2)).to(flux_units)
        unc = np.sqrt(term1**2 + term2**2)
        
        return Spectrum(self.spectrum[0], flux, unc)
        
            
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
        
        # Update the flux and unc arrays
        self._fluxtable = self.flux*self.flux_units.to(flux_units)
        self._unctable = self.unc*self.flux_units.to(flux_units)
            
        # Set the flux_units!
        self._flux_units = flux_units
        self.units = [self._wave_units, self._flux_units, self._flux_units]
        
        
    def integral(self, units=q.erg/q.s/q.cm**2):
        """Include uncertainties in integrate() method"""
        # Calculate the factor for the given units
        m = (self.flux_units*self.wave_units).to(units)
    
        # Calculate integral and uncertainty
        u_arr = unp.uarray(self.data[1:])
        value = np.trapz(u_arr, x=self.wave)
    
        # Apply the units
        val = float(unp.nominal_values(value)*m)*units
        unc = float(unp.std_devs(value)*m)*units
    
        return val, unc
    
    
    def norm_to_mags(self, photometry, exclude=[]):
        """
        Normalize the spectrum to the given bandpasses
    
        Parameters
        ----------
        photometry: astropy.table.QTable
            A table of the photometry
        exclude: sequence
            A list of bands to exclude from normalization
    
        Returns
        -------
        pysynphot.spectrum.ArraySpectralElement
            The normalized spectrum object
        """
        if len(photometry)==0:
            print('No photometry to normalize this spectrum.')
            norm = 1
            
        else:
            # Calculate all the synthetic magnitudes
            data = []
            for row in photometry:
                if row['band'] not in exclude:
                    try:
                        bp = row['bandpass']
                        syn_flx, syn_unc = self.synthetic_flux(bp)
                        weight = max(bp.wave)-min(bp.wave)
                        flx, unc = list(row['app_flux','app_flux_unc'])
                        data.append([flx.value, unc.value, syn_flx.value, syn_unc.value, weight])
                    except IOError:
                        pass
                
            # Check if there is overlap
            if len(data)==0:
                print('No photometry in the range {} to normalize this spectrum.'.format([self.min, self.max]))
                
            # Calculate the weighted normalization
            else:
                f1, e1, f2, e2, weights = np.array(data).T
                numerator = np.nansum(weight * f1 * f2 / (e1 ** 2 + e2 ** 2))
                denominator = np.nansum(weight * f2 ** 2 / (e1 ** 2 + e2 ** 2))
                norm = numerator/denominator
        
        return Spectrum(self.spectrum[0], self.spectrum[1]*norm, self.spectrum[2]*norm)
        
        
    @property
    def plot(self, fig=None, components=False, **kwargs):
        """Plot the spectrum"""
        # Make the figure
        if fig is None:
            fig = figure()
            fig.xaxis.axis_label = "Wavelength [{}]".format(self.wave_units)
            fig.yaxis.axis_label = "Flux Density [{}]".format(self.flux_units)
        
        # Plot the spectrum
        fig.line(self.wave, self.flux, color=next(COLORS))
        
        # Plot the components
        if self.components is not None:
            for spec in self.components:
                fig.line(spec.wave, spec.flux, color=next(COLORS))
            
        return fig
        
        
    def renormalize(self, mag, bandpass, system='vegamag', force=True, no_spec=False):
        """Include uncertainties in renorm() method
        
        Parameters
        ----------
        mag: float
            The target magnitude
        bandpass: SEDkit.synphot.Bandpass
            The bandpass to use
        system: str
            The magnitude system to use
        force: bool
            Force the synthetic photometry even if incomplete coverage
        no_spec: bool
            Return the normalization constant only
        
        Returns
        -------
        float, pysynphot.spectrum.ArraySpectralElement
            The normalization constant or normalized spectrum object
        """
        # Caluclate the remornalized flux
        spec = self.renorm(mag, system, bandpass, force)
    
        # Caluclate the normalization factor
        norm = np.mean(self.flux)/np.mean(spec.flux)
    
        # Just return the normalization factor
        if no_spec:
            return norm
    
        # Scale the spectrum
        data = [spec.wave, self.flux*norm, self.unc*norm]
    
        return Spectrum(*[i*Q for i,Q in zip(data, self.units)])
    
    
    @property
    def spectrum(self):
        """Store the spectrum with units
        """
        return [i*Q for i,Q in zip(self.data, self.units)]


    def synthetic_flux(self, bandpass, plot=False):
        """
        Calculate the magnitude in a bandpass
    
        Parameters
        ----------
        bandpass: pysynphot.spectrum.ArraySpectralElement
            The bandpass to use
        plot: bool
            Plot the spectrum and the flux point
    
        Returns
        -------
        float
            The magnitude
        """
        # Convert self to bandpass units
        self.wave_units = bandpass.wave_units
        
        # Caluclate the bits
        wav = bandpass.wave*bandpass.wave_units
        rsr = bandpass.throughput
        erg = (wav/(ac.h*ac.c)).to(1/q.erg)
        grad = np.gradient(wav).value
        
        # Interpolate the spectrum to the filter wavelengths
        f = np.interp(wav, self.wave, self.flux, left=0, right=0)*self.flux_units
        sig_f = np.interp(wav, self.wave, self.unc, left=0, right=0)*self.flux_units
        
        # Calculate the flux
        flx = (np.trapz(f*rsr, x=wav)/np.trapz(rsr, x=wav)).to(self.flux_units)
        unc = np.sqrt(np.sum(((sig_f*rsr*grad)**2).to(self.flux_units**2)))
        
        # Plot it
        if plot:
            fig = figure()
            fig.line(self.wave, self.flux, color='navy')
            fig.circle(bandpass.pivot(), flx, color='red')
            show(fig)
    
        return flx, unc
        
        
    def synthetic_magnitude(self, bandpass):
        """
        Calculate the synthetic magnitude in the given bandpass
        
        Parameters
        ----------
        bandpass: pysynphot.spectrum.ArraySpectralElement
            The bandpass to use
        
        Returns
        -------
        tuple
            The flux and uncertainty
        """
        flx = self.synthetic_flux(bandpass)
        
        # Calculate the magnitude
        mag = syn.flux2mag(flx, bandpass)
        
        return mag
        
        
    @property
    def unc(self):
        """A property for auncertainty"""
        return self._unctable
        
        
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
        
        # Update the wavelength array
        self.convert(str(wave_units))
        
        # Update min and max
        self.min = min(self.spectrum[0]).to(wave_units)
        self.max = max(self.spectrum[0]).to(wave_units)
            
        # Set the wave_units!
        self._wave_units = wave_units
        self.units = [self._wave_units, self._flux_units, self._flux_units]


class Vega(Spectrum):
    """A Spectrum object of Vega"""
    def __init__(self, wave_units=q.AA, flux_units=q.erg/q.s/q.cm**2/q.AA):
        """Initialize the Spectrum object
        
        Parameters
        ----------
        wave_units: astropy.units.quantity.Quantity
            The desired wavelength units
        flux_units: astropy.units.quantity.Quantity
            The desired flux units
        """
        # Get the data and apply units
        wave, flux = np.genfromtxt(resource_filename('SEDkit', 'data/STScI_Vega.txt'), unpack=True)
        wave *= q.AA
        flux *= q.erg/q.s/q.cm**2/q.AA
        
        # Convert to target units
        wave = wave.to(wave_units)
        flux = flux.to(flux_units)
        
        # Make the Spectrum object
        super().__init__(wave, flux)
        