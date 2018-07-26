#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
Make nicer spectrum objects to pass around SED class
"""
import numpy as np
import astropy.units as q
import astropy.io.votable as vo
import pysynphot as ps
import copy
from . import synphot as syn
from . import utilities as u
import astropy.constants as ac
from astropy.io import fits
from bokeh.plotting import figure, output_file, show, save
from pkg_resources import resource_filename


COLORS = u.color_gen('Category10')


class Spectrum(ps.ArraySpectrum):
    """A spectrum object to add uncertainty handling and spectrum stitching to ps.ArraySpectrum
    """
    def __init__(self, wave, flux, unc=None, snr=10, snr_trim=5, trim=None, name=None, verbose=False):
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
        name: str
            A name for the spectrum
        verbose: bool
            Print helpful stuff
        """
        # Meta
        self.name = name
        self.verbose = verbose
        
        # Make sure the arrays are the same shape
        if not wave.shape == flux.shape and ((unc is None) or not (unc.shape==flux.shape)):
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
        
        # Remove nans, negatives, zeros, and infs
        wave, flux, unc = u.scrub([wave, flux, unc])
            
        # Strip and store units
        self._wave_units = q.AA
        self._flux_units = flux.unit
        self.units = [self._wave_units, self._flux_units, self._flux_units]
        wave, flux, unc = [i.value for i in [wave,flux,unc]]
        
        # Make negatives and zeros into nans
        idx, = np.where(flux[flux > 0])
        wave, flux, unc = [i[idx] for i in [wave, flux, unc]]
        
        # Inherit from ArraySpectrum
        super().__init__(wave, flux)
        
        # Add the uncertainty
        self._unctable = unc
        
        # Store components is added
        self.components = None
        
        # Convert back to input units
        self.wave_units = wave_units
        
        # Trim spectrum edges by SNR value
        if isinstance(snr_trim, (float, int)):
            idx, = np.where(flux/unc >= snr_trim)
            if any(idx):
                wave, flux, unc = [i[np.nanmin(idx):np.nanmax(idx)+1] for i in [wave, flux, unc]]
        
        # Trim manually
        if trim is not None:
            self.trim(trim)
        
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
            if s1[0][-1] > s1[0][0] > s2[0][-1] or s2[0][-1] > s2[0][0] > s1[0][-1]:
                overlap = False
            
            # Concatenate and order two segments if no overlap
            if not overlap:
            
                # Concatenate arrays and sort by wavelength
                spec = np.concatenate([s1,s2], axis=1).T
                spec = spec[np.argsort(spec[:, 0])].T
                
            # Otherwise there are three segments, (left, overlap, right)
            else:
            
                # Get the left segemnt
                left = s1[:, s1[0] <= s2[0][0]]
                if not np.any(left):
                    left = s2[:, s2[0] <= s1[0][0]]
                
                # Get the right segment
                right = s1[:, s1[0] >= s2[0][-1]]
                if not np.any(right):
                    right = s2[:, s2[0] >= s1[0][-1]]
                
                # Get the overlapping segements
                o1 = s1[:, np.where((s1[0] < right[0][0]) & (s1[0] > left[0][-1]))].squeeze()
                o2 = s2[:, np.where((s2[0] < right[0][0]) & (s2[0] > left[0][-1]))].squeeze()
                
                # Get the resolutions
                r1 = s1.shape[1]/(max(s1[0])-min(s1[0]))
                r2 = s2.shape[1]/(max(s2[0])-min(s2[0]))
                
                # Make higher resolution s1
                if r1 < r2:
                    o1, o2 = o2, o1
                    
                # Interpolate s2 to s1
                o2_flux = np.interp(o1[0], s2[0], s2[1])
                o2_unc = np.interp(o1[0], s2[2], s2[2])
                
                # Get the average
                o_flux = np.nanmean([o1[1], o2_flux], axis=0)
                o_unc = np.sqrt(o1[2]**2 + o2_unc**2)
                overlap = np.array([o1[0], o_flux, o_unc])
                
                # Make sure it is 2D
                if overlap.shape == (3,):
                    overlap.shape = 3, 1
                    
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
            
        # Make sure the values are in flux density units
        try:
            _ = flux_units.to(q.erg/q.s/q.cm**2/q.AA)
        except:
            raise TypeError("flux_units must be in flux density units, e.g. 'erg/s/cm2/A'")
        
        # Update the flux and unc arrays
        self._fluxtable = self.flux*self.flux_units.to(flux_units)
        self._unctable = self.unc*self.flux_units.to(flux_units)
            
        # Set the flux_units!
        self._flux_units = flux_units
        self.units = [self._wave_units, self._flux_units, self._flux_units]
        
    def integral(self, units=q.erg/q.s/q.cm**2):
        """Include uncertainties in integrate() method
        
        Parameters
        ----------
        units: astropy.units.quantity.Quantity
            The target units for the integral
        
        Returns
        -------
        sequence
            The integrated flux and uncertainty
        """
        # Make sure the target units are flux units
        try:
            _ = units.to(q.erg/q.s/q.cm**2)
        except:
            raise TypeError("units must be in flux units, e.g. 'erg/s/cm2'")
            
        # Calculate the factor for the given units
        m = self.flux_units*self.wave_units
        
        val = np.trapz(self.flux, x=self.wave)*m
        unc = np.sqrt(np.sum((self.unc*np.gradient(self.wave)*m)**2))
    
        return val.to(units), unc.to(units)
    
    def norm_to_mags(self, photometry, force=False, exclude=[], include=[]):
        """
        Normalize the spectrum to the given bandpasses

        Parameters
        ----------
        photometry: astropy.table.QTable
            A table of the photometry
        exclude: sequence (optional)
            A list of bands to exclude from the normalization
        include: sequence (optional)
            A list of bands to include in the normalization

        Returns
        -------
        pysynphot.spectrum.ArraySpectralElement
            The normalized spectrum object
        """
        # Default norm
        norm = 1

        # Compile list of photometry to include
        keep = []
        for band in photometry['band']:

            # Keep only explicitly included bands...
            if include:
                if band in include:
                    keep.append(band)

            # ...or just keep non excluded bands
            else:
                if band not in exclude:
                    keep.append(band)

        # Trim the table
        idx = np.sum([photometry['band'] == k for k in keep], axis=0)
        photometry = photometry[np.where(idx)]

        if len(photometry) == 0:
            if self.verbose:
                print('No photometry to normalize this spectrum.')

        else:
            # Calculate all the synthetic magnitudes
            data = []
            for row in photometry:

                try:
                    bp = row['bandpass']
                    syn_flx, syn_unc = self.synthetic_flux(bp, force=force)
                    if syn_flx is not None:
                        flx, unc = list(row['app_flux','app_flux_unc'])
                        weight = bp.FWHM
                        data.append([flx.value, unc.value, syn_flx.value, syn_unc.value, weight])
                except IOError:
                    pass

            # Check if there is overlap
            if len(data) == 0:
                if self.verbose:
                    print('No photometry in the range {} to normalize this spectrum.'.format([self.min, self.max]))

            # Calculate the weighted normalization
            else:
                f1, e1, f2, e2, weights = np.array(data).T
                numerator = np.nansum(weight * f1 * f2 / (e1 ** 2 + e2 ** 2))
                denominator = np.nansum(weight * f2 ** 2 / (e1 ** 2 + e2 ** 2))
                norm = numerator/denominator

        return Spectrum(self.spectrum[0], self.spectrum[1]*norm, self.spectrum[2]*norm)

    def norm_to_spec(self, spectrum, exclude=[], include=[]):
        """Normalize the spectrum to another spectrum
        
        Parameters
        ----------
        spectrum: SEDkit.spectrum.Spectrum
            The spectrum to normalize to
        exclude: sequence (optional)
            A list of wavelength ranges to exclude from the normalization
        include: sequence (optional)
            A list of wavelength ranges to include in the normalization

        Returns
        -------
        SEDkit.spectrum.Spectrum
          The normalized spectrum
        """
        spec = self.spectrum
        
        # Make into the same units
        spectrum.wave_units = self.wave_units
        spectrum.flux_units = self.flux_units
        temp = spectrum.spectrum
        
        # Find wavelength range of overlap for array masking
        spec_mask = np.logical_and(spec[0] > temp[0][0], spec[0] < temp[0][-1])
        temp_mask = np.logical_and(temp[0] > spec[0][0], temp[0] < spec[0][-1])
        spectrum = [i[spec_mask] for i in spec]
        temp = [i[temp_mask] for i in temp]

        # Also mask arrays in wavelength ranges specified in *exclude*
        for r in exclude:
            spec_mask = np.logical_and(spec[0] > r[0], spec[0] < r[-1])
            temp_mask = np.logical_and(temp[0] > r[0], temp[0] < r[-1])
            spec = [i[~spec_mask] for i in spec]
            temp = [i[~temp_mask] for i in temp]

        # Normalize the spectrum to the temp based on equal integrated flux inincluded wavelength ranges
        norm = np.trapz(temp[1], x=temp[0]) / np.trapz(spec[1], x=spec[0])

        return Spectrum(self.spectrum[0], self.spectrum[1]*norm, self.spectrum[2]*norm)

    def plot(self, fig=None, components=False, draw=False, **kwargs):
        """Plot the spectrum"""
        # Make the figure
        if fig is None:
            fig = figure()
            fig.xaxis.axis_label = "Wavelength [{}]".format(self.wave_units)
            fig.yaxis.axis_label = "Flux Density [{}]".format(self.flux_units)
        
        # Plot the spectrum
        c = kwargs.get('color', next(COLORS))
        fig.line(self.wave, self.flux, color=c)
        
        # Plot the uncertainties
        band_x = np.append(self.wave, self.wave[::-1])
        band_y = np.append(self.flux-self.unc, (self.flux+self.unc)[::-1])
        fig.patch(band_x, band_y, color=c, fill_alpha=0.1, line_alpha=0)
        
        # Plot the components
        if components and self.components is not None:
            for spec in self.components:
                fig.line(spec.wave, spec.flux, color=next(COLORS))
            
        if draw:
            show(fig)
        else:
            return fig
        
    def renormalize(self, mag, bandpass, system='vegamag', force=False, no_spec=False):
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
        # # Caluclate the remornalized flux
        # spec = self.renorm(mag, system, bandpass, force)
        #
        # # Caluclate the normalization factor
        # norm = np.mean(self.flux)/np.mean(spec.flux)
        
        # My solution
        norm = syn.mag2flux(mag, bandpass)[0]/self.synthetic_flux(bandpass, force=force)[0]
    
        # Just return the normalization factor
        if no_spec:
            return norm
            
        # Scale the spectrum
        data = [self.wave, self.flux*norm, self.unc*norm]
    
        return Spectrum(*[i*Q for i,Q in zip(data, self.units)])
        
        
    def resamp(self, wave=None, resolution=None):
        """Resample the spectrum onto a new wavelength array or to a new
        resolution
        
        Parameters
        ----------
        wave: astropy.units.quantity.Quantity (optional)
            The wavelength array to resample onto
        resolution: int (optional)
            The new resolution to resample to, keeping the same
            wavelength range
        """
        mn = np.nanmin(self.wave)
        mx = np.nanmax(self.wave)
        
        if resolution is not None:
            
            # Make the wavelength array
            d_lam = (mx-mn)/resolution
            wave = np.arange(mn, mx, d_lam)
            
        elif wave is not None:
            wave = wave.to(self.wave_units).value
            
        else:
            return
            
        # Trim the wavelength
        wave = wave[np.logical_and(wave > mn, wave <= mx)]

        # Calculate the new spectrum
        binned = u.spectres(wave, self.wave, self.flux, self.unc)
        
        # Update the spectrum
        spectrum = [i*Q for i, Q in zip(binned, self.units)]
        
        return Spectrum(*spectrum)

    def smooth(self, beta, window=11):
        """
        Smooths the spectrum using a Kaiser-Bessel smoothing window of
        narrowness *beta* (~1 => very smooth, ~100 => not smooth)
        
        Parameters
        ----------
        beta: float, int
            The narrowness of the window
        window: int
            The length of the window
        
        Returns
        -------
        SEDkit.spectrum.Spectrum
            The smoothed spectrum
        """
        s = np.r_[self.flux[window - 1:0:-1], self.flux, self.flux[-1:-window:-1]]
        w = np.kaiser(window, beta)
        y = np.convolve(w / w.sum(), s, mode='valid')
        smoothed = y[5:len(y) - 5]

        return Spectrum(self.spectrum[0], smoothed*self.flux_units, self.spectrum[2])

    @property
    def spectrum(self):
        """Store the spectrum with units
        """
        return [i*Q for i,Q in zip(self.data, self.units)]

    def synthetic_flux(self, bandpass, force=False, plot=False):
        """
        Calculate the magnitude in a bandpass
    
        Parameters
        ----------
        bandpass: pysynphot.spectrum.ArraySpectralElement
            The bandpass to use
        force: bool
            Force the magnitude calculation even if
            overlap is only partial
        plot: bool
            Plot the spectrum and the flux point
    
        Returns
        -------
        float
            The magnitude
        """
        # Test overlap
        overlap = bandpass.overlap(self.spectrum)
        
        # Initialize
        flx = unc = None
        
        if overlap=='full' or (overlap=='partial' and force):

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
                fig.circle([bandpass.eff], [flx], color='red')
                show(fig)
    
        return flx, unc
        
    def synthetic_magnitude(self, bandpass, force=False):
        """
        Calculate the synthetic magnitude in the given bandpass
        
        Parameters
        ----------
        bandpass: pysynphot.spectrum.ArraySpectralElement
            The bandpass to use
        force: bool
            Force the magnitude calculation even if
            overlap is only partial
        
        Returns
        -------
        tuple
            The flux and uncertainty
        """
        flx = self.synthetic_flux(bandpass, force=force)
        
        # Calculate the magnitude
        mag = syn.flux2mag(flx, bandpass)
        
        return mag
            
    def trim(self, ranges):
        """Trim the spectrum in the given wavelength ranges
        
        Parameters
        ----------
        ranges: sequence
            The (min_wave, max_wave) ranges to trim from the spectrum
        """
        if isinstance(ranges, (list,tuple)):
            for mn,mx in ranges:
                try:
                    idx, = np.where((self.spectrum[0] < mn) | 
                                    (self.spectrum[0] > mx))
                    
                    if len(idx) > 0:
                        wave, flux, unc = [i[idx] for i in self.spectrum]
                        
                        # Update the object
                        spec = Spectrum(wave, flux, unc)
                        self.__dict__ = spec.__dict__
                        del spec
                
                except TypeError:
                    print("""Please provide a list of (lower,upper) bounds\
                             with units to trim, e.g. [(0*q.um,0.8*q.um)]""")
        
        else:
            raise TypeError("""Please provide a list of (lower,upper) bounds\
                             with units to trim, e.g. [(0*q.um,0.8*q.um)]""")
        
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
        good_units = q.core.PrefixUnit, q.core.Unit, q.core.CompositeUnit
        if not isinstance(wave_units, good_units):
            raise TypeError('wave_units must be in astropy units')
            
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


class Blackbody(Spectrum):
    """A spectrum object specifically for blackbodies"""
    def __init__(self, wavelength, Teff, radius=None, distance=None, **kwargs):
        """
        Given a wavelength array and temperature, returns an array of Planck
        function values in [erg s-1 cm-2 A-1]
    
        Parameters
        ----------
        lam: array-like
            The array of wavelength values to evaluate the Planck function
        Teff: astropy.unit.quantity.Quantity
            The effective temperature
        Teff_unc: astropy.unit.quantity.Quantity
            The effective temperature uncertainty
        """
        try:
            wavelength.to(q.um)
        except:
            raise TypeError("Wavelength must be in astropy units, e.g. 'um'")
            
        # Store parameters
        if not isinstance(Teff, (q.quantity.Quantity, tuple, list)):
            if not isinstance(Teff[0], q.quantity.Quantity):
                raise TypeError("Teff must be in astropy units, eg. 'K'")

        if isinstance(Teff, (tuple, list)):
            self.Teff, self.Teff_unc = Teff
        else:
            self.Teff, self.Teff_unc = Teff, None

        if isinstance(radius, (tuple, list)):
            self.radius, self.radius_unc = radius
        else:
            self.radius, self.radius_unc = radius, None

        if isinstance(distance, (tuple, list)):
            self.distance, self.distance_unc, *_ = distance
        else:
            self.distance, self.distance_unc = distance, None

        # Evaluate
        I, I_unc = self.eval(wavelength)

        # Inherit from Spectrum
        super().__init__(wavelength, I, I_unc, **kwargs)

        self.name = '{} Blackbody'.format(Teff)

    def eval(self, wavelength, Flam=False):
        """Evaluate the blackbody at the given wavelengths

        Parameters
        ----------
        wavelength: sequence
            The wavelength values

        Returns
        -------
        list
            The blackbody flux and uncertainty arrays
        """
        try:
            wavelength.to(q.um)
        except:
            raise TypeError("Wavelength must be in astropy units, e.g. 'um'")

        units = q.erg/q.s/q.cm**2/(1 if Flam else q.AA)

        # Check for radius and distance
        if self.radius is not None and self.distance is not None:
            scale = (self.radius**2/self.distance**2).decompose()
        else:
            scale = 1.

        # Get numerator and denominator
        const = ac.h*ac.c/(wavelength*ac.k_B)
        numer = 2*np.pi*ac.h*ac.c**2*scale/(wavelength**(4 if Flam else 5))
        denom = np.exp((const/self.Teff)).decompose()

        # Calculate intensity
        I = (numer/(denom-1.)).to(units)

        # Calculate dI/dr
        if self.radius is not None and self.radius_unc is not None:
            dIdr = (self.radius_unc*2*I/self.radius).to(units)
        else:
            dIdr = 0.*units

        # Calculate dI/dd
        if self.distance is not None and self.distance_unc is not None:
            dIdd = (self.distance_unc*2*I/self.distance).to(units)
        else:
            dIdd = 0.*units

        # Calculate dI/dT
        if self.Teff is not None and self.Teff_unc is not None:
            dIdT = (self.Teff_unc*I*ac.h*ac.c/wavelength/ac.k_B/self.Teff**2).to(units)
        else:
            dIdT = 0.*units

        # Calculate sigma_I from derivative terms
        I_unc = np.sqrt(dIdT**2 + dIdr**2 + dIdd**2)
        if isinstance(I_unc.value, float):
            I_unc = None

        return I, I_unc


class FileSpectrum(Spectrum):
    def __init__(self, file, wave_units=None, flux_units=None, ext=0,
                 survey=None, **kwargs):
        """Create a spectrum from an ASCII or FITS file

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
            The name of the survey
        """
        # Read the fits data...
        if file.endswith('.fits'):

            raw = fits.getdata(file, ext=ext)

            if survey == 'SDSS':
                head = fits.getheader(file)
                flux_units = 1E-17*q.erg/q.s/q.cm**2/q.AA
                wave_units = q.AA
                log_w = head['COEFF0']+head['COEFF1']*np.arange(len(raw.flux))
                data = [10**log_w, raw.flux, raw.ivar]

            # Check if it is a recarray
            elif isinstance(raw, fits.fitsrec.FITS_rec):

                # Check if it's an SDSS spectrum
                raw = fits.getdata(file, ext=ext)
                data = raw['WAVELENGTH'], raw['FLUX'], raw['ERROR']

            # Otherwise just an array
            else:
                data = raw

        # ...or the ascii data...
        elif file.endswith('.txt'):
            data = np.genfromtxt(file, unpack=True)
            
        # ...or the VO Table
        elif file.endswith('.xml'):
            vot = vo.parse_single_table(file)
            data = np.array([list(i) for i in vot.array]).T

        else:
            raise IOError('The file needs to be ASCII, XML, or FITS.')

        # Apply units
        wave = data[0]*wave_units
        flux = data[1]*flux_units
        if len(data) > 2:
            unc = data[2]*flux_units
        else:
            unc = None
            
        super().__init__(wave, flux, unc, **kwargs)


class Vega(Spectrum):
    """A Spectrum object of Vega"""
    def __init__(self, wave_units=q.AA, flux_units=q.erg/q.s/q.cm**2/q.AA,
                 **kwargs):
        """Initialize the Spectrum object
        
        Parameters
        ----------
        wave_units: astropy.units.quantity.Quantity
            The desired wavelength units
        flux_units: astropy.units.quantity.Quantity
            The desired flux units
        """
        # Get the data and apply units
        vega_file = resource_filename('SEDkit', 'data/STScI_Vega.txt')
        wave, flux = np.genfromtxt(vega_file, unpack=True)
        wave *= q.AA
        flux *= q.erg/q.s/q.cm**2/q.AA

        # Convert to target units
        wave = wave.to(wave_units)
        flux = flux.to(flux_units)

        # Make the Spectrum object
        super().__init__(wave, flux, **kwargs)

        self.name = 'Vega'
        