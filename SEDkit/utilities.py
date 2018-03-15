#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
Some utilities to accompany SEDkit
"""
import re
import numpy as np
import astropy.units as q
import astropy.constants as ac
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from svo_filters import svo
from astropy.utils.data_info import ParentDtypeInfo

class ArrayWrapper(object):
    """
    Minimal mixin using a simple wrapper around a numpy array
    """
    info = ParentDtypeInfo()

    def __init__(self, data):
        self.data = np.array(data)
        self.unit = data.unit if hasattr(data, 'unit') else 1
        if 'info' in getattr(data, '__dict__', ()):
            self.info = data.info

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            out = self.data[item]
        else:
            out = self.__class__(self.data[item])
            if 'info' in self.__dict__:
                out.info = self.info
        return out*self.unit

    def __setitem__(self, item, value):
        self.data[item] = value

    def __len__(self):
        return len(self.data)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return ("<{0} name='{1}' data={2}>"
                .format(self.__class__.__name__, self.info.name, self.data))

def blackbody(lam, Teff, Teff_unc='', Flam=False, radius=1*ac.R_jup, dist=10*q.pc, plot=False):
    """
    Given a wavelength array and temperature, returns an array of Planck function values in [erg s-1 cm-2 A-1]
    
    Parameters
    ----------
    lam: array-like
        The array of wavelength values to evaluate the Planck function
    Teff: astropy.unit.quantity.Quantity
        The effective temperature
    Teff_unc: astropy.unit.quantity.Quantity
        The effective temperature uncertainty
    
    Returns
    -------
    np.array
        The array of intensities at the input wavelength values
    """
    # Check for radius and distance
    if isinstance(radius, q.quantity.Quantity) and isinstance(dist, q.quantity.Quantity):
        r_over_d =  (radius**2/dist**2).decompose()
    else:
        r_over_d = 1.
        
    # Get constant
    const = np.pi*2*ac.h*ac.c**2*r_over_d/(lam**(4 if Flam else 5))
    
    # Calculate intensity
    I = (const/(np.exp((ac.h*ac.c/(lam*ac.k_B*Teff)).decompose())-1)).to(q.erg/q.s/q.cm**2/(1 if Flam else q.AA))
    
    # Calculate the uncertainty
    I_unc = ''
    try:
        ex = (1-np.exp(-1.*(ac.h*ac.c/(lam*ac.k_B*Teff)).decompose()))
        I_unc = 10*(Teff_unc*I/ac.h/ac.c*lam*ac.k_B/ex).to(q.erg/q.s/q.cm**2/(1 if Flam else q.AA))
    except IOError:
        pass
        
    # Plot it
    if plot:
        plt.loglog(lam, I, label=Teff)
        try:
            plt.fill_between(lam.value, (I-I_unc).value, (I+I_unc).value, alpha=0.1)
        except IOError:
            pass
            
        plt.legend(loc=0, frameon=False)
        plt.yscale('log', nonposy='clip')
        
    return I, I_unc

def isnumber(s):
    """
    Tests to see if the given string is an int, float, or exponential
    
    Parameters
    ----------
    s: str
        The string to test
    
    Returns
    -------
    bool
        The boolean result
    """
    return s.replace('.','').replace('-','').replace('+','').isnumeric()

def finalize_spec(spec, wave_units=q.um, flux_units=q.erg/q.s/q.cm**2/q.AA):
    """
    Sort by wavelength and remove nans, negatives and zeroes

    Parameters
    ----------
    spec: sequence
        The [W,F,E] to be cleaned up
        
    Returns
    -------
    spec: sequence
        The cleaned and ordered [W,F,E]
    """
    spec = list(zip(*sorted(zip(*map(list, [[i.value if hasattr(i, 'unit') else i for i in j] for j in spec])), key=lambda x: x[0])))
    return scrub([spec[0]*wave_units, spec[1]*flux_units, spec[2]*flux_units])

def flux_calibrate(mag, dist, sig_m='', sig_d='', scale_to=10*q.pc):
    """
    Flux calibrate a magnitude to be at the distance *scale_to*
    
    Parameters
    ----------
    mag: float
        The magnitude
    dist: astropy.unit.quantity.Quantity
        The distance of the source
    sig_m: float
        The magnitude uncertainty
    sig_d: astropy.unit.quantity.Quantity
        The uncertainty in the distance
    scale_to: astropy.unit.quantity.Quantity
        The distance to flux calibrate the magnitude to
        
    Returns
    -------
    list
        The flux calibrated magnitudes
    """
    try:
        
        if isinstance(dist, q.quantity.Quantity):
            
            Mag = mag*(dist/scale_to)**2
            Mag = Mag.round(3)
            
            if isinstance(sig_d, q.quantity.Quantity) and sig_m!='':
                Mag_unc = np.sqrt((sig_m*dist/scale_to)**2 + (2*mag*(sig_d*dist/scale_to**2))**2)
                Mag_unc = Mag_unc.round(3)
                
            else:
                Mag_unc = np.nan
                
            return [Mag, Mag_unc]
            
        else:
            
            print('Could not flux calibrate that input to distance {}.'.format(dist))
            return [np.nan, np.nan]
            
    except IOError:
        
        print('Could not flux calibrate that input to distance {}.'.format(dist))
        return [np.nan, np.nan]
        

def fnu2flam(f_nu, lam, units=q.erg/q.s/q.cm**2/q.AA):
    """
    Convert a flux density as a function of frequency 
    into a function of wavelength
    
    Parameters
    ----------
    f_nu: astropy.unit.quantity.Quantity
        The flux density
    lam: astropy.unit.quantity.Quantity
        The effective wavelength of the flux
    units: astropy.unit.quantity.Quantity
        The desired units
    """
    # ergs_per_photon = (ac.h*ac.c/lam).to(q.erg)
    
    f_lam = (f_nu*ac.c/lam**2).to(units)
    
    return f_lam

def group_spectra(spectra):
    """
    Puts a list of *spectra* into groups with overlapping wavelength arrays
    """
    groups, idx, i = [], [], 'wavelength' if isinstance(spectra[0], dict) else 0
    for N, S in enumerate(spectra):
        if N not in idx:
            group, idx = [S], idx + [N]
            for n, s in enumerate(spectra):
                if n not in idx and any(np.where(np.logical_and(S[i] < s[i][-1], S[i] > s[i][0]))[0]):
                    group.append(s), idx.append(n)
            groups.append(group)
    return groups
    
def idx_include(x, include):
    try:
        return np.where(np.array(map(bool, map(sum, zip(*[np.logical_and(x > i[0], x < i[1]) for i in include])))))[0]
    except TypeError:
        try:
            return \
            np.where(np.array(map(bool, map(sum, zip(*[np.logical_and(x > i[0], x < i[1]) for i in [include]])))))[0]
        except TypeError:
            return range(len(x))
            
def idx_exclude(x, exclude):
    try:
        return np.where(~np.array(map(bool, map(sum, zip(*[np.logical_and(x > i[0], x < i[1]) for i in exclude])))))[0]
    except TypeError:
        try:
            return \
            np.where(~np.array(map(bool, map(sum, zip(*[np.logical_and(x > i[0], x < i[1]) for i in exclude])))))[0]
        except TypeError:
            return range(len(x))

def mag2flux(band, mag, sig_m='', units=q.erg/q.s/q.cm**2/q.AA):
    """
    Caluclate the flux for a given magnitude
    
    Parameters
    ----------
    band: str, svo.Filter
        The bandpass
    mag: float, astropy.unit.quantity.Quantity
        The magnitude
    sig_m: float, astropy.unit.quantity.Quantity
        The magnitude uncertainty
    units: astropy.unit.quantity.Quantity
        The unit for the output flux
    """
    try:
        if isinstance(band, str):
            band = svo.Filter(band)
        
        # Make mag unitless
        if hasattr(mag,'unit'):
            mag = mag.value
        if hasattr(sig_m,'unit'):
            sig_m = sig_m.value
        
        # Calculate the flux density
        zp = q.Quantity(band.ZeroPoint, band.ZeroPointUnit)
        f = zp*10**(mag/-2.5)
        
        if isinstance(sig_m,str):
            sig_m = np.nan
        
        sig_f = f*sig_m*np.log(10)/2.5
            
        return [f, sig_f]
        
    except IOError:
        return [np.nan, np.nan]

def make_composite(spectra):
    """
    Creates a composite spectrum from a list of overlapping spectra
    """
    units = [i.unit for i in spectra[0]]
    spectrum = spectra.pop(0)
    if spectra:
        spectra = [norm_spec(spec, spectrum) for spec in spectra]
        spectrum = [i.value for i in spectrum]
        for n, spec in enumerate(spectra):
            spec = [i.value for i in spec]
            IDX, idx = np.where(np.logical_and(spectrum[0] < spec[0][-1], spectrum[0] > spec[0][0]))[0], \
                       np.where(np.logical_and(spec[0] > spectrum[0][0], spec[0] < spectrum[0][-1]))[0]
            low_res, high_res = [i[IDX] for i in spectrum], rebin_spec([i[idx] for i in spec], spectrum[0][IDX])
            mean_spec = [spectrum[0][IDX], np.array(
                    [np.average([hf, lf], weights=[1 / he, 1 / le]) for hf, he, lf, le in
                     zip(high_res[1], high_res[2], low_res[1], low_res[2])]),
                         np.sqrt(low_res[2] ** 2 + high_res[2] ** 2)]
            spec1, spec2 = sorted([spectrum, spec], key=lambda x: x[0][0])
            spec1, spec2 = [i[np.where(spec1[0] < spectrum[0][IDX][0])[0]] for i in spec1], [
                i[np.where(spec2[0] > spectrum[0][IDX][-1])[0]] for i in spec2]
            spectrum = [np.concatenate([i[:-1], j[1:-1], k[1:]]) for i, j, k in zip(spec1, mean_spec, spec2)]
    return [i * Q for i, Q in zip([i.value if hasattr(i, 'unit') else i for i in spectrum], units)]
    
def norm_spec(spectrum, template, exclude=[]):
    """
    Parameters
    ----------
    spectrum: sequence
      The [w,f] or [w,f,e] astropy quantities spectrum to normalize
    template: sequence
      The [w,f] or [w,f,e] astropy quantities spectrum to be normalized to
    exclude: sequence (optional)
      A sequence of tuples defining the wavelength ranges to exclude in the normalization
    include: sequence (optional)
      A sequence of tuples defining the wavelength ranges to include in the normalization

    Returns
    -------
    spectrum: sequence
      The normalized [w,f] or [w,f,e] astropy quantities spectrum
    """
    template, spectrum, spectrum_units = np.array([np.asarray(i.value) for i in template]), np.array(
            [np.asarray(i.value) for i in spectrum]), [i.unit for i in spectrum]
    normed_spectrum = spectrum.copy()

    # Smooth both spectrum and template
    template[1], spectrum[1] = [smooth(x, 1) for x in [template[1], spectrum[1]]]

    # Find wavelength range of overlap for array masking
    spec_mask = np.logical_and(spectrum[0] > template[0][0], spectrum[0] < template[0][-1])
    temp_mask = np.logical_and(template[0] > spectrum[0][0], template[0] < spectrum[0][-1])
    spectrum, template = [i[spec_mask] for i in spectrum], [i[temp_mask] for i in template]

    # Also mask arrays in wavelength ranges specified in *exclude*
    for r in exclude:
        spec_mask = np.logical_and(spectrum[0] > r[0], spectrum[0] < r[-1])
        temp_mask = np.logical_and(template[0] > r[0], template[0] < r[-1])
        spectrum, template = [i[~spec_mask] for i in spectrum], [i[~temp_mask] for i in template]

    # Normalize the spectrum to the template based on equal integrated flux inincluded wavelength ranges
    norm = np.trapz(template[1], x=template[0]) / np.trapz(spectrum[1], x=spectrum[0])
    normed_spectrum[1:] = [i * norm for i in normed_spectrum[1:]]

    return [i * Q for i, Q in zip(normed_spectrum, spectrum_units)]

def pi2pc(dist, dist_unc='', pc2pi=False):
    """
    Calculate the parallax from a distance or vice versa
    
    Parameters
    ----------
    dist: astropy.unit.quantity.Quantity
        The parallax or distance
    dist_unc: astropy.unit.quantity.Quantity
        The uncertainty
    pc2pi: bool
        Convert from distance to parallax
    """
    if dist:
        unit = q.mas if pc2pi else q.pc
        if isinstance(dist_unc,str):
            dist_unc = 0*unit
        
        val = ((1*q.pc*q.arcsec)/dist).to(unit)
        err = (dist_unc*val/dist).to(unit)
        
        return [val.round(2), err.round(2)]
        
    else:
        return [np.nan, np.nan]
        
def rebin_spec(spec, wavnew, oversamp=100, plot=False):
    """
    Rebin a spectrum to a new wavelength array while preserving 
    the total flux
    
    Parameters
    ----------
    spec: array-like
        The wavelength and flux to be binned
    wavenew: array-like
        The new wavelength array
        
    Returns
    -------
    np.ndarray
        The rebinned flux
    
    """
    nlam = len(spec[0])
    x0 = np.arange(nlam, dtype=float)
    x0int = np.arange((nlam-1.)*oversamp + 1., dtype=float)/oversamp
    w0int = np.interp(x0int, x0, spec[0])
    spec0int = np.interp(w0int, spec[0], spec[1])/oversamp
    try:
        err0int = np.interp(w0int, spec[0], spec[2])/oversamp
    except:
        err0int = ''
        
    # Set up the bin edges for down-binning
    maxdiffw1 = np.diff(wavnew).max()
    w1bins = np.concatenate(([wavnew[0]-maxdiffw1], .5*(wavnew[1::]+wavnew[0:-1]), [wavnew[-1]+maxdiffw1]))
    
    # Bin down the interpolated spectrum:
    w1bins = np.sort(w1bins)
    nbins = len(w1bins)-1
    specnew = np.zeros(nbins)
    errnew = np.zeros(nbins)
    inds2 = [[w0int.searchsorted(w1bins[ii], side='left'), w0int.searchsorted(w1bins[ii+1], side='left')] for ii in range(nbins)]

    for ii in range(nbins):
        specnew[ii] = np.sum(spec0int[inds2[ii][0]:inds2[ii][1]])
        try:
            errnew[ii] = np.sum(err0int[inds2[ii][0]:inds2[ii][1]])
        except:
            pass
            
    if plot:
        plt.figure()
        plt.loglog(spec[0], spec[1], c='b')    
        plt.loglog(wavnew, specnew, c='r')
        
    return [wavnew,specnew,errnew]
    
def scrub(data):
    """
    For input data [w,f,e] or [w,f] returns the list with NaN, negative, and zero flux (and corresponsing wavelengths and errors) removed.
    """
    units = [i.unit if hasattr(i, 'unit') else 1 for i in data]
    data = [np.asarray(i.value if hasattr(i, 'unit') else i, dtype=np.float32) for i in data if isinstance(i, np.ndarray)]
    data = [i[np.where(~np.isinf(data[1]))] for i in data]
    data = [i[np.where(np.logical_and(data[1] > 0, ~np.isnan(data[1])))] for i in data]
    data = [i[np.unique(data[0], return_index=True)[1]] for i in data]
    return [i[np.lexsort([data[0]])] * Q for i, Q in zip(data, units)]
        
def smooth(x, beta):
    """
    Smooths a spectrum *x* using a Kaiser-Bessel smoothing window of narrowness *beta* (~1 => very smooth, ~100 => not smooth)
    """
    window_len = 11
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, beta)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[5:len(y) - 5] * (x.unit if hasattr(x, 'unit') else 1)
    
def specType(SpT, types=[i for i in 'OBAFGKMLTY'], verbose=False):
    """
    Converts between float and letter/number spectral types (e.g. 14.5 => 'B4.5' and 'A3' => 23).
    
    Parameters
    ----------
    SpT: float, str
        Float spectral type or letter/number spectral type between O0.0 and Y9.9
    types: list
        The MK spectral type letters to include, e.g. ['M','L','T','Y']
      
    Returns
    -------
    list, str
        The [spectral type, uncertainty, prefix, gravity, luminosity class] of the spectral type
    """
    try:
        # String input
        if isinstance(SpT, (str,bytes)):
            
            # Convert bytes to string
            if isinstance(SpT, bytes):
                SpT = SpT.decode("utf-8")
                
            # Get the MK spectral class
            MK = types[np.where([i in SpT for i in types])[0][0]]
            
            if MK:
                
                # Get the stuff before and after the MK class
                pre, suf = SpT.split(MK)
                
                # Get the numerical value
                val = float(re.findall(r'[0-9]\.?[0-9]?', suf)[0])
                
                # Add the class value
                val += types.index(MK)*10
                
                # See if low SNR
                if '::' in suf:
                    unc = 2
                    suf = suf.replace('::','')
                elif ':' in suf:
                    unc = 1
                    suf = suf.replace(':','')
                else:
                    unc = 0.5
                    
                # Get the gravity class
                if 'b' in suf or 'beta' in suf:
                    grv = 'b'
                elif 'g' in suf or 'gamma' in suf:
                    grv = 'g'
                else:
                    grv = ''
                    
                # Clean up the suffix
                suf = suf.replace(str(val), '').replace('n','').replace('e','')\
                         .replace('w','').replace('m','').replace('a','')\
                         .replace('Fe','').replace('-1','').replace('?','')\
                         .replace('-V','').replace('p','')
                        
                # Check for luminosity class
                LC = []
                for cl in ['III','V','IV']:
                    if cl in suf:
                        LC.append(cl)
                        suf.replace(cl, '')
                LC = '/'.join(LC) or 'V'
                            
                return [val, unc, pre, grv, LC]
            
            else:
                print('Not in list of MK spectral classes',types)
                return [np.nan, np.nan, '', '', '']
                
        # Numerical or list input
        elif isinstance(SpT, (float,int,list,tuple)):
            if isinstance(SpT, (int,float)):
                SpT = [SpT]
                
            # Get the MK class
            MK = ''.join(types)[int(SpT[0]//10)]
            num = int(SpT[0]%10) if SpT[0]%10==int(SpT[0]%10) else SpT[0]%10
            
            # Get the uncertainty
            if len(SpT)>1:
                if SpT[1]==':' or SpT[1]==1:
                    unc = ':'
                elif SpT[1]=='::' or SpT[1]==2:
                    unc = '::'
                else:
                    unc = ''
            else:
                unc = ''
                
            # Get the prefix
            if len(SpT)>2:
                pre = str(SpT[2])
            else:
                pre = ''
                
            # Get the gravity
            if len(SpT)>3:
                grv = str(SpT[3])
            else:
                grv = ''
                
            # Get the gravity
            if len(SpT)>4:
                LC = str(SpT[4])
            else:
                LC = ''
                
            return ''.join([pre,MK,str(num),grv,LC,unc])
            
        # Bogus input
        else:
            if verbose:
                print('Spectral type',SpT,'must be a float between 0 and',len(types)*10,'or a string of class',types)
            return
        
    except IOError:
        return
        
def str2Q(x, target=''):
    """
    Given a string of units unconnected to a number, returns the units as a quantity to be multiplied with the number.
    Inverse units must be represented by a forward-slash prefix or negative power suffix, e.g. inverse square seconds may be "/s2" or "s-2"

    *x*
      The units as a string, e.g. str2Q('W/m2/um') => np.array(1.0) * W/(m**2*um)
    *target*
      The target units as a string if rescaling is necessary, e.g. str2Q('Wm-2um-1',target='erg/s/cm2/cm') => np.array(10000000.0) * erg/(cm**3*s)
    """
    if x:
        def Q(IN):
            OUT = 1
            text = ['Jy', 'erg', '/s', 's-1', 's', '/um', 'um-1', 'um', '/nm', 'nm-1', 'nm', '/cm2', 'cm-2', 'cm2',
                    '/cm', 'cm-1', 'cm', '/A', 'A-1', 'A', 'W', '/m2', 'm-2', 'm2', '/m', 'm-1', 'm', '/Hz', 'Hz-1']
            vals = [q.Jy, q.erg, q.s ** -1, q.s ** -1, q.s, q.um ** -1, q.um ** -1, q.um, q.nm ** -1, q.nm ** -1, q.nm,
                    q.cm ** -2, q.cm ** -2, q.cm ** 2, q.cm ** -1, q.cm ** -1, q.cm, q.AA ** -1, q.AA ** -1, q.AA, q.W,
                    q.m ** -2, q.m ** -2, q.m ** 2, q.m ** -1, q.m ** -1, q.m, q.Hz ** -1, q.Hz ** -1]
            for t, v in zip(text, vals):
                if t in IN:
                    OUT = OUT * v
                    IN = IN.replace(t, '')
            return OUT

        unit = Q(x)
        if target:
            z = str(Q(target)).split()[-1]
            try:
                unit = unit.to(z)
            except ValueError:
                print("{} could not be rescaled to {}".format(unit, z))

        return unit
    else:
        return q.Unit('')
        
def trim_spectrum(spectrum, regions, smooth_edges=False):
    trimmed_spec = [i[idx_exclude(spectrum[0], regions)] for i in spectrum]
    if smooth_edges:
        for r in regions:
            try:
                if any(spectrum[0][spectrum[0] > r[1]]):
                    trimmed_spec = inject_average(trimmed_spec, r[1], 'right', n=smooth_edges)
            except:
                pass
            try:
                if any(spectrum[0][spectrum[0] < r[0]]):
                    trimmed_spec = inject_average(trimmed_spec, r[0], 'left', n=smooth_edges)
            except:
                pass
    return trimmed_spec