#!/usr/bin/python
# Synthetic Photomotery
import warnings
import os
import numpy as np
import astropy.units as q
import astropy.constants as ac
import astropy.table as at
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from . import utilities as u
from svo_filters import svo
import pkg_resources

FILTERS = svo.filters()
warnings.simplefilter('ignore')

def all_mags(spectrum, bands='', plot=False, **kwargs):
    """
    Calculate the magnitudes of the given spectrum in all the given bands
    
    Parameters
    ----------
    spectrum: array-like
        The [w,f,e] of the spectrum with astropy.units
    bands: array-like (optional)
        The photometric bands to try
    
    Returns
    -------
    astropy.table
        A table of the caluclated synthetic photometry
    """
    # Get list of bands to try
    if not any(bands):
        bands = FILTERS['Band']
    w_unit = spectrum[0].unit
    f_unit = spectrum[1].unit
    
    # Calculate the magnitude
    mag_list = []
    for bandpass in bands:
        m, sig_m, F, sig_F = get_mag(spectrum, bandpass, fetch='both', **kwargs)
        
        # Only add it to the table if the magnitude is calculated
        if isinstance(m, float) and m!=np.nan:
            eff = FILTERS.loc[bandpass]['WavelengthEff']
            w_unit = q.Unit(FILTERS.loc[bandpass]['WavelengthUnit'])
            f_unit = F.unit
            mag_list.append([bandpass, eff, m, sig_m, F.value, sig_F.value])
            
    # Make the table of values
    data = list(map(list, zip(*mag_list))) if mag_list else None
    table = at.QTable(data, names=['band','eff','app_magnitude','app_magnitude_unc','app_flux','app_flux_unc'])
    
    if plot and data:
        plt.figure()
        plt.step(spectrum[0], spectrum[1], color='k', label='Spectrum')
        plt.errorbar(table['eff'], table['app_flux'], yerr=table['app_flux_unc'], marker='o', ls='none', label='Magnitudes')
        try:
            plt.fill_between(spectrum[0], spectrum[1]+spectrum[2], spectrum[1]+spectrum[2], color='k', alpha=0.1)
        except:
            pass
        plt.xlabel(w_unit)
        plt.ylabel(f_unit)
        plt.legend(loc=0, frameon=False)
    
    # Add units to the columns
    table['eff'].unit = w_unit
    table['app_magnitude'].unit = q.mag
    table['app_magnitude_unc'].unit = q.mag
    table['app_flux'].unit = f_unit
    table['app_flux_unc'].unit = f_unit
    
    return table

def norm_to_mags(spec, to_mags, weighting=True, reverse=False, extend='', plot=False):
    """
    Normalize the given spectrum to the given dictionary of magnitudes

    Parameters
    ----------
    spec: sequence
        The [W,F,E] to be normalized
    to_mags: astropy.table
        The table of magnitudes to normalize to
    weighting: bool
        Weight the normalization based on the bandwidth
    reverse: bool
        Norm `to_mags` to `spec`
    extend: sequence
        An additional [W,F,E] to append to force band coverage
        (e.g. a blackbody RJ tail for IRS coverage of W3 band)
    
    Returns
    -------
    list
        The normalized [W,F,E]
    """
    norm = 1.
    spec = [spec[0] * (q.um if not hasattr(spec[0], 'unit') else 1.), \
            spec[1] * (q.erg / q.s / q.cm ** 2 / q.AA if not hasattr(spec[1], 'unit') else 1.), \
            spec[2] * (q.erg / q.s / q.cm ** 2 / q.AA if not hasattr(spec[2], 'unit') else 1.)]
    units = [i.unit for i in spec]
            
    # Native red and blue limits
    spec_original = spec.copy()
    blue, red = spec[0][0], spec[0][-1]
    
    # Append extended spectrum if overlapping
    if isinstance(extend,(list,tuple,np.ndarray)):
        if (extend[0][0]<spec[0][-1] and extend[0][-1]>spec[0][-1])\
        or (extend[0][0]<spec[0][0] and extend[0][-1]>spec[0][0]):
            spec = [np.concatenate([i.value,j.value]) for i,j in zip(spec,extend)]
            spec = [i*Q for i,Q in zip(spec,units)]
    
    # Fudge for J band coverage
    if blue > 1.08 * q.um and blue < 1.12 * q.um:
        spec[0][0] *= 1.08 / blue.value
    elif blue > 1.47 * q.um and blue < 1.55 * q.um:
        spec[0][0] *= 1.47 / blue.value
    elif blue > 1.95 * q.um and blue < 2.00 * q.um:
        spec[0][0] *= 1.95 / blue.value
    else:
        pass
        
    # Fudge for K band coverage
    if red > 1.3 * q.um and red < 1.41 * q.um:
        spec[0][-1] *= 1.41 / red.value
    elif red > 1.76 * q.um and red < 1.825 * q.um:
        spec[0][-1] *= 1.825 / red.value
    elif red > 2.3 * q.um and red < 2.356 * q.um:
        spec[0][-1] *= 2.356 / red.value
    else:
        pass
        
    # Calculate all synthetic magnitudes
    mags = all_mags(spec, bands=to_mags['band'])
    
    # Restore original spectrum
    spec = spec_original
        
    try:
        # Get list of all bands in common
        bands = [b for b in list(set(mags['band']).intersection(set(to_mags['band'])))]
        
        # Get eff, m, m_unc from each
        eff = [FILTERS.loc[b]['WavelengthEff'] for b in bands]
        weight = np.ones(len(bands))
        if weighting:
            weight = np.array([FILTERS.loc[b]['WavelengthMax']-FILTERS.loc[b]['WavelengthMin'] for b in bands])
        
        # Get the observed magnitudes
        to_mags.add_index('band')
        f1 = np.array([np.nanmean(to_mags.loc[b]['app_flux'].value) for b in bands])
        e1 = np.array([np.nanmean(to_mags.loc[b]['app_flux_unc'].value) for b in bands])
        
        # Get the synthetic magnitudes 
        mags.add_index('band')
        f2 = np.array([np.nanmean(mags.loc[b]['app_flux'].value) for b in bands])
        e2 = np.array([np.nanmean(mags.loc[b]['app_flux_unc'].value) for b in bands])
        
        # Calculate normalization factor that minimizes the function
        norm = np.nansum(weight*f1*f2/(e1**2+e2**2))/np.nansum(weight*f2**2/(e1**2+e2**2))
        
        # Plotting test
        if plot:
            plt.loglog(spec[0].value, spec[1].value, label='old', color='k')
            plt.loglog(spec[0].value, spec[1].value*norm, label='new', color='r')
            plt.scatter(eff, f1, c='r')
            plt.scatter(eff, f2, c='k')
            plt.legend()
            
        return [spec[0], spec[1]/norm, spec[2]/norm] if reverse else [spec[0], spec[1]*norm, spec[2]*norm]
        
    except IOError:
        print('No overlapping photometry for normalization!')
        return spec

def vega(bbody=False):
    '''
    Returns the wavelength [um] and energy flux in [erg/s/cm2/A] calibrated to 10pc for Vega (
    http://www.stsci.edu/hst/observatory/cdbs/calspec.html)
    '''
    pi = 130.23
    w, f = np.genfromtxt(pkg_resources.resource_filename('SEDkit', 'data/STScI_Vega.txt'), unpack=True)
    w, f = (w * q.AA).to(q.um), f * q.erg / q.s / q.cm ** 2 / q.AA
    return [w, f, u.blackbody(w, 9610, Flam=False, radius=27.3)] if bbody else [w, f]
    
def norm_to_mag(spectrum, magnitude, bandpass):
    """
    Returns the flux of a given *spectrum* [W,F] normalized to the given *magnitude* in the specified photometric *band*
    """
    # Get the unnormalized flux
    flx, flx_unc = get_mag(spectrum, bandpass, fetch='flux')
    
    # Convert input magnitude to flux
    target, target_unc = mag2flux(bandpass.filterID.split('/')[1], magnitude, sig_m='', units=spectrum[1].unit)
    
    # Normalize the spectrum
    spectrum[1] *= target/flx
    
    return spectrum

def flux2mag(bandpass, f, sig_f='', photon=False):
    """
    For given band and flux returns the magnitude value (and uncertainty if *sig_f*)
    """
    eff = bandpass.WavelengthEff
    zp = bandpass.ZeroPoint
    unit = q.erg/q.s/q.cm**2/q.AA
    
    # Calculate zeropoint
    zp = get_mag(vega(), bandpass, fetch='flux')[0]
    # print(bandpass.filterID,bandpass.ZeroPoint/zp)
    
    # Convert to f_lambda if necessary
    if f.unit == 'Jy':
        f,  = (ac.c*f/eff**2).to(unit)
        sig_f = (ac.c*sig_f/eff**2).to(unit)
    
    # Convert energy units to photon counts
    if photon:
        f = (f*(eff/(ac.h*ac.c)).to(1/q.erg)).to(unit/q.erg), 
        sig_f = (sig_f*(eff/(ac.h*ac.c)).to(1/q.erg)).to(unit/q.erg)
    
    # Calculate magnitude
    m = -2.5*np.log10((f/zp).value)
    sig_m = (2.5/np.log(10))*(sig_f/f).value if sig_f else ''
    
    return [m, sig_m]

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
        # Get the band info
        filt = FILTERS.loc[band]
        
        # Make mag unitless
        if hasattr(mag,'unit'):
            mag = mag.value
        if hasattr(sig_m,'unit'):
            sig_m = sig_m.value
        
        # Calculate the flux density
        zp = q.Quantity(filt['ZeroPoint'], filt['ZeroPointUnit'])
        f = zp*10**(mag/-2.5)
        
        if isinstance(sig_m,str):
            sig_m = np.nan
        
        sig_f = f*sig_m*np.log(10)/2.5
            
        return [f, sig_f]
        
    except IOError:
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
        plt.loglog(spec[0], spec[1], c='b', label='Input')
        plt.loglog(wavnew, specnew, c='r', label='Output')
        plt.legend()
        
    return [wavnew,specnew,errnew]

def get_mag(spectrum, bandpass, exclude=[], fetch='mag', photon=False, Flam=False, plot=False):
    """
    Returns the integrated flux of the given spectrum in the given band
    
    Parameters
    ---------
    spectrum: array-like
        The [w,f,e] of the spectrum with astropy.units
    bandpass: str, svo_filters.svo.Filter
        The bandpass to calculate
    exclude: sequecne
        The wavelength ranges to exclude by linear interpolation between gap edges
    photon: bool
        Use units of photons rather than energy
    Flam: bool
        Use flux units rather than the default flux density units
    plot: bool
        Plot it
    
    Returns
    -------
    list
        The integrated flux of the spectrum in the given band
    """
    # Get the Filter object if necessary
    if isinstance(bandpass, str):
        bandpass = svo.Filter(bandpass)
        
    # Get filter data in order
    unit = q.Unit(bandpass.WavelengthUnit)
    mn = bandpass.WavelengthMin*unit
    mx = bandpass.WavelengthMax*unit
    wav, rsr = bandpass.raw
    wav = wav*unit
    
    # Unit handling
    a = (1 if photon else q.erg)/q.s/q.cm**2/(1 if Flam else q.AA)
    b = (1 if photon else q.erg)/q.s/q.cm**2/q.AA
    c = 1/q.erg
    
    # Test if the bandpass has full spectral coverage
    coverage = np.logical_and(mx<np.max(spectrum[0]), mn>np.min(spectrum[0]))
    excludes = [np.logical_or(all([i<mn for i in rng]), all([i>mx for i in rng])) for rng in exclude]
    if coverage and all(excludes):
        
        # Rebin spectrum to bandpass wavelengths
        # w, f, sig_f = rebin_spec([i.value for i in spectrum], wav.value, plot=plot)
        w = wav
        f = np.interp(wav.value, spectrum[0].value, spectrum[1].value, left=0, right=0)
        try:
            sig_f = np.interp(wav.value, spectrum[0].value, spectrum[2].value, left=0, right=0)
        except:
            sig_f = ''
        
        w = w*unit
        f = f*b
        sig_f = sig_f*b
        
        # Calculate the integrated flux, subtracting the filter shape
        F = (np.trapz((f*rsr*((wav/(ac.h*ac.c)).to(c) if photon else 1)).to(b), x=wav)/(np.trapz(rsr, x=wav))).to(a)
        
        # Caluclate the uncertainty
        if sig_f:
            sig_F = np.sqrt(np.sum(((sig_f*rsr*np.gradient(wav).value*((wav/(ac.h*ac.c)).to(c) if photon else 1))**2).to(a**2)))
        else:
            sig_F = ''
            
        # Make a plot
        if plot:
            plt.figure()
            plt.step(spectrum[0], spectrum[1], color='k', label='Spectrum')
            plt.errorbar(bandpass.WavelengthEff, F.value, yerr=sig_F.value, marker='o', label='Magnitude')
            try:
                plt.fill_between(spectrum[0], spectrum[1]+spectrum[2], spectrum[1]+spectrum[2], color='k', alpha=0.1)
            except:
                pass
            plt.plot(bandpass.rsr[0], bandpass.rsr[1]*F, label='Bandpass')
            plt.xlabel(unit)
            plt.ylabel(a)
            plt.legend(loc=0, frameon=False)
            
        # Get magnitude from flux
        if fetch=='flux':
            m = sig_m = np.nan
        else:
            m, sig_m = flux2mag(bandpass, F, sig_f=sig_F)
        
        return [m, sig_m, F, sig_F] if fetch=='both' else [F, sig_F] if fetch=='flux' else [m, sig_m]
        
    else:
        return ['']*4 if fetch=='both' else ['']*2