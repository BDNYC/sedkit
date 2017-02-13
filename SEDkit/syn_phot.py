#!/usr/bin/python
# Synthetic Photomotery
import warnings
import glob
import os
import numpy as np
import astropy.units as q
import astropy.constants as ac
import matplotlib.pyplot as plt
from ExoCTK import core

warnings.simplefilter('ignore')

def get_filters(path, systems=[]):
    """
    Load a dictionary of ExoCTK.core.Filter() objects
    
    Parameters
    ----------
    path: str
        The path to the directory of filters
    systems: list
        The photometric systems to include
    """
    # Get only specific systems
    if systems:
        filters = []
        for system in systems:
            filters += glob.glob(path+'{}.*'.format(system))
    
    # Get all the filters
    else:
        filters = glob.glob(path+'*')
    
    # Add it to the filter dictionary
    filter_dict = {}
    for f in filters:
        try:                                           
            name = os.path.basename(f)
            filter_dict[name] = core.Filter(name, os.path.dirname(f)+'/')
        except:
            pass
            
    if not filter_dict:
        print('No filters in {}{}'.format(path, 
        ' with systems {}'.format(systems) if systems else ''))
    
    return filter_dict

def get_mag(band, spectrum, units=''):
    """
    Calculate the synthetic magnitude of the given spectrum
    
    Parameters
    ----------
    band: str
        The photometric band
    spectrum: array-like
        The wavelength [um] and flux of the spectrum to
        be convolved
    units: astropy.units (optional)
        The units of the output magnitude, 
        e.g. erg/s/cm2/A (flux density), 
        photons/s/cm2 (photon flux),
        or returns mags if False
        
    Returns
    -------
    np.ndarray
        The convolved spectrum
        
    """
    # Get the filter
    name = os.path.basename(band)
    filt = core.Filter(name, os.path.dirname(band)+'/')
    
    # Get the spectrum arrays
    wav, flx = spectrum[:2]
    err = ''
    if len(spectrum)==3:
        err = spectrum[2]
        
    # Get input units and decomposed base units
    units_in = flx.unit
    base = 1*units.decompose().unit
    
    # Apply the filter
    flx = filt.convolve([wav,flx])
    if err:
        err = filt.convolve([wav,err])
    
    # Power density or energy flux
    if base==q.kg/q.m/q.s**3\
    or base==q.kg/q.s**3:
        factor = 1.
            
    # Photon density or photon flux
    elif base==1/q.m**3/q.s\
    or base==1/q.m**2/q.s:
        factor = wav/ac.h/ac.c
    
    # Calculate the integrated flux and error
    F = np.trapz(flx*factor, x=wav)/np.trapz(filt.rsr[1], x=wav)
    E = np.sqrt(np.sum((err*np.gradient(wav).value*factor)**2)) \
        if err else ''
    
    # Return fluxes or mags
    if units:
        return [F.to(units), E.to(units)]
    else:
        return u.flux2mag(name, F, sig_f=E)


# def all_mags(spectrum, bands=RSR.keys(), photon=True, to_flux=False, Flam=True, exclude=[], eff=False, to_list=False):
#     magDict, magList = {}, []
#     for band in bands:
#         M = get_mag(band, spectrum, photon=photon, to_flux=to_flux, Flam=Flam, exclude=exclude, eff=eff)
#         if M[0]:
#             magDict[band], magDict[band + '_unc'], magDict[band + '_eff'] = M
#             magList.append(M)
#     return sorted(magList, key=lambda x: x[-1]) if to_list else magDict
