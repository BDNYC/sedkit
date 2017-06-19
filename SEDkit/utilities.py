#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
Some utilities to accompany SEDkit
"""
import numpy as np
import astropy.units as q
import astropy.constants as ac
from svo_filters import svo

FILTERS = svo.filters()

from astropy.utils.data_info import ParentDtypeInfo

class ArrayWrapper(object):
    """
    Minimal mixin using a simple wrapper around a numpy array
    """
    info = ParentDtypeInfo()

    def __init__(self, data):
        self.data = np.array(data)
        if 'info' in getattr(data, '__dict__', ()):
            self.info = data.info

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            out = self.data[item]
        else:
            out = self.__class__(self.data[item])
            if 'info' in self.__dict__:
                out.info = self.info
        return out

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
        Mag = mag*(dist/scale_to)**2
        
        if sig_d.value and sig_m.value:
            Mag_unc = np.sqrt((sig_m*dist/scale_to)**2 + (2*mag*(sig_d*dist/scale_to**2))**2)
        
        else:
            Mag_unc = np.nan
        
        return [Mag.round(3), Mag_unc.round(3)]
        
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
        zp = filt['ZeroPoint']*q.Unit(filt['ZeroPointUnit'])
        f = zp*10**(mag/-2.5)
        
        if isinstance(sig_m,str):
            sig_m = np.nan
        
        sig_f = f*sig_m*np.log(10)/2.5
            
        return [f, sig_f]
        
    except IOError:
        return [np.nan, np.nan]
    
    
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