#!/usr/bin/python
# Synthetic Photomotery
import warnings, glob, os, cPickle, numpy as np, astropy.units as q, astropy.constants as ac, matplotlib.pyplot as plt, utilities as u
from itertools import combinations, chain, groupby
warnings.simplefilter('ignore')
RSR = u.get_filters()

def vega(bbody=False):
  '''
  Returns the wavelength [um] and energy flux in [erg/s/cm2/A] calibrated to 10pc for Vega (http://www.stsci.edu/hst/observatory/cdbs/calspec.html)
  '''
  (w, f), pi = [np.array(map(float,i)) for i in zip(*u.txt2dict('/Models/Vega/STSci_Vega.txt', to_list=True, skip=['#']))], 130.23
  w, f = (w*q.AA).to(q.um), f*q.erg/q.s/q.cm**2/q.AA    
  return [w, f, u.blackbody(w, 9610, Flam=False, radius=27.3)] if bbody else [w, f]

def get_eff(band, wave, flux, photon=True, calculate=False):
  '''
  Calculates the filter effective wavelength for a given *band*, *wave* and *flux*. If *wave* is out of range (uncalculable), returns the given value.
  '''
  filt = RSR[band]
  if (wave[0]<filt['wav'][0]) and (wave[-1]>filt['wav'][-1]) and calculate:
    I = np.interp(wave.value, filt['wav'].value, filt['rsr'].value, left=0, right=0)
    return (np.trapz(wave*flux*I*(wave/(ac.h*ac.c) if photon else 1), x=wave)/np.trapz(flux*I*(wave/(ac.h*ac.c) if photon else 1), x=wave)).to(q.um)
  else:
    return filt['eff']

def get_zp(band, photon=True):
  '''
  Calculates the zero point flux density for a given *band* using a flux calibrated Vega SED [erg/s/cm2/A]
  '''
  # return (np.trapz((u.rebin_spec(vega(), RSR[band]['wav'])[1]*RSR[band]['rsr']*((RSR[band]['wav']/(ac.h*ac.c)).to(1/q.erg) if photon else 1)).to((1 if photon else q.erg)/q.s/q.cm**2/q.AA), x=RSR[band]['wav'])*q.um*(1 if photon else q.erg)/q.s/q.cm**2/(np.trapz(RSR[band]['rsr'], x=RSR[band]['wav'])*q.um)).to((1 if photon else q.erg)/q.s/q.cm**2)
  return (np.trapz((u.rebin_spec(vega(), RSR[band]['wav'])[1]*RSR[band]['rsr']*((RSR[band]['wav']/(ac.h*ac.c)).to(1/q.erg) if photon else 1)).to((1 if photon else q.erg)/q.s/q.cm**2/q.AA), x=RSR[band]['wav'])/np.trapz(RSR[band]['rsr'], x=RSR[band]['wav'])).to((1 if photon else q.erg)/q.s/q.cm**2/q.AA)

def get_mag(band, spectrum, exclude=[], airmass=0, photon=True, plot=False, to_flux=False, Flam=False, eff=False):
  '''
  Returns the magnitude in *band* for given *wave* in [um] and *flux* in [erg/s/cm2/A] with *airmass* corrections if provided. If uncertainty is provided, returns [mag, mag_uncertainty] instead.
  '''
  spectrum, filt = u.scrub(spectrum), RSR.get(band)
  if len(spectrum)==2: spectrum += ['']
  if filt and np.logical_and(filt['max']<spectrum[0][-1], filt['min']>spectrum[0][0]) and all([np.logical_or(all([i<filt['min'].value for i in rng]),all([i>filt['max'].value for i in rng])) for rng in exclude]):
    # Calculate synthetic magnitude
    f, sig_f = u.rebin_spec(spectrum, filt['wav'])[1:] if spectrum[2] is not '' else [u.rebin_spec(spectrum, filt['wav'])[1], '']
    F = (np.trapz((f*filt['rsr']*((filt['wav']/(ac.h*ac.c)).to(1/q.erg) if photon else 1)).to((1 if photon else q.erg)/q.s/q.cm**2/q.AA), x=filt['wav'])/(np.trapz(filt['rsr'], x=filt['wav']))).to((1 if photon else q.erg)/q.s/q.cm**2/(1 if Flam else q.AA))
    sig_F = np.sqrt(np.sum(((sig_f*filt['rsr']*np.gradient(filt['wav']).value*((filt['wav']/(ac.h*ac.c)).to(1/q.erg) if photon else 1))**2).to((1 if photon else q.erg)**2/q.s**2/q.cm**4/(1 if Flam else q.AA**2)))) if sig_f else '' 
  
    # Calculate effective wavelength
    w = get_eff(band, spectrum[0], spectrum[1], photon=photon, calculate=eff)
    
    return (u.flux2mag(band, F, sig_f=sig_F, filter_dict=RSR) if not to_flux else [F, sig_F])+[w]
    
  else: return ['','','']

def all_mags(spectrum, bands=RSR.keys(), airmass=0, photon=True, to_flux=False, Flam=True, exclude=[], eff=False, to_list=False):
  magDict, magList = {}, []
  for band in bands:
    M = get_mag(band, spectrum, airmass=airmass, photon=photon, to_flux=to_flux, Flam=Flam, exclude=exclude, eff=eff)
    if M[0]: 
      magDict[band], magDict[band+'_unc'], magDict[band+'_eff'] = M
      magList.append(M)
  return sorted(magList, key=lambda x: x[-1]) if to_list else magDict