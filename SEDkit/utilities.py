#!/usr/bin/python
# Utilities
import warnings, glob, os, re, xlrd, cPickle, itertools, astropy.units as q, astropy.constants as ac, numpy as np, matplotlib.pyplot as plt, astropy.coordinates as apc, scipy.stats as st, astropy.io.fits as pf, scipy.optimize as opt
from random import random
from heapq import nsmallest, nlargest
from scipy.interpolate import Rbf
warnings.simplefilter('ignore')

def blackbody(lam, T, Flam=False, radius=1, dist=10, emitted=False):
  """
  Given a wavelength array [um] and temperature [K], returns an array of Planck function values in [erg s-1 cm-2 A-1]
  """
  lam, T = lam.to(q.cm), T*q.K
  I = np.pi*(2*ac.h*ac.c**2 / (lam**(4 if Flam else 5) * (np.exp((ac.h*ac.c / (lam*ac.k_B*T)).decompose()) - 1))).to(q.erg/q.s/q.cm**2/(1 if Flam else q.AA))
  return I if emitted else I*((ac.R_jup*radius/(dist*q.pc))**2).decompose()
  
def filter_info(band):
  """
  Effective, min, and max wavelengths in [um] and zeropoint in [erg s-1 cm-2 A-1] and [photon s-1 cm-2 A-1] for SDSS, Bessel, 2MASS, IRAC and WISE filters IN THE VEGA SYSTEM. Values from SVO filter profile service.
  
  *band*
      Name of filter band (e.g. 'J' from 2MASS, 'W1' from WISE, etc.) or list of filter systems (e.g. ['SDSS','2MASS','WISE'])
  """
  Filters = { "GALEX_FUV":     { 'eff': 0.154226, 'min': 0.134032, 'max': 0.180643, 'zp': 6.486734e-09, 'zp_photon': 5.035932e+02, 'toVega':0,      'ext': 2.62,   'system': 'GALEX' },
              "GALEX_NUV":     { 'eff': 0.227437, 'min': 0.169252, 'max': 0.300667, 'zp': 4.511628e-09, 'zp_photon': 5.165788e+02, 'toVega':0,      'ext': 2.94,   'system': 'GALEX' },

              "Johnson_U":       { 'eff': 0.357065, 'min': 0.303125, 'max': 0.417368, 'zp': 3.656264e-09, 'zp_photon': 6.576522e+02, 'toVega':0.0915, 'ext': 1.56,   'system': 'Johnson' }, 
              "Johnson_B":       { 'eff': 0.437812, 'min': 0.363333, 'max': 0.549706, 'zp': 6.286883e-09, 'zp_photon': 1.385995e+03, 'toVega':0.0069, 'ext': 1.31,   'system': 'Johnson' }, 
              "Johnson_V":       { 'eff': 0.544579, 'min': 0.473333, 'max': 0.687500, 'zp': 3.571744e-09, 'zp_photon': 9.837109e+02, 'toVega':0,      'ext': 1.02,   'system': 'Johnson' },
              "Cousins_R":       { 'eff': 0.641420, 'min': 0.550435, 'max': 0.883333, 'zp': 2.157178e-09, 'zp_photon': 6.971704e+02, 'toVega':0.0018, 'ext': 0.83,   'system': 'Cousins' },
              "Cousins_I":       { 'eff': 0.797880, 'min': 0.704167, 'max': 0.916667, 'zp': 1.132454e-09, 'zp_photon': 4.549636e+02, 'toVega':-0.0014,'ext': 0.61,   'system': 'Cousins' },

              "SDSS_u":       { 'eff': 0.3543,   'min': 0.304828, 'max': 0.402823, 'zp': 3.617963e-09, 'zp_photon': 6.546739e+02, 'toVega':0.91,   'ext': 1.58,   'system': 'SDSS' },  # AB to Vega transformations from Blanton et al. (2007)
              "SDSS_g":       { 'eff': 0.4770,   'min': 0.378254, 'max': 0.554926, 'zp': 5.491077e-09, 'zp_photon': 1.282871e+03, 'toVega':-0.08,  'ext': 1.23,   'system': 'SDSS' },  # AB to Vega transformations from Blanton et al. (2007)
              "SDSS_r":       { 'eff': 0.6231,   'min': 0.541534, 'max': 0.698914, 'zp': 2.528924e-09, 'zp_photon': 7.794385e+02, 'toVega':0.16,   'ext': 0.89,   'system': 'SDSS' },  # AB to Vega transformations from Blanton et al. (2007)
              "SDSS_i":       { 'eff': 0.7625,   'min': 0.668947, 'max': 0.838945, 'zp': 1.409436e-09, 'zp_photon': 5.278550e+02, 'toVega':0.37,   'ext': 0.68,   'system': 'SDSS' },  # AB to Vega transformations from Blanton et al. (2007)
              "SDSS_z":       { 'eff': 0.9134,   'min': 0.796044, 'max': 1.083325, 'zp': 8.501067e-10, 'zp_photon': 3.807540e+02, 'toVega':0.54,   'ext': 0.52,   'system': 'SDSS' },  # AB to Vega transformations from Blanton et al. (2007)
              
              "DES_u":   { 'eff': 0.3543,   'min': 0.304828, 'max': 0.402823, 'zp': 5.360165e-09, 'zp_photon': 1.038526e+03, 'toVega':0,      'ext': 0,      'system': 'DES' },
              "DES_g":   { 'eff': 0.4770,   'min': 0.378254, 'max': 0.554926, 'zp': 5.215897e-09, 'zp_photon': 1.243521e+03, 'toVega':0,      'ext': 0,      'system': 'DES' },
              "DES_r":   { 'eff': 0.6231,   'min': 0.541534, 'max': 0.698914, 'zp': 2.265389e-09, 'zp_photon': 7.234969e+02, 'toVega':0,      'ext': 0,      'system': 'DES' },
              "DES_i":   { 'eff': 0.7625,   'min': 0.668947, 'max': 0.838945, 'zp': 1.235064e-09, 'zp_photon': 4.820083e+02, 'toVega':0,      'ext': 0,      'system': 'DES' },
              "DES_z":   { 'eff': 0.9134,   'min': 0.796044, 'max': 1.083325, 'zp': 8.072777e-10, 'zp_photon': 3.712548e+02, 'toVega':0,      'ext': 0,      'system': 'DES' },
              "DES_Y":   { 'eff': 1.0289,   'min': 0.930000, 'max': 1.074600, 'zp': 6.596909e-10, 'zp_photon': 3.280450e+02, 'toVega':0,      'ext': 0,      'system': 'DES' },

              "FourStar_J1":      { 'eff': 1.052129, 'min': 0.990799, 'max': 1.120951, 'zp': 5.358674e-10, 'zp_photon': 2.838244e+02, 'toVega':0,      'ext': 0.40,   'system': 'FourStar' }, 
              "FourStar_J2":      { 'eff': 1.140731, 'min': 1.060065, 'max': 1.238092, 'zp': 4.088281e-10, 'zp_photon': 2.347727e+02, 'toVega':0,      'ext': 0.35,   'system': 'FourStar' }, 
              "FourStar_J3":      { 'eff': 1.283508, 'min': 1.200310, 'max': 1.377945, 'zp': 2.709316e-10, 'zp_photon': 1.750579e+02, 'toVega':0,      'ext': 0.29,   'system': 'FourStar' }, 
              
              "2MASS_J":       { 'eff': 1.2350,   'min': 1.080647, 'max': 1.406797, 'zp': 3.129e-10,    'zp_photon': 1.943482e+02, 'toVega':0,      'ext': 0.0166, 'system': '2MASS' }, # ZP from Cohen et al. (2003)
              "2MASS_H":       { 'eff': 1.6620,   'min': 1.478738, 'max': 1.823102, 'zp': 1.133e-10,    'zp_photon': 9.437966e+01, 'toVega':0,      'ext': 0.0146, 'system': '2MASS' }, # ZP from Cohen et al. (2003)
              "2MASS_Ks":      { 'eff': 2.1590,   'min': 1.954369, 'max': 2.355240, 'zp': 4.283e-11,    'zp_photon': 4.664740e+01, 'toVega':0,      'ext': 0.0710, 'system': '2MASS' }, # ZP from Cohen et al. (2003)

              "MKO_Y":   { 'eff': 1.02894,  'min': 0.9635,   'max': 1.1025,   'zp': 5.869238e-10, 'zp_photon': 3.033632e+02, 'toVega':0,      'ext': 0.41,   'system': 'MKO' },
              "MKO_J":   { 'eff': 1.250,    'min': 1.148995, 'max': 1.348332, 'zp': 3.01e-10,     'zp_photon': 1.899569e+02, 'toVega':0,      'ext': 0.30,   'system': 'MKO' },   # eff and ZP from Tokunaga & Vacca (2005)
              "MKO_H":   { 'eff': 1.644,    'min': 1.450318, 'max': 1.808855, 'zp': 1.18e-10,     'zp_photon': 9.761983e+01, 'toVega':0,      'ext': 0.20,   'system': 'MKO' },   # eff and ZP from Tokunaga & Vacca (2005)
              "MKO_K":   { 'eff': 2.198,    'min': 1.986393, 'max': 2.397097, 'zp': 4.00e-11,     'zp_photon': 4.488476e+01, 'toVega':0,      'ext': 0.12,   'system': 'MKO' },   # eff and ZP from Tokunaga & Vacca (2005)
              "MKO_L'":   { 'eff': 3.754,    'min': 3.326622, 'max': 4.207764, 'zp': 5.31e-12,     'zp_photon': 1.016455e+01, 'toVega':0,      'ext': 0.06,   'system': 'MKO' },   # eff and ZP from Tokunaga & Vacca (2005)
              "MKO_M'":   { 'eff': 4.702,    'min': 4.496502, 'max': 4.865044, 'zp': 2.22e-12,     'zp_photon': 5.305197e+00, 'toVega':0,      'ext': 0.05,   'system': 'MKO' },   # eff and ZP from Tokunaga & Vacca (2005)

              "DENIS_I": { 'eff': 0.78621,  'min': 0.7007,   'max': 0.9140,   'zp': 1.182102e-09, 'zp_photon': 4.681495e+02, 'toVega':0,      'ext': 0.63,   'system': 'DENIS' },
              "DENIS_J": { 'eff': 1.22106,  'min': 1.0508,   'max': 1.3980,   'zp': 3.190256e-10, 'zp_photon': 1.961698e+02, 'toVega':0,      'ext': 0.31,   'system': 'DENIS' },
              "DENIS_Ks":{ 'eff': 2.14650,  'min': 1.9474,   'max': 2.3979,   'zp': 4.341393e-11, 'zp_photon': 4.691482e+01, 'toVega':0,      'ext': 0.13,   'system': 'DENIS' },              

              "WISE_W1":      { 'eff': 3.3526,   'min': 2.754097, 'max': 3.872388, 'zp': 8.1787e-12,   'zp_photon': 1.375073e+01, 'toVega':0,      'ext': 0.07,   'system': 'WISE' }, # eff and ZP from Jarrett et al. (2011)
              "WISE_W2":      { 'eff': 4.6028,   'min': 3.963326, 'max': 5.341360, 'zp': 2.4150e-12,   'zp_photon': 5.586982e+00, 'toVega':0,      'ext': 0.05,   'system': 'WISE' }, # eff and ZP from Jarrett et al. (2011)
              "WISE_W3":      { 'eff': 11.5608,  'min': 7.443044, 'max': 17.26134, 'zp': 6.5151e-14,   'zp_photon': 3.567555e-01, 'toVega':0,      'ext': 0.06,   'system': 'WISE' }, # eff and ZP from Jarrett et al. (2011)
              "WISE_W4":      { 'eff': 22.0883,  'min': 19.52008, 'max': 27.91072, 'zp': 5.0901e-15,   'zp_photon': 5.510352e-02, 'toVega':0,      'ext': 0.02,   'system': 'WISE' }, # eff and ZP from Jarrett et al. (2011)

              "IRAC_ch1":   { 'eff': 3.507511, 'min': 3.129624, 'max': 3.961436, 'zp': 6.755364e-12, 'zp_photon': 1.192810e+01, 'toVega':0,      'ext': 0.07,   'system': 'IRAC' },
              "IRAC_ch2":   { 'eff': 4.436578, 'min': 3.917328, 'max': 5.056057, 'zp': 2.726866e-12, 'zp_photon': 6.090264e+00, 'toVega':0,      'ext': 0.05,   'system': 'IRAC' },
              "IRAC_ch3":   { 'eff': 5.628102, 'min': 4.898277, 'max': 6.508894, 'zp': 1.077512e-12, 'zp_photon': 3.052866e+00, 'toVega':0,      'ext': 0.04,   'system': 'IRAC' },
              "IRAC_ch4":     { 'eff': 7.589159, 'min': 6.299378, 'max': 9.587595, 'zp': 3.227052e-13, 'zp_photon': 1.232887e+00, 'toVega':0,      'ext': 0.03,   'system': 'IRAC' },              
              "MIPS_ch1":    { 'eff': 23.20960, 'min': 19.88899, 'max': 30.93838, 'zp': 3.935507e-15, 'zp_photon': 4.598249e-02, 'toVega':0,      'ext': 0.02,   'system': 'MIPS' },

              "Gaia_G":       { 'eff': 0.60,     'min': 0.321,    'max': 1.103,    'zp': 2.862966e-09, 'zp_photon': 8.053711e+02, 'toVega':0,      'ext': 0,      'system': 'Gaia'},
              "Gaia_BP":      { 'eff': 0.55,     'min': 0.321,    'max': 0.680,    'zp': 4.298062e-09, 'zp_photon': 1.067265e+03, 'toVega':0,      'ext': 0,      'system': 'Gaia'},
              "Gaia_RP":      { 'eff': 0.75,     'min': 0.627,    'max': 1.103,    'zp': 1.294828e-09, 'zp_photon': 4.948727e+02, 'toVega':0,      'ext': 0,      'system': 'Gaia'},

              "HST_F090M":   { 'eff': 0.897360, 'min': 0.784317, 'max': 1.013298, 'zp': 8.395228e-10, 'zp_photon': 3.792477e+02, 'toVega':0,      'ext': 0.51,   'system': 'HST' },
              "HST_F110W":   { 'eff': 1.059175, 'min': 0.782629, 'max': 1.432821, 'zp': 4.726040e-10, 'zp_photon': 2.519911e+02, 'toVega':0,      'ext': 0.39,   'system': 'HST' },
              "HST_F140W":   { 'eff': 1.364531, 'min': 1.185379, 'max': 1.612909, 'zp': 2.133088e-10, 'zp_photon': 1.465263e+02, 'toVega':0,      'ext': 0.26,   'system': 'HST' },
              "HST_F164N":   { 'eff': 1.646180, 'min': 1.629711, 'max': 1.663056, 'zp': 1.109648e-10, 'zp_photon': 9.195720e+01, 'toVega':0,      'ext': 0.19,   'system': 'HST' },
              "HST_F170M":   { 'eff': 1.699943, 'min': 1.579941, 'max': 1.837134, 'zp': 1.015711e-10, 'zp_photon': 8.692163e+01, 'toVega':0,      'ext': 0.18,   'system': 'HST' },
              "HST_F190N":   { 'eff': 1.898486, 'min': 1.880845, 'max': 1.917673, 'zp': 6.957714e-11, 'zp_photon': 6.649628e+01, 'toVega':0,      'ext': 0.15,   'system': 'HST' },
              "HST_F215N":   { 'eff': 2.148530, 'min': 2.128579, 'max': 2.168078, 'zp': 4.355167e-11, 'zp_photon': 4.710529e+01, 'toVega':0,      'ext': 0.13,   'system': 'HST' },
              "HST_F336W":   { 'eff': 0.332930, 'min': 0.295648, 'max': 0.379031, 'zp': 3.251259e-09, 'zp_photon': 5.486427e+02, 'toVega':0,      'ext': 1.70,   'system': 'HST' },
              "HST_F390N":   { 'eff': 0.388799, 'min': 0.384000, 'max': 0.393600, 'zp': 5.673647e-09, 'zp_photon': 1.143901e+03, 'toVega':0,      'ext': 1.48,   'system': 'HST' },
              "HST_F475W":   { 'eff': 0.470819, 'min': 0.386334, 'max': 0.556272, 'zp': 5.331041e-09, 'zp_photon': 1.260475e+03, 'toVega':0,      'ext': 1.21,   'system': 'HST' },
              "HST_F555W":   { 'eff': 0.533091, 'min': 0.458402, 'max': 0.620850, 'zp': 4.062007e-09, 'zp_photon': 1.061011e+03, 'toVega':0,      'ext': 1.05,   'system': 'HST' },
              "HST_F625W":   { 'eff': 0.626619, 'min': 0.544589, 'max': 0.709961, 'zp': 2.478260e-09, 'zp_photon': 7.679998e+02, 'toVega':0,      'ext': 0.68,   'system': 'HST' },
              "HST_F656N":   { 'eff': 0.656368, 'min': 0.653838, 'max': 0.658740, 'zp': 1.434529e-09, 'zp_photon': 4.737886e+02, 'toVega':0,      'ext': 0.81,   'system': 'HST' },
              "HST_F673N":   { 'eff': 0.673224, 'min': 0.667780, 'max': 0.678367, 'zp': 1.908442e-09, 'zp_photon': 6.499706e+02, 'toVega':0,      'ext': 0.78,   'system': 'HST' },
              "HST_F775W":   { 'eff': 0.765263, 'min': 0.680365, 'max': 0.863185, 'zp': 1.323662e-09, 'zp_photon': 5.055354e+02, 'toVega':0,      'ext': 0.65,   'system': 'HST' },
              "HST_F850LP":  { 'eff': 0.963736, 'min': 0.832000, 'max': 1.100000, 'zp': 8.069014e-10, 'zp_photon': 3.706372e+02, 'toVega':0,      'ext': 0.46,   'system': 'HST' }}    
  
  if isinstance(band,list):
    for i in Filters.keys(): 
      if Filters[i]['system'] not in band: Filters.pop(i)
    return Filters
  elif isinstance(band,str):
    return Filters[band]
 
  
def find(filename, tree):
  """                                                                               
  For given filename and directory tree, returns the path to the file. 
  For only file extension given as filename, returns list of paths to all files with that extnsion in that directory tree.  

  *filename*
    Filename or file extension to search for (e.g. 'my_file.txt' or just '.txt')
  *tree*
    Directory tree base to start the walk (e.g. '/Users/Joe/Documents/')
  """
  import os
  result = []

  for root, dirs, files in os.walk(tree):
    if filename.startswith('.'):
      for f in files:
        if f.endswith(filename):
          result.append(os.path.join(root, f))
    else:  
      if filename in files:
        result.append(os.path.join(root, filename))

  return result

def flux_calibrate(mag, dist, sig_m='', sig_d='', scale_to=10*q.pc):
  if isinstance(mag,(float,int)): return [round(mag-5*np.log10((dist.to(q.pc)/scale_to.to(q.pc)).value),3), round(np.sqrt(sig_m**2 + 25*(sig_d.to(q.pc)/(dist.to(q.pc)*np.log(10))).value**2),3) if sig_m and sig_d else '']
  elif hasattr(mag,'unit'): return [float('{:.4g}'.format(mag.value*(dist/scale_to).value**2))*mag.unit, float('{:.4g}'.format(np.sqrt((sig_m*(dist/scale_to).value)**2 + (2*mag*(sig_d*dist/scale_to**2).value)**2)))*mag.unit if sig_m!='' and sig_d else '']
  else: print 'Could not flux calibrate that input to distance {}.'.format(dist)

def flux2mag(band, f, sig_f='', photon=False, filter_dict=''): 
  """
  For given band and flux returns the magnitude value (and uncertainty if *sig_f*)
  """
  filt = filter_dict[band]
  if f.unit=='Jy': f, sig_f = (ac.c*f/filt['eff']**2).to(q.erg/q.s/q.cm**2/q.AA), (ac.c*sig_f/filt['eff']**2).to(q.erg/q.s/q.cm**2/q.AA)
  if photon: f, sig_f = (f*(filt['eff']/(ac.h*ac.c)).to(1/q.erg)).to(1/q.s/q.cm**2/q.AA), (sig_f*(filt['eff']/(ac.h*ac.c)).to(1/q.erg)).to(1/q.s/q.cm**2/q.AA)
  m = -2.5*np.log10((f/filt['zp_photon' if photon else 'zp']).value)
  sig_m = (2.5/np.log(10))*(sig_f/f).value if sig_f else ''  
  return [m,sig_m]

def get_filters(filter_directories=['./SEDkit/Data/Filters/{}/'.format(i) for i in ['2MASS','SDSS','WISE','IRAC','MIPS','FourStar','HST','Johnson','Cousins','MKO','GALEX','DENIS','Gaia','DES']], systems=['2MASS','SDSS','WISE','IRAC','MIPS','FourStar','HST','Johnson','Cousins','MKO','GALEX','DENIS','Gaia','DES']):
  """
  Grabs all the .txt spectral response curves and returns a dictionary of wavelength array [um], filter response [unitless], effective, min and max wavelengths [um], and zeropoint [erg s-1 cm-2 A-1]. 
  """
  files = glob.glob(filter_directories+'*.txt') if isinstance(filter_directories, basestring) else [j for k in [glob.glob(i+'*.txt') for i in filter_directories] for j in k]

  if len(files) == 0: print 'No filters in', filter_directories
  else:
    filters = {}
    for filepath in files:
      try:
        filter_name = os.path.splitext(os.path.basename(filepath))[0]
        RSR_x, RSR_y = np.genfromtxt(filepath, unpack=True, comments='#')
        RSR_x, RSR_y = (RSR_x*(q.um if min(RSR_x)<100 else q.AA)).to(q.um), RSR_y*q.um/q.um
        Filt = filter_info(filter_name)
        filters[filter_name] = {'wav':RSR_x, 'rsr':RSR_y, 'system':Filt['system'], 'eff':Filt['eff']*q.um, 'min':Filt['min']*q.um, 'max':Filt['max']*q.um, 'ext':Filt['ext'], 'toVega':Filt['toVega'], 'zp':Filt['zp']*q.erg/q.s/q.cm**2/q.AA, 'zp_photon':Filt['zp_photon']/q.s/q.cm**2/q.AA }
      except: pass
    for i in filters.keys():
      if filters[i]['system'] not in systems: filters.pop(i)    
    return filters

def goodness(spec1, spec2, array=False, exclude=[], filt_dict=None, weighting=True, verbose=False):
  if isinstance(spec1,dict) and isinstance(spec2,dict) and filt_dict:
    bands, w1, f1, e1, f2, e2, weight, bnds = [i for i in filt_dict.keys() if all([i in spec1.keys(),i in spec2.keys()]) and i not in exclude], [], [], [], [], [], [], []
    for eff,b in sorted([(filt_dict[i]['eff'],i) for i in bands]):
      if all([spec1[b],spec1[b+'_unc'],spec2[b],spec2[b+'_unc']]): bnds.append(b), w1.append(eff.value), f1.append(spec1[b].value), e1.append(spec1[b+'_unc'].value), f2.append(spec2[b].value), e2.append(spec2[b+'_unc'].value if b+'_unc' in spec2.keys() else 0), weight.append((filt_dict[b]['max']-filt_dict[b]['min']).value if weighting else 1)
    bands, w1, f1, e1, f2, e2, weight = map(np.array, [bnds, w1, f1, e1, f2, e2, weight])
    if verbose: printer(['Band','W_spec1','F_spec1','E_spec1','F_spec2','E_spec2','Weight','g-factor'],zip(*[bnds, w1, f1, e1, f2, e2, weight, weight*(f1-f2*(sum(weight*f1*f2/(e1**2 + e2**2))/sum(weight*f2**2/(e1**2 + e2**2))))**2/(e1**2 + e2**2)]))
  else:
    spec1, spec2 = [[i.value if hasattr(i,'unit') else i for i in j] for j in [spec1,spec2]]
    if exclude: spec1 = [i[idx_exclude(spec1[0],exclude)] for i in spec1]
    (w1, f1, e1), (f2, e2), weight = spec1, rebin_spec(spec2, spec1[0])[1:], np.gradient(spec1[0])
    if exclude: weight[weight>np.std(weight)] = 0
  C = sum(weight*f1*f2/(e1**2 + e2**2))/sum(weight*f2**2/(e1**2 + e2**2))
  G = weight*(f1-f2*C)**2/(e1**2 + e2**2)
  if verbose: plt.figure(), plt.loglog(w1, f1, 'k', label='spec1', alpha=0.6), plt.loglog(w1, f2*C, 'b', label='spec2 binned', alpha=0.6), plt.grid(True), plt.legend(loc=0)
  return [G if array else sum(G), C]

def group(lst, n):
  for i in range(0, len(lst), n):
    val = lst[i:i+n]
    if len(val) == n: yield tuple(val)

def group_spectra(spectra):
  """
  Puts a list of *spectra* into groups with overlapping wavelength arrays
  """
  groups, idx, i = [], [], 'wavelength' if isinstance(spectra[0],dict) else 0
  for N,S in enumerate(spectra):
    if N not in idx:
      group, idx = [S], idx+[N]
      for n,s in enumerate(spectra):
        if n not in idx and any(np.where(np.logical_and(S[i]<s[i][-1],S[i]>s[i][0]))[0]): group.append(s), idx.append(n)
      groups.append(group)
  return groups

def idx_include(x, include):
  try: return np.where(np.array(map(bool,map(sum, zip(*[np.logical_and(x>i[0],x<i[1]) for i in include])))))[0]
  except TypeError:
    try: return np.where(np.array(map(bool,map(sum, zip(*[np.logical_and(x>i[0],x<i[1]) for i in [include]])))))[0] 
    except TypeError: return range(len(x))

def idx_exclude(x, exclude):
  try: return np.where(~np.array(map(bool,map(sum, zip(*[np.logical_and(x>i[0],x<i[1]) for i in exclude])))))[0]
  except TypeError: 
    try: return np.where(~np.array(map(bool,map(sum, zip(*[np.logical_and(x>i[0],x<i[1]) for i in exclude])))))[0]
    except TypeError: return range(len(x))

def inject_average(spectrum, position, direction, n=10):
  """
  Used to smooth edges after trimming a spectrum. Injects a new data point into a *spectrum* at given *position* with flux value equal to the average of the *n* elements in the given *direction*.
  """
  units, spectrum, rows = [i.unit if hasattr(i,'unit') else 1 for i in spectrum], [i.value if hasattr(i,'unit') else i for i in spectrum], zip(*[i.value if hasattr(i,'unit') else i for i in spectrum])
  new_pos = [position, np.interp(position, spectrum[0], spectrum[1]), np.interp(position, spectrum[0], spectrum[2])]
  rows = sorted(map(list,rows)+[new_pos])
  sample = [np.array(i) for i in zip(*rows[rows.index(new_pos)-(n if direction=='left' else 0) : rows.index(new_pos)+(n if direction=='right' else 0)])]
  final_pos = [position, np.average(sample[1], weights=1/sample[2]), np.sqrt(sum(sample[2])**2)]
  rows[rows.index(new_pos)] = final_pos
  spectrum = zip(*rows)
  return [i*j for i,j in zip(units,spectrum)]

def Jy2mag(band, jy, jy_unc, filter_dict=''): return flux2mag(band, (ac.c*jy/filter_dict[band]['eff']**2).to(q.erg/q.s/q.cm**2/q.AA), sig_f=(ac.c*jy_unc/filter_dict[band]['eff']**2).to(q.erg/q.s/q.cm**2/q.AA), photon=False, filter_dict=filter_dict)

def mag2flux(band, mag, sig_m='', photon=False, filter_dict=''):
  """
  For given band and magnitude returns the flux value (and uncertainty if *sig_m*) in [ergs][s-1][cm-2][A-1]
  """
  if band.startswith('M_'): band = band[2:]
  filt = filter_dict[band]
  f = (filt['zp_photon' if photon else 'zp']*10**(-mag/2.5)).to((1 if photon else q.erg)/q.s/q.cm**2/q.AA)
  sig_f = f*sig_m*np.log(10)/2.5 if sig_m else ''
  return [f, sig_f]

def make_composite(spectra):
  """
  Creates a composite spectrum from a list of overlapping spectra
  """
  units = [i.unit for i in spectra[0]]
  spectrum = spectra.pop(0)
  if spectra:
    spectra = [norm_spec(spec, spectrum) for spec in spectra]
    spectrum = [i.value for i in spectrum]
    for n,spec in enumerate(spectra):
      spec = [i.value for i in spec]
      IDX, idx = np.where(np.logical_and(spectrum[0]<spec[0][-1],spectrum[0]>spec[0][0]))[0], np.where(np.logical_and(spec[0]>spectrum[0][0],spec[0]<spectrum[0][-1]))[0]
      low_res, high_res = [i[IDX] for i in spectrum], rebin_spec([i[idx] for i in spec], spectrum[0][IDX])
      mean_spec = [spectrum[0][IDX], np.array([np.average([hf,lf], weights=[1/he,1/le]) for hf,he,lf,le in zip(high_res[1],high_res[2],low_res[1],low_res[2])]), np.sqrt(low_res[2]**2 + high_res[2]**2)]
      spec1, spec2 = sorted([spectrum,spec], key=lambda x: x[0][0])
      spec1, spec2 = [i[np.where(spec1[0]<spectrum[0][IDX][0])[0]] for i in spec1], [i[np.where(spec2[0]>spectrum[0][IDX][-1])[0]] for i in spec2]
      spectrum = [np.concatenate([i[:-1],j[1:-1],k[1:]]) for i,j,k in zip(spec1,mean_spec,spec2)]
  return [i*Q for i,Q in zip([i.value if hasattr(i,'unit') else i for i in spectrum],units)]

def manual_legend(labels, colors, markers='', edges='', sizes='', errors='', styles='', text_colors='', fontsize=14, overplot='', bbox_to_anchor='', loc=0, ncol=1, figlegend=False):
  """
  Add manually created legends to plots and subplots
  
  Parameters
  ----------
  labels: sequence
    A list of strings to appear as legend text, e.g. ['Foo','Bar','Baz']
  colors: sequence
    A list of colors for the legend markers, e.g. ['r','g','b']
  markers: sequence (optional)
    A list of markers or linestyles to use in the legend, e.g. ['o','^','--'], defaults to 'o'
  edges: sequence (optional)
    A list of colors to use as marker edge colors, e.g. ['m','None','k'], defaults to *colors*
  sizes: sequence (optional)
    A list of integers to specify the marker size of points or the linewidth of lines, e.g. [8,12,2], defaults to 10
  errors: sequence (optional)
    A list of boolean statements to indicate whether markers should display error bars of not, e.g. [True,False,False], defaults to False
  styles: sequence (optional)
    A list indicating whether each legend item should display a point 'p' or a line 'l', e.g. ['p','p','l'], defaults to 'p'
  text_colors: sequence (optional)
    A list of colors for each legend label, defaults to 'k'
  overplot: axes object (optional)
    The axes to draw the legend on, defaults to the active axes
  fontsize: int
    The fontsize of the legend text
  loc: int
    The 0-8 integer location of the legend
  ncol: int
    The integer number of columns to divide the legend markers into
  bbox_to_anchor: sequence (optional)
    The legend bbox_to_anchor parametrs to place it outside the axes
  figlegend: bool
    Plot as the plt.figlegend instead of an axes legend
  """
  ax = overplot or plt.gca()
  handles = [plt.errorbar((1,0), (0,0), xerr=[0,0] if r else None, yerr=[0,0] if r else None, marker=m if t=='p' else '', color=c, ls=m if t=='l' else 'none', lw=s if t=='l' else 2, markersize=s, markerfacecolor=c, markeredgecolor=e, markeredgewidth=2, capsize=0, ecolor=e) for m,c,e,s,r,t in zip(markers or ['o' for i in colors], colors, edges or colors, sizes or [10 for i in colors], errors or [False for i in colors], styles or ['p' for i in colors])]
  [i[0].remove() for i in handles]
  if figlegend:
    plt.figlegend(handles, labels, figlegend, frameon=False, numpoints=1, handletextpad=1 if 'l' in styles else 0, fontsize=fontsize, handleheight=2, handlelength=1.5, ncol=ncol)
  else:
    add_legend = ax.legend(handles, labels, loc=loc, frameon=False, numpoints=1, handletextpad=1 if 'l' in styles else 0, handleheight=2, handlelength=1.5, fontsize=fontsize, ncol=ncol, bbox_to_anchor=bbox_to_anchor, mode="expand", borderaxespad=0.) if bbox_to_anchor else ax.legend(handles, labels, loc=loc, frameon=False, numpoints=1, handletextpad=1 if 'l' in styles else 0, handleheight=2, handlelength=1.5, fontsize=fontsize, ncol=ncol)
    ax.add_artist(add_legend)
    
  if text_colors:
    ltext = plt.gca().get_legend().get_texts()
    for n,(t,c) in enumerate(zip(ltext,text_colors)): plt.setp(ltext[n], color=c) 

def multiplot(rows, columns, ylabel='', xlabel='', xlabelpad='', ylabelpad='', hspace=0, wspace=0, figsize=(15,7), fontsize=22, sharey=True, sharex=True):
  """
  Creates subplots with given number or *rows* and *columns*.
  """
  fig, axes = plt.subplots(rows, columns, sharey=sharey, sharex=sharex, figsize=figsize)
  plt.rc('text', usetex=True, fontsize=fontsize)
  if ylabel:
    if isinstance(ylabel,str): fig.text(0.04, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=fontsize)
    else:
      if columns>1: axes[0].set_ylabel(ylabel, fontsize=fontsize, labelpad=ylabelpad or fontsize)
      else:
        for a,l in zip(axes,ylabel): a.set_xlabel(l, fontsize=fontsize, labelpad=xlabelpad or fontsize)
  if xlabel:
    if isinstance(xlabel,str): fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=fontsize)
    else:
      if rows>1: axes[0].set_ylabel(ylabel, fontsize=fontsize, labelpad=ylabelpad or fontsize)
      else:
        for a,l in zip(axes,xlabel): a.set_xlabel(l, fontsize=fontsize, labelpad=xlabelpad or fontsize)
  plt.subplots_adjust(right=0.96, top=0.96, bottom=0.15, left=0.12, hspace=hspace, wspace=wspace), fig.canvas.draw()
  return [fig]+list(axes)
  
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
  template, spectrum, spectrum_units = np.array([np.asarray(i.value) for i in template]), np.array([np.asarray(i.value) for i in spectrum]), [i.unit for i in spectrum]
  normed_spectrum = spectrum.copy()
  
  # Smooth both spectrum and template
  template[1], spectrum[1] = [smooth(x,1) for x in [template[1],spectrum[1]]]
  
  # Find wavelength range of overlap for array masking
  spec_mask = np.logical_and(spectrum[0]>template[0][0],spectrum[0]<template[0][-1])
  temp_mask = np.logical_and(template[0]>spectrum[0][0],template[0]<spectrum[0][-1])
  spectrum, template = [i[spec_mask] for i in spectrum], [i[temp_mask] for i in template]
  
  # Also mask arrays in wavelength ranges specified in *exclude*
  for r in exclude: 
    spec_mask = np.logical_and(spectrum[0]>r[0],spectrum[0]<r[-1])
    temp_mask = np.logical_and(template[0]>r[0],template[0]<r[-1])
    spectrum, template = [i[~spec_mask] for i in spectrum], [i[~temp_mask] for i in template]
    
  # Normalize the spectrum to the template based on equal integrated flux inincluded wavelength ranges
  norm = np.trapz(template[1], x=template[0])/np.trapz(spectrum[1], x=spectrum[0])
  normed_spectrum[1:] = [i*norm for i in normed_spectrum[1:]]
 
  return [i*Q for i,Q in zip(normed_spectrum,spectrum_units)]

def norm_spec_fmin(spectrum, template, exclude=[], weighting=True, plot=False):
  """
  Normalizes a spectrum to a template in wavelength range of overlap, excluding any specified wavelength ranges using function minimization of the goodness-of-fit statistic.

  """
  template, spectrum, spectrum_units = [i.value for i in template], [i.value for i in spectrum], [i.unit for i in spectrum]
  normed_spectrum = spectrum
  
  if plot: plt.loglog(spectrum[0], spectrum[1], color='r')
  
  for x in exclude: normed_spectrum = [i[~np.logical_and(spectrum[0]>x[0],spectrum[0]<x[1])] for i in normed_spectrum]
  
  def errfunc(p, spec1, spec2, weighting=weighting): 
    (w1, f1, e1), (f2, e2), weight = spec1, rebin_spec(spec2, spec1[0])[1:], np.gradient(spec1[0]) if weighting else np.ones(len(spec1[0]))
    if exclude: weight[weight>np.std(weight)] = 0
    return sum(weight*f1*f2/(e1**2 + e2**2))/sum(weight*f2**2/(e1**2 + e2**2))
  
  norm = opt.fmin(errfunc, template[1][0]/normed_spectrum[1][0], args=(template, normed_spectrum), xtol=0.000000001, ftol=0.000000001, maxfun=1000)[0]
  spectrum[1:] = [i*norm for i in spectrum[1:]]
  
  if plot: 
    plt.loglog(spectrum[0], spectrum[1], color='k')
    plt.loglog(template[0], template[1], color='k')
    plt.fill_between(spectrum[0], spectrum[1]-spectrum[2], spectrum[1]+spectrum[2], color='k', alpha=0.1)
    plt.fill_between(template[0], template[1]-template[2], template[1]+template[2], color='k', alpha=0.1)
  return [i*Q for i,Q in zip(spectrum,spectrum_units)]

def norm_to_mag(spectrum, magnitude, band): 
  """
  Returns the flux of a given *spectrum* [W,F] normalized to the given *magnitude* in the specified photometric *band*
  """
  return [spectrum[0],spectrum[1]*magnitude/s.get_mag(band, spectrum, to_flux=True, Flam=False)[0],spectrum[2]]
  
def normalize(spectra, template, composite=True, plot=False, SNR=50, exclude=[], trim=[], replace=[], D_Flam=None):
  """
  Normalizes a list of *spectra* with [W,F,E] or [W,F] to a *template* spectrum.
  Returns one normalized, composite spectrum if *composite*, else returns the list of *spectra* normalized to the *template*.
  """
  if not template: 
    spectra = [scrub(i) for i in sorted(spectra, key=lambda x: x[1][-1])]
    template = spectra.pop()
        
  if trim:
    all_spec = [template]+spectra
    for n,x1,x2 in trim: all_spec[n] = [i[idx_exclude(all_spec[n][0],[(x1,x2)])] for i in all_spec[n]]
    template, spectra = all_spec[0], all_spec[1:] if len(all_spec)>1 else None
  
  (W, F, E), normalized = scrub(template), []
  if spectra:
    for S in spectra: normalized.append(norm_spec(S, [W,F,E], exclude=exclude+replace))
    if plot: 
      plt.loglog(W, F, alpha=0.5), plt.fill_between(W, F-E, F+E, alpha=0.1)
      for w,f,e in normalized: plt.loglog(w, f, alpha=0.5), plt.fill_between(w, f-e, f+e, alpha=0.2)
    
    if composite:
      for n,(w,f,e) in enumerate(normalized):
        tries = 0
        while tries<5:
          IDX, idx = np.where(np.logical_and(W<w[-1],W>w[0]))[0], np.where(np.logical_and(w>W[0],w<W[-1]))[0]
          if not any(IDX):
            normalized.pop(n), normalized.append([w,f,e])
            tries += 1
          else:
            if len(IDX)<=len(idx): (W0, F0, E0), (w0, f0, e0) = [i[IDX]*q.Unit('') for i in [W,F,E]], [i[idx]*q.Unit('') for i in [w,f,e]]
            else: (W0, F0, E0), (w0, f0, e0) = [i[idx]*q.Unit('') for i in [w,f,e]], [i[IDX]*q.Unit('') for i in [W,F,E]]
            f0, e0 = rebin_spec([w0,f0,e0], W0)[1:]
            f_mean, e_mean = np.array([np.average([fl,FL], weights=[1/er,1/ER]) for fl,er,FL,ER in zip(f0,e0,F0,E0)]), np.sqrt(e0**2 + E0**2)            
            spec1, spec2 = min([W,F,E], [w,f,e], key=lambda x: x[0][0]), max([W,F,E], [w,f,e], key=lambda x: x[0][-1])
            spec1, spec2 = [i[np.where(spec1[0]<W0[0])[0]] for i in spec1], [i[np.where(spec2[0]>W0[-1])[0]] for i in spec2]
            W, F, E = [np.concatenate([i[:-1],j[1:-1],k[1:]]) for i,j,k in zip(spec1,[W0,f_mean,e_mean],spec2)]
            tries = 5
            normalized.pop(n)

      if replace: W, F, E = modelReplace([W,F,E], replace=replace, D_Flam=D_Flam)

    if plot:
      if composite: plt.loglog(W, F, '--', c='k', lw=1), plt.fill_between(W, F-E, F+E, color='k', alpha=0.2)
      plt.yscale('log', nonposy='clip')

    if not composite: normalized.insert(0, template)
    else: normalized = [[W,F,E]]
    return normalized[0][:len(template)] if composite else normalized
  else: return [W,F,E]

def output_polynomial(n, m, sig='', x='x', y='y', title='', degree=1, c='k', ls='--', lw=2, legend=True, ax='', output_data=False, plot_rms='0.9'):
  p, residuals, rank, singular_values, rcond = np.polyfit(np.array(map(float,n)), np.array(map(float,m)), degree, w=1/np.array([i if i else 1 for i in sig]) if sig!='' else None, full=True)
  f = np.poly1d(p)
  w = np.linspace(min(n), max(n), 50)
  ax.plot(w, f(w), color=c, ls=ls, lw=lw, label='${}$'.format(poly_print(p, x=x, y=y)) if legend else '', zorder=10)
  rms = np.sqrt(sum((m-f(n))**2)/len(n))
  if plot_rms: ax.fill_between(w, f(w)-rms, f(w)+rms, color=plot_rms, zorder=-1)
  data = [[y, (min(n),max(n)), rms]+list(reversed(p))]
  print_data = [[y, r'{:.1f}\textless {}\textless {:.1f}'.format(min(n),x,max(n)), '{:.3f}'.format(rms)]+['{:.3e}'.format(v) for v in list(reversed(p))]]
  printer(['P(x)','x','rms']+[r'$c_{}$'.format(str(i)) for i in range(len(p))], print_data, title=title, to_txt='./Files/{} v {}.txt'.format(x,y) if output_data else False)
  return data 

def pi2pc(parallax, parallax_unc=0, pc2pi=False):
  if parallax: 
    if pc2pi:
      return ((1*q.pc*q.arcsec)/parallax).to(q.mas), (parallax_unc*q.pc*q.arcsec/parallax**2).to(q.mas)
    else:
      pi, sig_pi = parallax*q.arcsec/1000., parallax_unc*q.arcsec/1000.
      d, sig_d = (1*q.pc*q.arcsec)/pi, sig_pi*q.pc*q.arcsec/pi**2
      return (d.round(2), sig_d.round(2))
  else: return ['','']

def polynomial(values, coeffs, plot=False, color='g', ls='-', lw=2):
  '''
  Evaluates *values* given the list of ascending polynomial *coeffs*.
  
  Parameters
  ----------
  values: int, list, tuple, array 
    The value or values to to evaluated
  coeffs: list, tuple, array
    The sequence of ascending polynomial coefficients beginning with zeroth order
  plot: bool (optional)
    Plot the results in the given color
  color: str
    The color of the line or fill color of the point to plot
  ls: str
    The linestyle of the line to draw
  lw: int
    The linewidth of the line to draw
  
  Returns
  -------
  out: float, list
    The evaluated results
         
  '''
  def poly_eval(val): return sum([c*(val**o) for o,c in enumerate(coeffs)])
  
  if isinstance(coeffs,dict): coeffs = [coeffs[j] for j in sorted([i for i in coeffs.keys() if i.startswith('c')])]
    
  if isinstance(values, (int,float)):
    out = poly_eval(values)
    if plot: plt.errorbar([values], [out], marker='*', markersize=18, color=color, markeredgecolor='k', markeredgewidth=2, zorder=10)

  elif isinstance(values, (tuple,list,np.ndarray)):
    out = [poly_eval(v) for v in values]
    if plot: plt.plot(values, out, color=color, lw=lw, ls=ls)
  
  else: out = None; print "Input values must be an integer, float, or sequence of integers or floats!"
  
  return out 

def poly_print(coeff_list, x='x', y='y'): return '{} ={}'.format(y,' '.join(['{}{:.3e}{}'.format(' + ' if i>0 else ' - ', abs(i), '{}{}'.format(x if n>0 else '', '^{}'.format(n) if n>1 else '')) for n,i in enumerate(coeff_list[::-1])][::-1]))

def printer(labels, values, format='', truncate=150, to_txt=None, highlight=[], skip=[], empties=True, title=False):
  """
  Prints a nice table of *values* with *labels* with auto widths else maximum width if *same* else *col_len* if specified. 
  """
  def red(t): print "\033[01;31m{0}\033[00m".format(t),
  # if not to_txt: print '\r'
  labels = list(labels)
  values = [["-" if i=='' or i is None else "{:.6g}".format(i) if isinstance(i,(float,int)) else i if isinstance(i,(str,unicode)) else "{:.6g} {}".format(i.value,i.unit) if hasattr(i,'unit') else i for i in j] for j in values]
  auto, txtFile = [max([len(i) for i in j])+2 for j in zip(labels,*values)], open(to_txt, 'a') if to_txt else None
  lengths = format if isinstance(format,list) else [min(truncate,i) for i in auto]
  col_len = [max(auto) for i in lengths] if format=='max' else [150/len(labels) for i in lengths] if format=='fill' else lengths
  
  # If False, remove columns with no values
  if not empties:
    for n,col in enumerate(labels):
      if all([i[n]=='-' for i in values]):
        labels.pop(n)
        for i in values: i.pop(n)
  
  if title:
    if to_txt: txtFile.write(str(title))
    else: print str(title)
  for N,(l,m) in enumerate(zip(labels,col_len)):
    if N not in skip:
      if to_txt: txtFile.write(str(l)[:truncate].ljust(m) if ' ' in str(l) else str(l)[:truncate].ljust(m))
      else: print str(l)[:truncate].ljust(m),  
  for row_num,v in enumerate(values):
    if to_txt: txtFile.write('\n')
    else: print '\n',
    for col_num,(k,j) in enumerate(zip(v,col_len)):
      if col_num not in skip:
        if to_txt: txtFile.write(str(k)[:truncate].ljust(j) if ' ' in str(k) else str(k)[:truncate].ljust(j))
        else:
          if (row_num,col_num) in highlight: red(str(k)[:truncate].ljust(j))
          else: print str(k)[:truncate].ljust(j),
  if not to_txt: print '\n'

def rebin_spec(spec, wavnew, waveunits='um'):
  from pysynphot import spectrum, observation
  # Gives same error answer: Err = np.array([np.sqrt(sum(spec[2].value[idx_include(wavnew,[((wavnew[0] if n==0 else wavnew[n-1]+wavnew[n])/2,wavnew[-1] if n==len(wavnew) else (wavnew[n]+wavnew[n+1])/2)])]**2)) for n in range(len(wavnew)-1)])*spec[2].unit if spec[2] is not '' else ''
  if len(spec)==2: spec += ['']
  try: Flx, Err, filt = spectrum.ArraySourceSpectrum(wave=spec[0].value, flux=spec[1].value), spectrum.ArraySourceSpectrum(wave=spec[0].value, flux=spec[2].value) if spec[2] else '', spectrum.ArraySpectralElement(spec[0].value, np.ones(len(spec[0])), waveunits=waveunits)
  except:
    spec, wavnew = [i*q.Unit('') for i in spec], wavnew*q.Unit('')
    Flx, Err, filt = spectrum.ArraySourceSpectrum(wave=spec[0].value, flux=spec[1].value), spectrum.ArraySourceSpectrum(wave=spec[0].value, flux=spec[2].value) if spec[2] else '', spectrum.ArraySpectralElement(spec[0].value, np.ones(len(spec[0])), waveunits=waveunits)
  return [wavnew, observation.Observation(Flx, filt, binset=wavnew.value, force='taper').binflux*spec[1].unit, observation.Observation(Err, filt, binset=wavnew.value, force='taper').binflux*spec[2].unit if spec[2] else np.ones(len(wavnew))*spec[1].unit]

def scrub(data):
  """
  For input data [w,f,e] or [w,f] returns the list with NaN, negative, and zero flux (and corresponsing wavelengths and errors) removed. 
  """
  units = [i.unit if hasattr(i,'unit') else 1 for i in data]
  data = [np.asarray(i.value if hasattr(i,'unit') else i, dtype=np.float32) for i in data if isinstance(i,np.ndarray)]
  data = [i[np.where(~np.isinf(data[1]))] for i in data]
  data = [i[np.where(np.logical_and(data[1]>0,~np.isnan(data[1])))] for i in data]
  data = [i[np.unique(data[0], return_index=True)[1]] for i in data]
  return [i[np.lexsort([data[0]])]*Q for i,Q in zip(data,units)]

def smooth(x,beta):
  """
  Smooths a spectrum *x* using a Kaiser-Bessel smoothing window of narrowness *beta* (~1 => very smooth, ~100 => not smooth) 
  """
  window_len = 11
  s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
  w = np.kaiser(window_len,beta)
  y = np.convolve(w/w.sum(), s, mode='valid')
  return y[5:len(y)-5]*(x.unit if hasattr(x, 'unit') else 1)
  
def specType(SpT):
  """
  (By Joe Filippazzo)

  Converts between float and letter/number M, L, T and Y spectral types (e.g. 14.5 => 'L4.5' and 'T3' => 23).

  *SpT*
    Float spectral type between 0.0 and 39.9 or letter/number spectral type between M0.0 and Y9.9
  """
  if isinstance(SpT,str) and SpT[0] in ['M','L','T','Y']:
    try: return [l+float(SpT[1:]) for m,l in zip(['M','L','T','Y'],[0,10,20,30]) if m == SpT[0]][0]
    except: print "Spectral type must be a float between 0 and 40 or a string of class M, L, T or Y."; return SpT
  elif isinstance(SpT,float) or isinstance(SpT,int) and 0.0 <= SpT < 40.0: 
    try: return '{}{}'.format('MLTY'[int(SpT//10)], int(SpT%10) if SpT%10==int(SpT%10) else SpT%10)
    except: print "Spectral type must be a float between 0 and 40 or a string of class M, L, T or Y."; return SpT
  else: return SpT
  
def str2Q(x,target=''):
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
      text = ['Jy', 'erg', '/s', 's-1', 's', '/um', 'um-1', 'um', '/cm2', 'cm-2', 'cm2', '/cm', 'cm-1', 'cm', '/A', 'A-1', 'A', 'W', '/m2', 'm-2', 'm2', '/m', 'm-1', 'm', '/Hz', 'Hz-1']
      vals = [q.Jy, q.erg, q.s**-1, q.s**-1, q.s, q.um**-1, q.um**-1, q.um, q.cm**-2, q.cm**-2, q.cm**2, q.cm**-1, q.cm**-1, q.cm, q.AA**-1, q.AA**-1, q.AA, q.W, q.m**-2, q.m**-2, q.m**2, q.m**-1, q.m**-1, q.m, q.Hz**-1, q.Hz**-1]
      for t,v in zip(text,vals):
        if t in IN:
          OUT = OUT*v
          IN = IN.replace(t,'')
      return OUT

    unit = Q(x)
    if target:
      z = str(Q(target)).split()[-1]
      try:
        unit = unit.to(z)
      except ValueError:
        print "{} could not be rescaled to {}".format(unit,z)

    return unit 
  else:
    return q.Unit('')
      
def squaredError(a, b, c):
  """
  Computes the squared error of two arrays. Pass to scipy.optimize.fmin() to find least square or use scipy.optimize.leastsq()
  """
  a -= b
  a *= a 
  c = np.array([1 if np.isnan(e) else e for e in c])
  return sum(a/c)

def trim_spectrum(spectrum, regions, smooth_edges=False):
  trimmed_spec = [i[idx_exclude(spectrum[0], regions)] for i in spectrum]
  if smooth_edges: 
    for r in regions:
      try:
        if any(spectrum[0][spectrum[0]>r[1]]): trimmed_spec = inject_average(trimmed_spec, r[1], 'right', n=smooth_edges)
      except: pass
      try: 
        if any(spectrum[0][spectrum[0]<r[0]]): trimmed_spec = inject_average(trimmed_spec, r[0], 'left', n=smooth_edges)
      except: pass 
  return trimmed_spec

def unc(spectrum, SNR=20):
  """
  Removes NaNs negatives and zeroes from *spectrum* arrays of form [W,F] or [W,F,E].
  Generates E at signal to noise *SNR* for [W,F] and replaces NaNs with the same for [W,F,E]. 
  """
  S = scrub(spectrum)
  if len(S)==3:
    try: S[2] = np.array([i/SNR if np.isnan(j) else j for i,j in zip(S[1],S[2])], dtype='float32')*(S[1].unit if hasattr(S[1],'unit') else 1)
    except: S[2] = np.array(S[1]/SNR)
  elif len(S)==2: S.append(S[1]/SNR)
  return S
