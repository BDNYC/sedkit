#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
Some utilities to accompany sedkit
"""
import glob
import itertools
import re
import warnings

import astropy.units as q
import astropy.constants as ac
import astropy.table as at
from astropy.modeling.blackbody import blackbody_lambda
from astropy.modeling import models
import bokeh.palettes as bpal
import numpy as np
import pandas as pd
import scipy.optimize as opt


warnings.simplefilter('ignore')


# A dict of BDNYCdb band names to work with sedkit
PHOT_ALIASES = {'2MASS_J': '2MASS.J', '2MASS_H': '2MASS.H',
                '2MASS_Ks': '2MASS.Ks', 'WISE_W1': 'WISE.W1',
                'WISE_W2': 'WISE.W2', 'WISE_W3': 'WISE.W3',
                'WISE_W4': 'WISE.W4', 'IRAC_ch1': 'IRAC.I1',
                'IRAC_ch2': 'IRAC.I2', 'IRAC_ch3': 'IRAC.I3',
                'IRAC_ch4': 'IRAC.I4', 'SDSS_u': 'SDSS.u',
                'SDSS_g': 'SDSS.g', 'SDSS_r': 'SDSS.r',
                'SDSS_i': 'SDSS.i', 'SDSS_z': 'SDSS.z',
                'MKO_J': 'NSFCam.J', 'MKO_Y': 'Wircam.Y',
                'MKO_H': 'NSFCam.H', 'MKO_K': 'NSFCam.K',
                "MKO_L'": 'NSFCam.Lp', "MKO_M'": 'NSFCam.Mp',
                'Johnson_V': 'Johnson.V', 'Cousins_R': 'Cousins.R',
                'Cousins_I': 'Cousins.I', 'FourStar_J': 'FourStar.J',
                'FourStar_J1': 'FourStar.J1', 'FourStar_J2': 'FourStar.J2',
                'FourStar_J3': 'FourStar.J3', 'HST_F125W': 'WFC3_IR.F125W'}


@models.custom_model
def blackbody(wavelength, temperature=2000):
    """
    Generate a blackbody of the given temperature at the given wavelengths

    Parameters
    ----------
    wavelength: array-like
        The wavelength array [um]
    temperature: float
        The temperature of the star [K]

    Returns
    -------
    astropy.quantity.Quantity
        The blackbody curve
    """
    wavelength = q.Quantity(wavelength, "um")
    temperature = q.Quantity(temperature, "K")
    max_val = blackbody_lambda((ac.b_wien/temperature).to(q.um), temperature).value

    return blackbody_lambda(wavelength, temperature).value/max_val


def color_gen(colormap='viridis', key=None, n=10):
    """Color generator for Bokeh plots

    Parameters
    ----------
    colormap: str, sequence
        The name of the color map

    Returns
    -------
    generator
        A generator for the color palette
    """
    if colormap in dir(bpal):
        palette = getattr(bpal, colormap)

        if isinstance(palette, dict):
            if key is None:
                key = list(palette.keys())[0]
            palette = palette[key]

        elif callable(palette):
            palette = palette(n)

        else:
            raise TypeError("pallette must be a bokeh palette name or a sequence of color hex values.")

    elif isinstance(colormap, (list, tuple)):
        palette = colormap

    else:
        raise TypeError("pallette must be a bokeh palette name or a sequence of color hex values.")

    yield from itertools.cycle(palette)


COLORS = color_gen('Category10')


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
    return s.replace('.', '').replace('-', '').replace('+', '').isnumeric()


def filter_table(table, **kwargs):
    """Retrieve the filtered rows

    Parameters
    ----------
    table: astropy.table.Table, pandas.DataFrame
        The table to filter
    param: str
        The parameter to filter by, e.g. 'Teff'
    value: str, float, int, sequence
        The criteria to filter by, 
        which can be single valued like 1400
        or a range with operators [<, <=, >, >=], 
        e.g. ('>1200', '<=1400')

    Returns
    -------
    astropy.table.Table, pandas.DataFrame
        The filtered table
    """
    pandas = False
    if isinstance(table, pd.DataFrame):
        pandas = True
        table = at.Table.from_pandas(table)

    for param, value in kwargs.items():

        # Check it is a valid column
        if param not in table.colnames:
            raise KeyError("No column named {}".format(param))

        # Wildcard case
        if isinstance(value, str) and '*' in value:

            # Get column data
            data = np.array(table[param])

            if not value.startswith('*'):
                value = '^'+value
            if not value.endswith('*'):
                value = value+'$'

            # Strip souble quotes
            value = value.replace("'", '').replace('"', '').replace('*', '(.*)')

            # Regex
            reg = re.compile(value, re.IGNORECASE)
            keep = list(filter(reg.findall, data))

            # Get indexes
            idx = np.where([i in keep for i in data])

            # Filter table
            table = table[idx]

        else:

            # Make single value string into conditions
            if isinstance(value, str):

                # Check for operator
                if any([value.startswith(o) for o in ['<', '>', '=']]):
                    value = [value]

                # Assume eqality if no operator
                else:
                    value = ['== '+value]

            # Turn numbers into strings
            if isinstance(value, (int, float)) or (isinstance(value, str) and isnumber(value)):
                value = ["== {}".format(value)]

            # Iterate through multiple conditions
            for cond in value:

                # Equality
                if cond.startswith('='):
                    v = cond.replace('=', '')
                    table = table[table[param] == eval(v)]

                # Less than or equal
                elif cond.startswith('<='):
                    v = cond.replace('<=', '')
                    table = table[table[param]<=eval(v)]

                # Less than
                elif cond.startswith('<'):
                    v = cond.replace('<', '')
                    table = table[table[param]<eval(v)]

                # Greater than or equal
                elif cond.startswith('>='):
                    v = cond.replace('>=', '')
                    table = table[table[param]>=eval(v)]

                # Greater than
                elif cond.startswith('>'):
                    v = cond.replace('>', '')
                    table = table[table[param]>eval(v)]

                else:
                    raise ValueError("'{}' operator not understood.".format(cond))

    if pandas:
        table = table.to_pandas()

    return table


def finalize_spec(spec, wave_units=q.um, flux_units=q.erg/q.s/q.cm**2/q.AA):
    """
    Sort by wavelength and remove nans, negatives and zeroes

    Parameters
    ----------
    spec: sequence
        The [W, F, E] to be cleaned up

    Returns
    -------
    spec: sequence
        The cleaned and ordered [W, F, E]
    """
    spec = list(zip(*sorted(zip(*map(list, [[i.value if hasattr(i, 'unit') else i for i in j] for j in spec])), key=lambda x: x[0])))
    return scrub([spec[0]*wave_units, spec[1]*flux_units, spec[2]*flux_units])


def flux_calibrate(mag, dist, sig_m=None, sig_d=None, scale_to=10*q.pc):
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

            # Mag = mag - 2.5*np.log10(dist/scale_to)**2
            Mag = mag - 5*np.log10(dist.value) + 5*np.log10(scale_to.value)
            Mag = Mag.round(3)

            if isinstance(sig_d, q.quantity.Quantity) and sig_m is not None: 
                Mag_unc = np.sqrt(sig_m**2 + (2.5*sig_d/(np.log(10)*dist))**2)
                Mag_unc = Mag_unc.round(3).value

            else:
                Mag_unc = np.nan

            return [Mag, Mag_unc]

        else:

            print('Could not flux calibrate that input to distance {}.'.format(dist))
            return [np.nan, np.nan]

    except IOError:

        print('Could not flux calibrate that input to distance {}.'.format(dist))
        return [np.nan, np.nan]


def flux2mag(flx, bandpass):
    """Calculate the magnitude for a given flux

    Parameters
    ----------
    flx: astropy.units.quantity.Quantity, sequence
        The flux or (flux, uncertainty)
    bandpass: pysynphot.spectrum.ArraySpectralElement
        The bandpass to use
    """
    if isinstance(flx, (q.core.PrefixUnit, q.core.Unit, q.core.CompositeUnit)):
        flx = flx, np.nan*flx.unit

    # Calculate the magnitude
    eff = bandpass.wave_eff
    zp = bandpass.zp
    flx, unc = flx
    unit = flx.unit

    # Convert energy units to photon counts
    flx = (flx*(eff/(ac.h*ac.c)).to(1/q.erg)).to(unit/q.erg)
    zp = (zp*(eff/(ac.h*ac.c)).to(1/q.erg)).to(unit/q.erg)
    unc = (unc*(eff/(ac.h*ac.c)).to(1/q.erg)).to(unit/q.erg)

    # Calculate magnitude
    m = -2.5*np.log10((flx/zp).value)
    m_unc = (2.5/np.log(10))*(unc/flx).value

    return m, m_unc


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


def errorbar(fig, x, y, xerr=None, yerr=None, color='black', point_kwargs={}, error_kwargs={}, legend=None):
    """
    Hack to make errorbar plots in bokeh

    Parameters
    ----------
    x: sequence
        The x axis data
    y: sequence
        The y axis data
    xerr: sequence (optional)
        The x axis errors
    yerr: sequence (optional)
        The y axis errors
    color: str
        The marker and error bar color
    point_kwargs: dict
        kwargs for the point styling
    error_kwargs: dict
        kwargs for the error bar styling
    legend: str
        The text for the legend
    """
    fig.circle(x, y, color=color, legend=legend, **point_kwargs)

    if xerr is not None:
        x_err_x = []
        x_err_y = []
        for px, py, err in zip(x, y, xerr):
            x_err_x.append((px - err, px + err))
            x_err_y.append((py, py))
        fig.multi_line(x_err_x, x_err_y, color=color, **error_kwargs)

    if yerr is not None:
        y_err_x = []
        y_err_y = []
        for px, py, err in zip(x, y, yerr):
            y_err_x.append((px, px))
            y_err_y.append((py - err, py + err))
        fig.multi_line(y_err_x, y_err_y, color=color, **error_kwargs)


def goodness(f1, f2, e1=None, e2=None, weights=None):
    """Calculate the goodness of fit statistic and normalization constant between two spectra

    Parameters
    ----------
    f1: sequence
        The flux of the first spectrum
    f2: sequence
        The flux of the second spectrum
    e1: sequence(optional)
        The uncertainty of the first spectrum
    e2: sequence (optional)
        The uncertainty of the second spectrum
    weights: sequence, float (optional)
        The weights of each point
    """
    if len(f1) != len(f2):
        raise ValueError("f1[{}] and f2[{}]. They must be the same length.".format(len(f1), len(f2)))

    # Fill in missing arrays
    if e1 is None:
        e1 = np.ones(len(f1))
    if e2 is None:
        e2 = np.ones(len(f2))
    if weights is None:
        weights = 1.

    # Calculate the goodness-of-fit statistic and normalization constant
    errsq = e1**2 + e2**2
    numerator = np.nansum(weights * f1 * f2 / errsq)
    denominator = np.nansum(weights * f2 ** 2 / errsq)
    norm = numerator/denominator
    gstat = np.nansum(weights*(f1-f2*norm)**2/errsq)

    return gstat, norm


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


def idx_exclude(x, exclude):
    try:
        return np.where(~np.array(map(bool, map(sum, zip(*[np.logical_and(x > i[0], x < i[1]) for i in exclude])))))[0]
    except TypeError:
        try:
            return \
            np.where(~np.array(map(bool, map(sum, zip(*[np.logical_and(x > i[0], x < i[1]) for i in exclude])))))[0]
        except TypeError:
            return range(len(x))


def idx_include(x, include):
    try:
        return np.where(np.array(map(bool, map(sum, zip(*[np.logical_and(x > i[0], x < i[1]) for i in include])))))[0]
    except TypeError:
        try:
            return \
            np.where(np.array(map(bool, map(sum, zip(*[np.logical_and(x > i[0], x < i[1]) for i in [include]])))))[0]
        except TypeError:
            return range(len(x))


def idx_overlap(s1, s2):
    """Force s2 to be completely overlapped by s1

    Paramters
    ---------
    s1: sequence
        The first array
    s2: sequence
        The second array

    Returns
    -------
    np.ndarray
        The indexes of the trimmed second sequence
    """
    return np.where((s2 > s1[0]) & (s2 < s1[-1]))[0]


def interp_flux(flux, params, values):
    """
    Interpolate a cube of synthetic spectra for a
    given index of mu

    Parameters
    ----------
    mu: int
        The index of the (Teff, logg, FeH, *mu*, wavelength)
        data cube to interpolate
    flux: np.ndarray
        The data array
    params: list
        A list of each free parameter range
    values: list
        A list of each free parameter values

    Returns
    -------
    tu
        The array of new flux values
    """
    # Iterate over each wavelength (-1 index of flux array)
    shp = flux.shape[-1]
    flx = np.zeros(shp)
    generators = []
    for lam in range(shp):
        interp_f = RegularGridInterpolator(params, flux[:, :, :, lam])
        f, = interp_f(values)

        flx[lam] = f
        generators.append(interp_f)

    return flx, generators


def mag2flux(band, mag, sig_m='', units=q.erg/q.s/q.cm**2/q.AA):
    """
    Caluclate the flux for a given magnitude

    Parameters
    ----------
    band: sedkit.synphot.Bandpass
        The bandpass
    mag: float, astropy.unit.quantity.Quantity
        The magnitude
    sig_m: float, astropy.unit.quantity.Quantity
        The magnitude uncertainty
    units: astropy.unit.quantity.Quantity
        The unit for the output flux
    """
    try:
        # Make mag unitless
        if hasattr(mag, 'unit'):
            mag = mag.value
        if hasattr(sig_m, 'unit'):
            sig_m = sig_m.value

        # Calculate the flux density
        zp = band.zp
        f = (zp*10**(mag/-2.5)).to(units)

        if isinstance(sig_m, str):
            sig_m = np.nan

        sig_f = (f*sig_m*np.log(10)/2.5).to(units)

        return np.array([f.value, sig_f.value])*units

    except IOError:
        return np.array([np.nan, np.nan])*units


def pi2pc(dist, unc_lower=None, unc_upper=None, pi_unit=q.mas, dist_unit=q.pc, pc2pi=False):
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
    unit = pi_unit if pc2pi else dist_unit

    if unc_lower is None:
        unc_lower = 0*dist.unit
    if unc_upper is None:
        unc_upper = 0*dist.unit

    val = ((1*q.pc*q.arcsec)/dist).to(unit).round(2)
    low = (unc_lower*val/dist).to(unit).round(2)
    upp = (unc_upper*val/dist).to(unit).round(2)

    if unc_lower is not None and unc_upper is not None:
        return val, low, upp

    else:
        return val, low


def rebin_spec(wavnew, wave, flux, err=None, oversamp=100, plot=False):
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
    nlam = len(wave)
    x0 = np.arange(nlam, dtype=float)
    x0int = np.arange((nlam-1.)*oversamp + 1., dtype=float)/oversamp
    w0int = np.interp(x0int, x0, wave)
    spec0int = np.interp(w0int, wave, flux)/oversamp
    try:
        err0int = np.interp(w0int, wave, err)/oversamp
    except:
        err0int = ''

    # Set up the bin edges for down-binning
    maxdiffw1 = np.diff(wavnew).max()
    w1bins = np.concatenate(([wavnew[0]-maxdiffw1], 
                              .5*(wavnew[1::]+wavnew[0:-1]), 
                              [wavnew[-1]+maxdiffw1]))

    # Bin down the interpolated spectrum:
    w1bins = np.sort(w1bins)
    nbins = len(w1bins)-1
    specnew = np.zeros(nbins)
    errnew = np.zeros(nbins)
    inds2 = [[w0int.searchsorted(w1bins[ii], side='left'), 
              w0int.searchsorted(w1bins[ii+1], side='left')] 
              for ii in range(nbins)]

    for ii in range(nbins):
        specnew[ii] = np.sum(spec0int[inds2[ii][0]:inds2[ii][1]])
        if err is not None:
            errnew[ii] = np.sum(err0int[inds2[ii][0]:inds2[ii][1]])

    # Fix edges
    specnew[0] = flux[0]
    specnew[-1] = flux[-1]

    return [wavnew, specnew] if not any(errnew) else [wavnew, specnew, errnew]


def scrub(data):
    """
    For input data [w, f, e] or [w, f] returns the list with NaN, negative, and zero flux (and corresponsing wavelengths and errors) removed.
    """
    # Unit check
    units = [i.unit if hasattr(i, 'unit') else 1 for i in data]

    # Ensure floats
    data = [np.asarray(i.value if hasattr(i, 'unit') else i, dtype=np.float32) for i in data if isinstance(i, np.ndarray)]

    # Remove infinities
    data = [i[np.where(~np.isinf(data[1]))] for i in data]

    # Remove zeros and negatives
    data = [i[np.where(data[1] > 0)] for i in data]

    # Remove nans
    data = [i[np.where(~np.isnan(data[1]))] for i in data]

    # Remove duplicate wavelengths
    data = [i[np.unique(data[0], return_index=True)[1]] for i in data]

    # Ensure monotonic and return units
    data = [i[np.lexsort([data[0]])] * Q for i, Q in zip(data, units)]

    return data


def set_resolution(spec, resolution):
    """Rebin the spectrum to the given resolution

    Parameters
    ----------
    spec: sequence
        The spectrum to set the resolution for
    resolution: float, int
        The desired resolution

    Return
    ------
    sequence
        The new spectrum
    """
    # Make the wavelength array
    mn = np.nanmin(spec[0])
    mx = np.nanmax(spec[0])
    d_lam = (mx-mn)/resolution
    wave = np.arange(mn, mx, d_lam)

    # Trim the wavelength
    dmn = (spec[0][1]-spec[0][0])/2.
    dmx = (spec[0][-1]-spec[0][-2])/2.
    wave = wave[np.logical_and(wave >= mn+dmn, wave <= mx-dmx)]

    # Calculate the new spectrum
    spec = spectres(wave, spec[0], spec[1])


def spectres(new_spec_wavs, old_spec_wavs, spec_fluxes, spec_errs=None):
    """
    Function for resampling spectra (and optionally associated uncertainties)
    onto a new wavelength basis.

    Parameters
    ----------
    new_spec_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the spectrum
        or spectra.
    old_spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the spectrum or
        spectra.
    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        old_spec_wavs, last dimension must correspond to the shape of
        old_spec_wavs. Extra dimensions before this may be used to include 
        multiple spectra.
    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.

    Returns
    -------
    resampled_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same length as
        new_spec_wavs, other dimensions are the same as spec_fluxes
    resampled_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in resampled_fluxes. Only
        returned if spec_errs was specified.

    Reference
    ---------
    https://github.com/ACCarnall/SpectRes/blob/master/spectres/spectral_resampling.py
    """
    # Trim new_spec_wavs so they are completely covered by old_spec_wavs
    idx = idx_overlap(old_spec_wavs, new_spec_wavs)
    if not any(idx):
        raise ValueError("spectres: The new wavelengths specified must fall at\
                          least partially within the range of the old\
                          wavelength values.")
    new_spec_wavs = new_spec_wavs[idx]

    # Generate arrays of left hand side positions and widths for the old
    # and new bins
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_widths = np.zeros(old_spec_wavs.shape[0])
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_lhs[0] = old_spec_wavs[0] - (old_spec_wavs[1] - old_spec_wavs[0])/2
    spec_widths[-1] = (old_spec_wavs[-1] - old_spec_wavs[-2])
    spec_lhs[1:] = (old_spec_wavs[1:] + old_spec_wavs[:-1])/2
    spec_widths[:-1] = spec_lhs[1:] - spec_lhs[:-1]

    filter_lhs = np.zeros(new_spec_wavs.shape[0]+1)
    filter_widths = np.zeros(new_spec_wavs.shape[0])
    filter_lhs[0] = new_spec_wavs[0] - (new_spec_wavs[1] - new_spec_wavs[0])/2
    filter_widths[-1] = (new_spec_wavs[-1] - new_spec_wavs[-2])
    filter_lhs[-1] = new_spec_wavs[-1]+(new_spec_wavs[-1]-new_spec_wavs[-2])/2
    filter_lhs[1:-1] = (new_spec_wavs[1:] + new_spec_wavs[:-1])/2
    filter_widths[:-1] = filter_lhs[1:-1] - filter_lhs[:-2]

    # # Check that the range of wavelengths to be resampled_fluxes onto 
    # # falls within the initial sampling region
    # if filter_lhs[0] < spec_lhs[0] or filter_lhs[-1] > spec_lhs[-1]:
    #     raise ValueError("spectres: The new wavelengths specified must\
    # fall within the range of the old wavelength values.")

    #Generate output arrays to be populated
    resampled_fluxes = np.zeros(spec_fluxes[..., 0].shape + new_spec_wavs.shape)

    if spec_errs is not None:
        if spec_errs.shape != spec_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape\
                              as spec_fluxes.")
        else:
            resampled_fluxes_errs = np.copy(resampled_fluxes)

    start = 0
    stop = 0

    # Calculate the new spectral flux and uncertainty values, 
    # loop over the new bins
    for j in range(new_spec_wavs.shape[0]-1):

        # Find the first old bin which is partially covered by the new bin
        while spec_lhs[start+1] <= filter_lhs[j]:
            start += 1

        # Find the last old bin which is partially covered by the new bin
        while spec_lhs[stop+1] < filter_lhs[j+1]:
            stop += 1

        # If the new bin falls entirely within one old bin the are the same
        # the new flux and new error are the same as for that bin
        if stop == start:

            resampled_fluxes[..., j] = spec_fluxes[..., start]
            if spec_errs is not None:
                resampled_fluxes_errs[..., j] = spec_errs[..., start]

        # Otherwise multiply the first and last old bin widths by P_ij, 
        # all the ones in between have P_ij = 1
        else:

            start_factor = (spec_lhs[start+1] - filter_lhs[j])/(spec_lhs[start+1] - spec_lhs[start])
            end_factor = (filter_lhs[j+1] - spec_lhs[stop])/(spec_lhs[stop+1] - spec_lhs[stop])

            spec_widths[start] *= start_factor
            spec_widths[stop] *= end_factor

            # Populate the resampled_fluxes spectrum and uncertainty arrays
            resampled_fluxes[..., j] = np.sum(spec_widths[start:stop+1]*spec_fluxes[..., start:stop+1], axis=-1)/np.sum(spec_widths[start:stop+1])

            if spec_errs is not None:
                resampled_fluxes_errs[..., j] = np.sqrt(np.sum((spec_widths[start:stop+1]*spec_errs[..., start:stop+1])**2, axis=-1))/np.sum(spec_widths[start:stop+1])

            # Put back the old bin widths to their initial values for later use
            spec_widths[start] /= start_factor
            spec_widths[stop] /= end_factor

    # If errors were supplied return the resampled_fluxes spectrum and error arrays
    if spec_errs is None:
        return [new_spec_wavs, resampled_fluxes]

    else:
        return [new_spec_wavs, resampled_fluxes, resampled_fluxes_errs]


def specType(SpT, types=[i for i in 'OBAFGKMLTY'], verbose=False):
    """
    Converts between float and letter/number spectral types (e.g. 14.5 => 'B4.5' and 'A3' => 23).

    Parameters
    ----------
    SpT: float, str
        Float spectral type or letter/number spectral type between O0.0 and Y9.9
    types: list
        The MK spectral type letters to include, e.g. ['M', 'L', 'T', 'Y']

    Returns
    -------
    list, str
        The [spectral type, uncertainty, prefix, gravity, luminosity class] of the spectral type
    """
    try:
        # String input
        if isinstance(SpT, (str, bytes)):

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
                if ': :' in suf:
                    unc = 2
                    suf = suf.replace(': :', '')
                elif ': ' in suf:
                    unc = 1
                    suf = suf.replace(': ', '')
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
                suf = suf.replace(str(val), '').replace('n', '').replace('e', '')\
                         .replace('w', '').replace('m', '').replace('a', '')\
                         .replace('Fe', '').replace('-1', '').replace('?', '')\
                         .replace('-V', '').replace('p', '')

                # Check for luminosity class
                LC = []
                for cl in ['III', 'V', 'IV']:
                    if cl in suf:
                        LC.append(cl)
                        suf.replace(cl, '')
                LC = '/'.join(LC) or 'V'

                return [val, unc, pre, grv, LC]

            else:
                print('Not in list of MK spectral classes', types)
                return [np.nan, np.nan, '', '', '']

        # Numerical or list input
        elif isinstance(SpT, (float, int, list, tuple)):
            if isinstance(SpT, (int, float)):
                SpT = [SpT]

            # Get the MK class
            MK = ''.join(types)[int(SpT[0]//10)]
            num = int(SpT[0]%10) if SpT[0]%10 == int(SpT[0]%10) else SpT[0]%10

            # Get the uncertainty
            if len(SpT)>1:
                if SpT[1] == ': ' or SpT[1] == 1:
                    unc = ': '
                elif SpT[1] == ': :' or SpT[1] == 2:
                    unc = ': :'
                else:
                    unc = ''
            else:
                unc = ''

            # Get the prefix
            if len(SpT)>2 and SpT[2]:
                pre = str(SpT[2])
            else:
                pre = ''

            # Get the gravity
            if len(SpT)>3 and SpT[3]:
                grv = str(SpT[3])
            else:
                grv = ''

            # Get the luminosity class
            if len(SpT)>4 and SpT[4]:
                LC = str(SpT[4])
            else:
                LC = ''

            return ''.join([pre, MK, str(num), grv, LC, unc])

        # Bogus input
        else:
            if verbose:
                print('Spectral type', SpT, 'must be a float between 0 and', len(types)*10, 'or a string of class', types)
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
      The target units as a string if rescaling is necessary, e.g. str2Q('Wm-2um-1', target='erg/s/cm2/cm') => np.array(10000000.0) * erg/(cm**3*s)
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

def trim_spectrum(spectrum, regions=None, wave_min=0*q.um, wave_max=40*q.um, smooth_edges=False):
    regions = regions or []
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

    # Get indexes to keep
    trimmed_spec = [i[idx_exclude(trimmed_spec[0], [(wave_min, wave_max)])] for i in spectrum]

    return trimmed_spec
