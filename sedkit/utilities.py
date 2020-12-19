#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
# !python3
"""
Some utilities to accompany sedkit
"""

import copy
import itertools
import os
import re
import warnings

import astropy.constants as ac
from astropy.io import fits, ascii
from astropy.modeling.blackbody import blackbody_lambda
from astropy.modeling import models
import astropy.table as at
import astropy.units as q
import bokeh.palettes as bpal
import numpy as np
import pandas as pd
import scipy.optimize as opt


warnings.simplefilter('ignore')

# Valid dtypes for units
UNITS = q.core.PrefixUnit, q.core.Unit, q.core.CompositeUnit, q.quantity.Quantity, q.core.IrreducibleUnit

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
                'Johnson_U': 'Johnson.U', 'Johnson_B': 'Johnson.B',
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
    max_val = blackbody_lambda((ac.b_wien / temperature).to(q.um), temperature).value

    return blackbody_lambda(wavelength, temperature).value / max_val


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


def convert_mag(band, mag, mag_unc, old='AB', new='Vega'):
    """Convert between magnitude systems (Blanton et al. 2007)

    Parameters
    ----------
    band: str
        The bandpass name
    mag: float
        The magnitude
    mag_unc: float
        The magnitude uncertainty
    old: str
        The system the input magnitude is in, ['Vega', 'AB']
    new: str
        The target mag system, ['Vega', 'AB']
    """
    # TODO: Add other system conversions
    # TODO: Add other bandpasses
    AB_to_Vega = {'Johnson.U': 0.79, 'Johnson.B': -0.09, 'Johnson.V': 0.02, 'Cousins.R': 0.21, 'Cousins.I': 0.45,
                  '2MASS.J': 0.91, '2MASS.H': 1.39, '2MASS.Ks': 1.85,
                  'SDSS.u': 0.91, 'SDSS.g': -0.08, 'SDSS.r': 0.16, 'SDSS.i': 0.37, 'SDSS.z': 0.54}

    if old == 'AB' and new == 'Vega':
        corr = AB_to_Vega.get(band, 0)
        return mag - corr, mag_unc

    elif old == 'Vega' and new == 'AB':
        corr = AB_to_Vega.get(band, 0)
        return mag + corr, mag_unc

    else:
        return mag, mag_unc


def equivalent(value, units):
    """Function to test equivalence of value(s) to given units

    Parameters
    ----------
    value: astropy.units.quantity.Quantity, sequence
        The value to check
    units: astropy.units.core.PrefixUnit, astropy.units.core.Unit, astropy.units.core.CompositeUnit
        The units to test for equivalency

    Returns
    -------
    bool
        Equivalent or not
    """
    eq = True

    # Check if it's a list or tuple
    if isinstance(value, (tuple, list)):
        for val in value:
            try:
                val *= 1
                if isinstance(val, UNITS):
                    if not val.unit.is_equivalent(units):
                        eq = False
                else:
                    eq = False
            except TypeError:
                eq = False

    # Otherwise array or astropy quantity
    else:
        if isinstance(value, UNITS):
            value *= 1
            if not value.unit.is_equivalent(units):
                eq = False
        else:
            eq = False

    return eq


def isnumber(s):
    """
    Tests to see if the input is an int, float, or exponential

    Parameters
    ----------
    s: str, int, float
        The input to test

    Returns
    -------
    bool
        The boolean result
    """
    if isinstance(s, (str, bytes)):
        return s.replace('.', '').replace('-', '').replace(' + ', '').isnumeric()

    elif isinstance(s, (int, float, np.float32)):
        return True

    else:
        return False


def issequence(seq, length=None):
    """
    Tests to see if the given input is a sequence of a given length

    Parameters
    ----------
    seq: sequence
        The sequence to test
    length: int, sequence
        The desired length(s) of the sequence

    Returns
    -------
    bool
        The boolean result
    """
    # Acceptible dtypes
    typs = (tuple, list, np.ndarray)

    # Strip units if necessary
    if hasattr(seq, 'unit'):
        seq = seq.value

    # Check dtype
    isseq = False
    if isinstance(seq, typs):

        # Any length
        if length is None:
            isseq = True

        # Single length
        elif isinstance(length, int):
            if len(seq) == length:
                isseq = True

        # Multiple lengths
        elif isinstance(length, typs):
            if len(seq) in length:
                isseq = True

    return isseq


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
                value = '^' + value
            if not value.endswith('*'):
                value = value + '$'

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
                    value = ['== ' + value]

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
                    table = table[table[param] <= eval(v)]

                # Less than
                elif cond.startswith('<'):
                    v = cond.replace('<', '')
                    table = table[table[param] < eval(v)]

                # Greater than or equal
                elif cond.startswith('>='):
                    v = cond.replace('>=', '')
                    table = table[table[param] >= eval(v)]

                # Greater than
                elif cond.startswith('>'):
                    v = cond.replace('>', '')
                    table = table[table[param] > eval(v)]

                else:
                    raise ValueError("'{}' operator not understood.".format(cond))

    if pandas:
        table = table.to_pandas()

    return table


def finalize_spec(spec, wave_units=q.um, flux_units=q.erg / q.s / q.cm**2 / q.AA):
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
    return scrub([spec[0] * wave_units, spec[1] * flux_units, spec[2] * flux_units])


def flux_calibrate(mag, dist, sig_m=None, sig_d=None, scale_to=10 * q.pc):
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
            Mag = mag - 5 * np.log10(dist.value) + 5 * np.log10(scale_to.value)
            Mag = Mag.round(3)

            if isinstance(sig_d, q.quantity.Quantity) and sig_m is not None:
                Mag_unc = np.sqrt(sig_m**2 + (2.5 * sig_d / (np.log(10) * dist))**2)
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
    if isinstance(flx, UNITS):
        flx = flx, np.nan * flx.unit

    # Calculate the magnitude
    eff = bandpass.wave_eff
    zp = bandpass.zp
    flx, unc = flx
    unit = flx.unit

    # Set uncertainty
    unc = unc or np.nan * unit

    # Convert energy units to photon counts
    flx = (flx * (eff / (ac.h * ac.c)).to(1 / q.erg)).to(unit / q.erg)
    zp = (zp * (eff / (ac.h * ac.c)).to(1 / q.erg)).to(unit / q.erg)
    unc = (unc * (eff / (ac.h * ac.c)).to(1 / q.erg)).to(unit / q.erg)

    # Calculate magnitude
    m = -2.5 * np.log10((flx / zp).value)
    m_unc = (2.5 / np.log(10)) * (unc / flx).value

    return m, m_unc


def fnu2flam(f_nu, lam, units=q.erg / q.s / q.cm**2 / q.AA):
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

    f_lam = (f_nu * ac.c / lam**2).to(units)

    return f_lam


def minimize_norm(arr1, arr2, **kwargs):
    """Minimize the function to find the normalization factor that best
    aligns arr2 with arr1

    Parameters
    ----------
    arr1: np.ndarray
        The first array
    arr2: np.ndarray
        The second array

    Returns
    -------
    float
        The normalization constant
    """
    def errfunc(p, a1, a2):
        return np.nansum(abs(a1 - (a2 * p)))

    # Initial guess
    p0 = np.nanmean(arr1) / np.nanmean(arr2)
    norm_factor = opt.fmin(errfunc, p0, args=(arr1, arr2), disp=0, **kwargs)

    return norm_factor


def errorbars(fig, x, y, xerr=None, xupper=None, xlower=None, yerr=None, yupper=None, ylower=None, color='red', point_kwargs={}, error_kwargs={}):
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
    # Add x errorbars if possible
    if xerr is not None or (xupper is not None and xlower is not None):
        x_err_x = []
        x_err_y = []

        # Symmetric uncertainties
        if xerr is not None:
            for px, py, err in zip(x, y, xerr):
                try:
                    x_err_x.append((px - err, px + err))
                    x_err_y.append((py, py))
                except TypeError:
                    pass

        # Asymmetric uncertainties
        elif xupper is not None and xlower is not None:
            for px, py, lower, upper in zip(x, y, xlower, xupper):
                try:
                    x_err_x.append((px - lower, px + upper))
                    x_err_y.append((py, py))
                except TypeError:
                    pass

        fig.multi_line(x_err_x, x_err_y, color=color, **error_kwargs)

    # Add y errorbars if possible
    if yerr is not None or (yupper is not None and ylower is not None):
        y_err_x = []
        y_err_y = []

        # Symmetric uncertainties
        if yerr is not None:
            for px, py, err in zip(x, y, yerr):
                try:
                    y_err_y.append((py - err, py + err))
                    y_err_x.append((px, px))
                except TypeError:
                    pass

        # Asymmetric uncertainties
        elif yupper is not None and ylower is not None:
            for px, py, lower, upper in zip(x, y, ylower, yupper):
                try:
                    y_err_y.append((py - lower, py + upper))
                    y_err_x.append((px, px))
                except TypeError:
                    pass

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
    norm = numerator / denominator
    gstat = np.nansum(weights * (f1 - f2 * norm)**2 / errsq)

    return gstat, norm


def group_spectra(spectra):
    """
    Put a list of *spectra* into groups with overlapping wavelength arrays
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
    """
    Return the indexes of the array that exclude the given ranges

    Parameters
    ----------
    x: array-like
        The array to slice
    exclude: sequence
        The list of ranges to exclude

    Returns
    -------
    np.ndarray
        The indexes that satisfy the criteria
    """
    try:
        return np.where(~np.array(list(map(bool, list(map(sum, zip(*[np.logical_and(x > i[0], x < i[1]) for i in exclude])))))))[0]

    except TypeError:
        try:
            return np.where(~np.logical_and(x > exclude[0], x < exclude[1]))[0]
        except TypeError:
            return np.array(range(len(x)))


def idx_include(x, include):
    """
    Return the indexes of the array that include the given ranges

    Parameters
    ----------
    x: array-like
        The array to slice
    include: sequence
        The list of ranges to include

    Returns
    -------
    np.ndarray
        The indexes that satisfy the criteria
    """
    try:
        return np.where(np.array(list(map(bool, list(map(sum, zip(*[np.logical_and(x > i[0], x < i[1]) for i in include])))))))[0]
    except TypeError:
        try:
            return np.where(np.logical_and(x > include[0], x < include[1]))[0]
        except TypeError:
            return np.array(range(len(x)))


def idx_overlap(s1, s2, inclusive=False):
    """
    Return the indices of s2 that overlap with s1

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
    if inclusive:
        return np.where((s2 >= s1[0]) & (s2 <= s1[-1]))[0]
    else:
        return np.where((s2 > s1[0]) & (s2 < s1[-1]))[0]


def mag2flux(band, mag, sig_m='', units=q.erg / q.s / q.cm**2 / q.AA):
    """
    Caluclate the flux for a given magnitude

    Parameters
    ----------
    band: svo_filters.svo.Filter
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
        f = (band.zp * 10**(mag / -2.5)).to(units)

        if isinstance(sig_m, str):
            sig_m = np.nan

        sig_f = (f * sig_m * np.log(10) / 2.5).to(units)

        return np.array([f.value, sig_f.value]) * units

    except IOError:
        return np.array([np.nan, np.nan]) * units


def monotonic(x):
    """
    Determine if a sequence is monotonic

    Parameters
    ----------
    x: array-like
        The array to test

    Returns
    -------
    bool
        Result of boolean test
    """
    # Get diff of array
    dx = np.diff(x)

    return np.all(dx <= 0) or np.all(dx >= 0)


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
        unc_lower = 0 * dist.unit
    if unc_upper is None:
        unc_upper = 0 * dist.unit

    val = ((1 * q.pc * q.arcsec) / dist).to(unit).round(2)
    low = (unc_lower * val / dist).to(unit).round(2)
    upp = (unc_upper * val / dist).to(unit).round(2)

    if unc_lower is not None and unc_upper is not None:
        return val, low, upp

    else:
        return val, low


def scrub(raw_data, fill_value=None):
    """
    For input data [w, f, e] or [w, f] returns the list with negative, and
    zero flux and corresponsing wavelengths and errors removed (default), or
    converted to fill_value
    """
    # Make a copy
    data = copy.copy(raw_data)

    # Unit check
    units = [i.unit if hasattr(i, 'unit') else 1 for i in data]

    # Ensure floats
    data = [np.asarray(i.value if hasattr(i, 'unit') else i, dtype=np.float32) for i in data if isinstance(i, np.ndarray)]

    # Change infinities to nan
    data[1][np.where(np.isinf(data[1]))] = np.nan

    # Change zeros and negatives to nan
    data[1][np.where(data[1] <= 0)] = np.nan

    # Remove nans or replace with fill_value
    if fill_value is None:
        data = [i[np.where(~np.isnan(data[1]))] for i in data]
    else:
        if not isinstance(fill_value, (int, float)):
            raise ValueError("Please use float or int for fill_value")
        data[1][np.where(np.isnan(data[1]))] = fill_value

    # Remove duplicate wavelengths
    data = [i[np.unique(data[0], return_index=True)[1]] for i in data]

    # Ensure monotonic and return units
    data = [i[np.lexsort([data[0]])] * Q for i, Q in zip(data, units)]

    return data


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
        raise ValueError("spectres: The new wavelengths specified must fall at least partially within the range of the old wavelength values.")
    spec_wavs = new_spec_wavs[idx]

    # Generate arrays of left hand side positions and widths for the old
    # and new bins
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_widths = np.zeros(old_spec_wavs.shape[0])
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_lhs[0] = old_spec_wavs[0] - (old_spec_wavs[1] - old_spec_wavs[0]) / 2
    spec_widths[-1] = (old_spec_wavs[-1] - old_spec_wavs[-2])
    spec_lhs[1:] = (old_spec_wavs[1:] + old_spec_wavs[:-1]) / 2
    spec_widths[:-1] = spec_lhs[1:] - spec_lhs[:-1]

    filter_lhs = np.zeros(spec_wavs.shape[0] + 1)
    filter_widths = np.zeros(spec_wavs.shape[0])
    filter_lhs[0] = spec_wavs[0] - (spec_wavs[1] - spec_wavs[0]) / 2
    filter_widths[-1] = (spec_wavs[-1] - spec_wavs[-2])
    filter_lhs[-1] = spec_wavs[-1] + (spec_wavs[-1] - spec_wavs[-2]) / 2
    filter_lhs[1:-1] = (spec_wavs[1:] + spec_wavs[:-1]) / 2
    filter_widths[:-1] = filter_lhs[1:-1] - filter_lhs[:-2]

    # Generate output arrays to be populated
    resampled_fluxes = np.zeros(spec_fluxes[..., 0].shape + spec_wavs.shape)
    resampled_fluxes_errs = np.zeros_like(resampled_fluxes)

    if spec_errs is not None:
        if spec_errs.shape != spec_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape\
                              as spec_fluxes.")

    start = 0
    stop = 0

    # Calculate the new spectral flux and uncertainty values,
    # loop over the new bins
    for j in range(spec_wavs.size):

        try:

            # Find the first old bin which is partially covered by the new bin
            while spec_lhs[start + 1] <= filter_lhs[j]:
                start += 1

            # Find the last old bin which is partially covered by the new bin
            while spec_lhs[stop + 1] < filter_lhs[j + 1]:
                stop += 1

            # If the new bin falls entirely within one old bin they are the same
            # the new flux and new error are the same as for that bin
            if stop == start:

                resampled_fluxes[..., j] = spec_fluxes[..., start]
                if spec_errs is not None:
                    resampled_fluxes_errs[..., j] = spec_errs[..., start]

            # Otherwise multiply the first and last old bin widths by P_ij,
            # all the ones in between have P_ij = 1
            else:

                start_factor = (spec_lhs[start + 1] - filter_lhs[j]) / (spec_lhs[start + 1] - spec_lhs[start])
                end_factor = (filter_lhs[j + 1] - spec_lhs[stop]) / (spec_lhs[stop + 1] - spec_lhs[stop])

                spec_widths[start] *= start_factor
                spec_widths[stop] *= end_factor

                # Populate the resampled_fluxes spectrum and uncertainty arrays
                resampled_fluxes[..., j] = np.sum(spec_widths[start:stop + 1] * spec_fluxes[..., start:stop + 1], axis=-1) / np.sum(spec_widths[start:stop + 1])

                if spec_errs is not None:
                    resampled_fluxes_errs[..., j] = np.sqrt(np.sum((spec_widths[start:stop + 1] * spec_errs[..., start:stop + 1])**2, axis=-1)) / np.sum(spec_widths[start:stop + 1])

                # Put back the old bin widths to their initial values for later use
                spec_widths[start] /= start_factor
                spec_widths[stop] /= end_factor

        except IndexError:
            resampled_fluxes[..., j] = np.nan
            if spec_errs is not None:
                resampled_fluxes_errs[..., j] = np.nan

    # Interpolate results onto original wavelength basis
    resampled_fluxes = np.interp(new_spec_wavs, spec_wavs, resampled_fluxes, left=np.nan, right=np.nan)

    # If errors were supplied return the resampled_fluxes spectrum and
    # error arrays
    if spec_errs is None:
        return [new_spec_wavs, resampled_fluxes]

    else:
        resampled_fluxes_errs = np.interp(new_spec_wavs, spec_wavs, resampled_fluxes_errs, left=np.nan, right=np.nan)
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
                val += types.index(MK) * 10

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
            MK = ''.join(types)[int(SpT[0] // 10)]
            num = int(SpT[0] % 10) if SpT[0] % 10 == int(SpT[0] % 10) else SpT[0] % 10

            # Get the uncertainty
            if len(SpT) > 1:
                if SpT[1] == ': ' or SpT[1] == 1:
                    unc = ': '
                elif SpT[1] == ': :' or SpT[1] == 2:
                    unc = ': :'
                else:
                    unc = ''
            else:
                unc = ''

            # Get the prefix
            if len(SpT) > 2 and SpT[2]:
                pre = str(SpT[2])
            else:
                pre = ''

            # Get the gravity
            if len(SpT) > 3 and SpT[3]:
                grv = str(SpT[3])
            else:
                grv = ''

            # Get the luminosity class
            if len(SpT) > 4 and SpT[4]:
                LC = str(SpT[4])
            else:
                LC = ''

            return ''.join([pre, MK, str(num), grv, LC, unc])

        # Bogus input
        else:
            if verbose:
                print('Spectral type', SpT, 'must be a float between 0 and', len(types) * 10, 'or a string of class', types)
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


def spectrum_from_fits(File, ext=0, verbose=False):
    """
    Converts a SPECTRUM data type stored in the database into a (W,F,E) sequence of arrays.

    Parameters
    ----------
    File: str
        The URL or filepath of the file to be converted into arrays.
    verbose: bool
        Whether or not to display some diagnostic information (Default: False)

    Returns
    -------
    sequence
        The converted spectrum.

    """
    spectrum, header = '', ''
    if isinstance(File, type(b'')):  # Decode if needed (ie, for Python 3)
        File = File.decode('utf-8')

    if isinstance(File, (str, type(u''))):

        # Convert variable path to absolute path
        if File.startswith('$'):
            abspath = os.popen('echo {}'.format(File.split('/')[0])).read()[:-1]
            if abspath:
                File = File.replace(File.split('/')[0], abspath)

        if File.startswith('http'):
            if verbose:
                print('Downloading {}'.format(File))
            downloaded_file = download_file(File, cache=True)  # download only once
        else:
            downloaded_file = File

        # Try FITS files first
        try:

            # Get the data
            spectrum, header = fits.getdata(downloaded_file, cache=True, header=True, ext=ext)

            # Check the key type
            # KEY_TYPE = ['CTYPE1']
            # setType = set(KEY_TYPE).intersection(set(header.keys()))
            # if len(setType) == 0:
            #     isLinear = True
            # if len(setType) != 0:
            #     valType = header[setType.pop()]
            #     isLinear = valType.strip().upper() == 'LINEAR'

            # Get wl, flux & error data from fits file
            spectrum = __get_spec(spectrum, header, File)

            # Generate wl axis when needed
            if not isinstance(spectrum[0], np.ndarray):
                tempwav = __create_waxis(header, len(spectrum[1]), File)

                # Check to see if it's a FIRE spectrum with CDELT1, if so needs wlog=True
                if 'INSTRUME' in header.keys():
                    if header['INSTRUME'].strip() == 'FIRE' and 'CDELT1' in header.keys():
                        tempwav = __create_waxis(header, len(spectrum[1]), File, wlog=True)

                spectrum[0] = tempwav

            # If no wl axis generated, then clear out all retrieved data for object
            if not isinstance(spectrum[0], np.ndarray):
                spectrum = None

            if verbose:
                print('Read as FITS...')

        except (IOError, KeyError):

            # Check if the FITS file is just Numpy arrays
            try:
                spectrum, header = fits.getdata(downloaded_file, cache=True, header=True, ext=ext)
                if verbose:
                    print('Read as FITS Numpy array...')

            except (IOError, KeyError):

                try:  # Try ascii
                    spectrum = ascii.read(downloaded_file)
                    spectrum = np.array([np.asarray(spectrum.columns[n]) for n in range(len(spectrum.columns))])
                    if verbose:
                        print('Read as ascii...')

                    txt, header = open(downloaded_file), []
                    for i in txt:
                        if any([i.startswith(char) for char in ['#', '|', '\\']]):
                            header.append(i.replace('\n', ''))
                    txt.close()

                except (IOError, KeyError, TypeError):
                    pass

    if spectrum == '':
        print('Could not retrieve spectrum at {}.'.format(File))
        return File
    else:
        return spectrum


def __create_waxis(fitsHeader, lenData, fileName, wlog=False, verb=True):
    """
    Create wavelength array from fits header information
    """
    # Define key names in
    KEY_MIN = ['COEFF0', 'CRVAL1']  # Min wl
    KEY_DELT = ['COEFF1', 'CDELT1', 'CD1_1']  # Delta of wl
    KEY_OFF = ['LTV1']  # Offset in wl to subsection start

    # Find key names for minimum wl, delta, and wl offset in fits header
    setMin = set(KEY_MIN).intersection(set(fitsHeader.keys()))
    setDelt = set(KEY_DELT).intersection(set(fitsHeader.keys()))
    setOff = set(KEY_OFF).intersection(set(fitsHeader.keys()))

    # Get the values for minimum wl, delta, and wl offset, and generate axis
    if len(setMin) >= 1 and len(setDelt) >= 1:
        nameMin = setMin.pop()
        valMin = fitsHeader[nameMin]

        nameDelt = setDelt.pop()
        valDelt = fitsHeader[nameDelt]

        if len(setOff) == 0:
            valOff = 0
        else:
            nameOff = setOff.pop()
            valOff = fitsHeader[nameOff]

        # generate wl axis
        if nameMin == 'COEFF0' or wlog is True:
            # SDSS fits files
            wAxis = 10 ** (np.arange(lenData) * valDelt + valMin)
        else:
            wAxis = (np.arange(lenData) * valDelt) + valMin - (valOff * valDelt)

    else:
        wAxis = None
        if verb:
            print('Could not re-create wavelength axis for ' + fileName + '.')

    return wAxis


def __get_spec(fitsData, fitsHeader, fileName, verb=True):
    """
    Generate a spectrum from fits file data
    """
    validData = [None] * 3

    # Identify number of data sets in fits file
    dimNum = len(fitsData)

    # Identify data sets in fits file
    fluxIdx = None
    waveIdx = None
    sigmaIdx = None

    if dimNum == 1:
        fluxIdx = 0
    elif dimNum == 2:
        if len(fitsData[0]) == 1:
            sampleData = fitsData[0][0][20]
        else:
            sampleData = fitsData[0][20]
        if sampleData < 0.0001:
            # 0-flux, 1-unknown
            fluxIdx = 0
        else:
            waveIdx = 0
            fluxIdx = 1
    elif dimNum == 3:
        waveIdx = 0
        fluxIdx = 1
        sigmaIdx = 2
    elif dimNum == 4:
        # 0-flux clean, 1-flux raw, 2-background, 3-sigma clean
        fluxIdx = 0
        sigmaIdx = 3
    elif dimNum == 5:
        # 0-flux, 1-continuum substracted flux, 2-sigma, 3-mask array, 4-unknown
        fluxIdx = 0
        sigmaIdx = 2
    elif dimNum > 10:
        # Implies that only one data set in fits file: flux
        fluxIdx = -1
        if np.isscalar(fitsData[0]):
            fluxIdx = -1
        elif len(fitsData[0]) == 2:
            # Data comes in a xxxx by 2 matrix (ascii origin)
            tmpWave = []
            tmpFlux = []
            for pair in fitsData:
                tmpWave.append(pair[0])
                tmpFlux.append(pair[1])
            fitsData = [tmpWave, tmpFlux]
            fitsData = np.array(fitsData)

            waveIdx = 0
            fluxIdx = 1
        else:
            # Indicates that data is structured in an unrecognized way
            fluxIdx = None
    else:
        fluxIdx = None

    # Fetch wave data set from fits file
    if fluxIdx is None:
        # No interpretation known for fits file data sets
        validData = None
        if verb:
            print('Unable to interpret data in ' + fileName + '.')
        return validData
    else:
        if waveIdx is not None:
            if len(fitsData[waveIdx]) == 1:
                # Data set may be a 1-item list
                validData[0] = fitsData[waveIdx][0]
            else:
                validData[0] = fitsData[waveIdx]

    # Fetch flux data set from fits file
    if fluxIdx == -1:
        validData[1] = fitsData
    else:
        if len(fitsData[fluxIdx]) == 1:
            validData[1] = fitsData[fluxIdx][0]
        else:
            validData[1] = fitsData[fluxIdx]

    # Fetch sigma data set from fits file, if requested
    if sigmaIdx is None:
        validData[2] = np.array([np.nan] * len(validData[1]))
    else:
        if len(fitsData[sigmaIdx]) == 1:
            validData[2] = fitsData[sigmaIdx][0]
        else:
            validData[2] = fitsData[sigmaIdx]

    # If all sigma values have the same value, replace them with nans
    if validData[2][10] == validData[2][11] == validData[2][12]:
        validData[2] = np.array([np.nan] * len(validData[1]))

    return validData
