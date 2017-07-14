#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jcfilippazzo@gmail.com
#!python2
from __future__ import absolute_import, division, print_function, unicode_literals

from itertools import chain, groupby
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib import cm
from astropy.coordinates.angles import Angle
import astropy.units as q, astropy.constants as ac, astropy.io.ascii as ascii, astropy.table as at
import sys, os, copy, pickle, re, pandas as pd, matplotlib.pyplot as plt, numpy as np, scipy.stats as st, \
    scipy.interpolate as si, matplotlib.ticker
import SEDkit.utilities as u
import SEDkit.syn_phot as s
import SEDkit.interact as interact
from SEDkit import mcmc_fit
import datetime

RSR = u.get_filters()
package = os.path.dirname(u.__file__)


def unitless(data, dtypes=''): return np.array(
    [np.asarray([i.value if hasattr(i, 'unit') else i for i in j], dtype=d) for j, d in
     zip(data, dtypes or ['string'] * len(data))])


def DMEstar(ages, xparam='Lbol', yparam='radius'):
    """
  Retrieve the DMEstar isochrones

  Parameters
  ----------
  ages: sequence
    The ages at which the isochrones should be evaluated
  xparam: str

  """
    from glob import glob
    D, data = [], glob(package + '/Data/Models/Evolutionary/DMESTAR/isochrones/*.txt')
    for f in data:
        age = int(os.path.basename(f).split('_')[1][:-5]) / 1000.
        if age in ages:
            mass, teff, Lbol, logg, radius = np.genfromtxt(f, usecols=(1, 3, 4, 2, 5), unpack=True)
            teff, radius, mass = 10 ** teff, 9.72847 * 10 ** radius, 1047.2 * mass
            D.append([age,
                      mass if xparam == 'mass' else logg if xparam == 'logg' else radius if xparam == 'radius' else Lbol,
                      mass if yparam == 'mass' else logg if yparam == 'logg' else radius if yparam == 'radius' else Lbol])
    return D


def mag_mag_relations(band, dictionary, consider, try_all=False, to_flux=True, mag_and_flux=False, data_table=''):
    """
  Returns the magnitude in the given *band* based on the distance and magnitudes of the object

  Parameters
  ----------
  band: str
    The desired magnitude to return. Use the band name to return the apparent magnitude (e.g. 'Ks') or M_band for the absolute magnitude (e.g. 'M_Ks')
  dictionary: str, dict
    The name of the object (e.g. 'HN Peg B') or a dictionary containing the object's distance 'd' and absolute magnitudes.
  consider: sequence
    A list of the bands to consider when estimating the new magnitude
  try_all: bool
    If none of the bands listed to *consider* produces a magnitude, try all bands
  to_flux: bool
    Return the magnitude as a flux value in units 'erg/s/cm2/A'
  mag_and_flux: bool
    Returns both the magnitude and flux
  data_table: dict (optional)
    If a name is given for *dictionary*, the full nested dictionary of objects must be supplied as *data_table*

  Returns
  -------
  The [magnitude,uncertainty,source] for the given *band* with the least uncertainty

  """
    try:

        pickle_path, pop = package + '/Data/Pickles/polynomial_relations.p', []

        # Allow name search instead of having to input the object dictionary
        if isinstance(dictionary, str) and data_table:
            if data_table.data.get(dictionary): D = data_table.data.get(dictionary)
        else:
            D = dictionary.copy()

        # Load the mag-mag pickle and pull out polynomials, determine if object is young
        Q = pickle.load(open(pickle_path, 'rb'))
        y = '_yng' if (D.get('gravity') or D.get('age_min') < 500) else '_fld'
        mags = []

        # Pull the estimated magnitudes from the 'fld' or 'yng' polynomials if the input magnitude is in range of the polynomial
        consider, b = ['M_' + c if not c.startswith('M_') else c for c in consider], 'M_' + band if not band.startswith(
            'M_') else band
        for c in consider:
            if Q[b + y].get(c) and D.get(c):
                if D.get(c, -999) > Q[b + y][c]['min'] and D.get(c, 999) < Q[b + y][c]['max']:
                    mags.append([Q[b + y][c]['rms'], u.polynomial(D.get(c), Q[b + y][c]), c, b + y])

        # If no magnitude yet, try the 'all'='fld'+'yng' polynomial
        if not mags:
            y = '_all'
            for c in consider:
                if Q[b + y].get(c):
                    if D.get(c, -999) > Q[b + y][c]['min'] and D.get(c, 999) < Q[b + y][c]['max']:
                        mags.append([Q[b + y][c]['rms'], u.polynomial(D.get(c), Q[b + y][c]), c, b + y])

        # If no magnitude yet, try all other polynomials
        if not mags and try_all:
            y = '_all'
            for c in RSR.keys():
                if Q[b + y].get(c):
                    if D.get(c, -999) > Q[b + y][c]['min'] and D.get(c, 999) < Q[b + y][c]['max']:
                        mags.append([Q[b + y][c]['rms'], u.polynomial(D.get(c), Q[b + y][c]), c, b + y])

        if mags:
            # If any magnitudes to return, use the one with the lowest uncertainty
            mag_unc, mag, c, p = sorted(mags)[0]

            # Calculate absolute magnitudes
            M1, E1 = round(mag, 3), round(mag_unc, 3)
            F1, U1 = u.mag2flux(b, M1, sig_m=E1, photon=False, filter_dict=RSR)

            # Caluclate apparent magnitudes
            m1, e1 = u.flux_calibrate(mag, 10 * q.pc, sig_m=mag_unc, sig_d=0.1 * q.pc, scale_to=D['d'])
            f1, u1 = u.mag2flux(b, m1, sig_m=e1, photon=False, filter_dict=RSR)

            if mag_and_flux:
                return [m1, e1, f1, u1, M1, E1, F1, U1, c]
            else:
                return ([F1, U1, c] if to_flux else [M1, E1, c]) if band.startswith('M_') else (
                    [f1, u1, c] if to_flux else [m1, e1, c])

        else:
            return ['', '', '', '', '', '', '', ''] if mag_and_flux else ['', '', '']
    except IOError:
        return ['', '', '', '', '', '', '', ''] if mag_and_flux else ['', '', '']


def get_Lbol(spectrum, d, sig_d, solar_units=False):
    """
  Returns the bolometric luminosity of *spectrum* scaled to the given distance *d*.

  Parameters
  ----------
  spectrum: sequence
    The [W,F,E] sequence of astropy.quantity arrays to be scaled and integrated
  d: astropy.quantity
    The distance to the source
  sig_d: astropy.quantity
    The uncertainty in the distance *d*
  solar_units: bool
    Return Lbol in solar units (e.g. -4.523) rather than 'erg/s'

  Returns
  -------
  The bolometric luminosity and uncertainty

  """
    fbol = (np.trapz(spectrum[1], x=spectrum[0])).to(q.erg / q.s / q.cm ** 2)
    sig_fbol = np.sqrt(np.sum((spectrum[2] * np.gradient(spectrum[0])).to(q.erg / q.s / q.cm ** 2).value ** 2))
    Lbol = (4 * np.pi * fbol * d ** 2).to(q.erg / q.s)
    sig_Lbol = Lbol * np.sqrt((sig_fbol / fbol).value ** 2 + (2 * sig_d / d).value ** 2)
    return [round(np.log10((Lbol / ac.L_sun).decompose().value), 3),
            round(abs(sig_Lbol / (Lbol * np.log(10))).value, 3)] if solar_units else [Lbol, sig_Lbol]


def get_Mbol(spectrum, d, sig_d, app=False):
    """
  Returns the bolometric magnitude of *spectrum* scaled to the given distance *d*.

  Parameters
  ----------
  spectrum: sequence
    The [W,F,E] sequence of astropy.quantity arrays to be scaled and integrated
  d: astropy.quantity
    The distance to the source
  sig_d: astropy.quantity
    The uncertainty in the distance *d*
  app: bool
    Return mbol rather than Mbol

  Returns
  -------
  The bolometric magnitude and uncertainty

  """
    fbol = (np.trapz(spectrum[1], x=spectrum[0])).to(q.erg / q.s / q.cm ** 2)
    sig_fbol = np.sqrt(np.sum((spectrum[2] * np.gradient(spectrum[0])).to(q.erg / q.s / q.cm ** 2).value ** 2))
    mbol, sig_mbol = -2.5 * np.log10(fbol.value) - 11.482, (2.5 / np.log(10)) * (
        sig_fbol / fbol).value  # Assuming L_sun = 3.86E26 W and Mbol_sun = 4.74
    Mbol, sig_Mbol = mbol - 5 * np.log10((d / 10 * q.pc).value), np.sqrt(
        sig_mbol ** 2 + ((2.5 / np.log(10)) * (sig_d / d).value) ** 2)
    return [round(mbol, 3), round(sig_mbol, 3)] if app else [round(Mbol, 3), round(sig_Mbol, 3)]


def get_teff(Lbol, sig_Lbol, r, sig_r):
    """
  Returns the effective temperature in Kelvin given the bolometric luminosity, radius, and uncertanties.

  Parameters
  ----------
  Lbol: astropy.quantity
    The bolometric luminosity
  sig_Lbol: astropy.quantity
    The uncertainty in the bolometric luminosity
  r: astropy.quantity
    The radius of the source in units of R_Jup
  sig_r: astropy.quantity
    The uncertainty in the radius

  Returns
  -------
  The effective temperature and uncertainty in Kelvin

  """
    # Lbol, r = (ac.L_sun*10**Lbol).to(q.W) if solar_units else Lbol, ac.R_jup*r
    # sig_Lbol, sig_r = (ac.L_sun*sig_Lbol/(Lbol*np.log(10))).to(q.W) if solar_units else sig_Lbol, ac.R_jup*sig_r
    r, sig_r = ac.R_jup * r, ac.R_jup * sig_r
    T = np.sqrt(np.sqrt((Lbol / (4 * np.pi * ac.sigma_sb * r ** 2)).to(q.K ** 4)))
    sig_T = T * np.sqrt((sig_Lbol / Lbol).value ** 2 + (2 * sig_r / r).value ** 2) / 4.
    return T.round(0), sig_T.round(0)


def avg_param(yparam, z, z_unc, min_age, max_age, spt, xparam='Lbol', plot=False):
    models = ['hybrid_solar_age'] + (
        filter(None, ['nc_solar_age', 'COND03' if z > -5.1 else None]) if spt > 17 else []) + (
                 filter(None, ['f2_solar_age', 'DUSTY00' if z > -5.1 else None]) if spt < 23 else [])
    x, sig_x = [np.array(i) for i in zip(
        *[isochrone_interp(z, z_unc, min_age, max_age, xparam=xparam, yparam=yparam, evo_model=m, plot=plot) for m in
          models])]
    min_x, max_x = min(x - sig_x), max(x + sig_x)
    X, X_unc = [(max_x + min_x) / 2., (max_x - min_x) / 2.]
    if plot: plt.axhline(y=X, color='k', ls='-', lw=2), plt.axhline(y=X - X_unc, color='k', ls='--', lw=2), plt.axhline(
        y=X + X_unc, color='k', ls='--', lw=2)
    return [X, X_unc]


def isochrone_interp(z, z_unc, min_age, max_age, xparam='Lbol', yparam='radius', xlabel='', ylabel='', xlims='',
                     ylims='', evo_model='hybrid_solar_age', plot=False, ages=[0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 10],
                     title=''):
    """
  Interpolates the model isochrones to obtain a range in y given a range in x
  """

    # Grab and plot the desired isochrones
    D = isochrones(evo_model=evo_model, xparam=xparam, yparam=yparam, ages=ages, plot=plot)
    Q = {d[0]: {'x': d[1], 'y': d[2]} for d in D}

    # Convert to Gyr in necessary
    if max_age > 10: min_age, max_age = min_age / 1000., max_age / 1000.

    # Pull out isochrones which lie just above and below *min_age* and *max_age*
    A = np.array(zip(*D)[0])
    min1, min2, max1, max2 = A[A <= min_age][-1] if min_age > 0.01 else 0.01, A[A >= min_age][0], A[A <= max_age][-1], \
                             A[A >= max_age][0]

    # Create a high-res x-axis in region of interest and interpolate isochrones horizontally onto new x-axis
    x = np.linspace(z - z_unc, z + z_unc, 20)
    for k, v in Q.items(): v['x'], v['y'] = x, np.interp(x, v['x'], v['y'])

    # Create isochrones interpolated vertically to *min_age* and *max_age*
    min_iso, max_iso = [np.interp(min_age, [min1, min2], [r1, r2]) for r1, r2 in zip(Q[min1]['y'], Q[min2]['y'])], [
        np.interp(max_age, [max1, max2], [r1, r2]) for r1, r2 in zip(Q[max1]['y'], Q[max2]['y'])]

    # Pull out least and greatest y value of interpolated isochrones in x range of interest
    y_min, y_max = min(min_iso + max_iso), max(min_iso + max_iso)

    if plot:
        ax = plt.gca()
        ax.set_ylabel(r'${}$'.format(ylabel or yparam), fontsize=22, labelpad=5), ax.set_xlabel(
            r'${}$'.format(xlabel or xparam), fontsize=22, labelpad=15), plt.grid(True, which='both'), plt.title(
            evo_model.replace('_', '-') if title else '')
        # ax.axvline(x=z-z_unc, ls='-', color='0.7', zorder=-3), ax.axvline(x=z+z_unc, ls='-', color='0.7', zorder=-3)
        ax.add_patch(plt.Rectangle((z - z_unc, 0), 2 * z_unc, 10, color='0.7', zorder=-3))
        xlims, ylims = ax.get_xlim(), ax.get_ylim()
        ax.fill_between([-100, 100], y_min, y_max, color='#99e6ff', zorder=-3)
        # plt.plot(x, min_iso, ls='--', color='r'), plt.plot(x, max_iso, ls='--', color='r')
        plt.xlim(xlims), plt.ylim(ylims)

    return [round(np.mean([y_min, y_max]), 2), round(abs(y_min - np.mean([y_min, y_max])), 2)]


def isochrones(evo_model='hybrid_solar_age', xparam='Lbol', yparam='radius',
               ages=[0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 10], plot=False, overplot=False):
    if plot:
        if overplot:
            ax = plt.gca()
        else:
            fig = plt.figure()
            ax = plt.subplot(111)
    DME = DMEstar(ages, xparam=xparam, yparam=yparam)
    D, data = [], [d for d in
                   np.genfromtxt(package + '/Data/Models/Evolutionary/{}.txt'.format(evo_model), delimiter=',',
                                 usecols=range(6)) if d[0] in ages and d[0] in zip(*DME)[0]]

    for k, g in groupby(data, key=lambda y: y[0]):
        age, mass, teff, Lbol, logg, radius = [np.array(i) for i in zip(*[list(i) for i in list(g)])[:6]]
        mass *= 1047.2
        radius *= 9.72847
        x = mass if xparam == 'mass' else logg if xparam == 'logg' else radius if xparam == 'radius' else Lbol
        y = mass if yparam == 'mass' else logg if yparam == 'logg' else radius if yparam == 'radius' else Lbol
        for idx, m in zip([15, 25, 0, 30, 28, 20, 20, 20], DME):
            if m[0] == k:
                (x1, y1), (x3, y3) = (x, y) if x[0] < m[1][0] else (m[1], m[2]), (x, y) if x[-1] > m[1][-1] else (
                    m[1], m[2])
                x2, y2 = np.arange(x1[0], x3[-1], 0.05), si.interp1d(np.concatenate([x1, x3]), np.concatenate([y1, y3]),
                                                                     kind='cubic')
                x2, y2 = x2[np.logical_and(x2 > x1[-1], x2 < x3[0])], y2(x2)[np.logical_and(x2 > x1[-1], x2 < x3[0])]
                xnew, ynew = np.concatenate([x1, x2, x3]), np.concatenate([y1, y2, y3])
                if plot:
                    ax.plot(x1, y1, ls='-', c='0.5', zorder=-2), ax.plot(x2, y2, ls='--', c='0.5', zorder=-2)
                    ax.annotate(k, color='0.5', xy=(xnew[idx], ynew[idx]), fontsize=15,
                                bbox=dict(boxstyle="round", fc='#99e6ff' if str(k) == '0.1' else 'w', ec='none'),
                                xycoords='data', xytext=(0, 0), textcoords='offset points',
                                horizontalalignment='center', verticalalignment='center', zorder=-1)
        D.append([k, xnew, ynew])
        if plot: ax.plot(x3, y3, ls='-', c='0.5', zorder=-2)
    return D


# ======================================================================================================================================================
# ============================================== PLOTTING =================================================================================================
# ======================================================================================================================================================

def features(spectrum, fs=20, color='k'):
    # band, bandhead, singlet, doublet
    feature = {'VO': {'type': 'band', 'start': 0.7534, 'end': 0.7734},
               'Na I': {'type': 'doublet', 'start': 0.8183, 'end': 0.8195},
               'K I': {'type': 'singlet', 'start': 1.516, 'end': None}}
    telluric = {'0.93-0.96': {'type': 'telluric', 'start': 0.93, 'end': 0.96}}
    # for f in feature.keys()+telluric.keys():
    for f, d in telluric.items():
        t, start, end = d['type'], d['start'], d['end']
        height = np.interp(start, spectrum[0].value, spectrum[1].value) if t == 'singlet' else max(
            spectrum[1][np.where((spectrum[0].value >= start) & (spectrum[0].value <= end))]) * 1.5
        if t == 'band':
            plt.loglog([start, end], [height, height], c=color), plt.text((end + start) / 2, height * 1.1, f,
                                                                          ha='center', fontsize=fs, color=color)
        elif t == 'bandhead':
            plt.annotate('CH_4', xy=((start+end)/2., height*1.1), xycoords='data', ha='center')
            plt.annotate('', xy=(start, height), xycoords='data', xytext=(end, height), textcoords='data', ha='center',
                  arrowprops=dict(arrowstyle="-", connectionstyle="bar", ec="k", shrinkA=20, shrinkB=5))
        elif t == 'singlet':
            plt.annotate(f, xy=(start, height), xytext=(start, height * 1.1),
                         arrowprops=dict(fc=color, ec=color, arrowstyle='-'), ha='center', fontsize=fs, color=color)
        elif t == 'doublet':
            plt.annotate(f, xy=(start, height), xytext=(end, height),
                         arrowprops=dict(fc=color, ec=color, arrowstyle="-", connectionstyle="bar", shrinkA=50,
                                         shrinkB=0), ha='center', fontsize=fs, color=color)
        elif t == 'telluric':
            plt.loglog([start, end], [height, height], c=color, ls='-'), plt.text((end + start) / 2, height * 1.1,
                                                                                  r'$\oplus$', ha='center', fontsize=fs,
                                                                                  color=color)


def spectral_index(spectrum, spec_index, db, data_table='', plot=False):
    '''
  Return value of given *spec_index* for input *spectrum* as spectrum_id or [w,f,e]
  '''
    indeces = {'IRS-CH4': {'w11': 8.2, 'w12': 8.8, 'w21': 9.7, 'w22': 10.3},
               'IRS-NH3': {'w11': 9.7, 'w12': 10.3, 'w21': 10.5, 'w22': 11.1},
               'H2O_A': {'w11': 1.55, 'w12': 1.56, 'w21': 1.492, 'w22': 1.502},
               'Na': {'w11': 1.15, 'w12': 1.16, 'w21': 1.134, 'w22': 1.144}}

    if isinstance(spectrum, int):
        data = db.query("SELECT * FROM spectra WHERE id={}".format(str(spectrum)), fetch='one', fmt='dict')
        w, f, e = data['wavelength'] / (1000. if data['wavelength'][0] > 500 else 1.), data['flux'], data['unc']
    elif isinstance(spectrum, list):
        w, f, e = [i.value for i in spectrum]

    if w[0] < indeces[spec_index]['w11'] and w[-1] > indeces[spec_index]['w12'] and w[0] < indeces[spec_index][
        'w21'] and w[-1] > indeces[spec_index]['w22']:
        try:
            (f11, e11), (f12, e12), (f21, e21), (f22, e22) = [np.interp(indeces[spec_index]['w11'], w, i) for i in
                                                              [f, e]], [np.interp(indeces[spec_index]['w12'], w, i) for
                                                                        i in [f, e]], [
                                                                 np.interp(indeces[spec_index]['w21'], w, i) for i in
                                                                 [f, e]], [np.interp(indeces[spec_index]['w22'], w, i)
                                                                           for i in [f, e]]
            w, f, e = [np.array(x) for x in zip(*sorted(zip(*[np.concatenate([i, np.array(v)]) for i, v in
                                                              zip([w, f, e], [[indeces[spec_index]['w11'],
                                                                               indeces[spec_index]['w12'],
                                                                               indeces[spec_index]['w21'],
                                                                               indeces[spec_index]['w22']],
                                                                              [f11, f12, f21, f22],
                                                                              [e11, e12, e21, e22]])])))]
            r1, r2 = np.logical_and(w >= indeces[spec_index]['w11'], w <= indeces[spec_index]['w12']), np.logical_and(
                w >= indeces[spec_index]['w21'], w <= indeces[spec_index]['w22'])
            idx, idx_unc = np.trapz(f[r1]) / np.trapz(f[r2]), np.sqrt(np.sum((e[r1] * np.gradient(w[r2])) ** 2))

            # Plot index
            if plot:
                db.plot_spectrum(spectrum)
                ax = plt.gca()
                xmin, xmax, ymin, ymax = min(indeces[spec_index]['w11'], indeces[spec_index]['w21']) * 0.9, max(
                    indeces[spec_index]['w12'], indeces[spec_index]['w22']) * 1.1, min(f11, f12, f21, f22) * 0.5, max(
                    f11, f12, f21, f22) * 1.5
                ax.add_patch(plt.Rectangle((indeces[spec_index]['w11'], ymin),
                                           indeces[spec_index]['w12'] - indeces[spec_index]['w11'], ymax, color='k',
                                           alpha=0.2, zorder=5)), ax.add_patch(
                    plt.Rectangle((indeces[spec_index]['w21'], ymin),
                                  indeces[spec_index]['w22'] - indeces[spec_index]['w21'], ymax, color='k', alpha=0.2,
                                  zorder=5)), ax.set_xlim(xmin, xmax), ax.set_ylim(ymin, ymax)
        except:
            idx, idx_unc = '', ''

    else:
        idx, idx_unc = '', ''

    return [idx, idx_unc]


def NYMGs():
    D = {'TW Hya': {'age_min': 8, 'age_max': 20, 'age_ref': 0},
         'beta Pic': {'age_min': 12, 'age_max': 22, 'age_ref': 0},
         'Tuc-Hor': {'age_min': 10, 'age_max': 40, 'age_ref': 0},
         'Columba': {'age_min': 10, 'age_max': 40, 'age_ref': 0},
         'Carina': {'age_min': 10, 'age_max': 40, 'age_ref': 0},
         'Argus': {'age_min': 30, 'age_max': 50, 'age_ref': 0},
         'AB Dor': {'age_min': 50, 'age_max': 120, 'age_ref': 0},
         'Pleiades': {'age_min': 110, 'age_max': 130, 'age_ref': 0}}
    return D


class GetData(object):
    def __init__(self, pickle_path):
        """
        Loads the data pickle constructed from SED calculations

        Parameters
        ----------
        pickle_path: str
          The path to the data pickle

        """
        try:
            self.pickle = open(pickle_path, 'rb')
            self.data = pickle.load(self.pickle)
            self.path = pickle_path
            self.close = self.pickle.close()
            print('Data from {} loaded!'.format(pickle_path))

        except IOError:
            print("Data from {} not loaded! Try again.".format(pickle_path))

    def add_source(self, data_dict, name, update=False):
        """
        Adds data to the pickle

        Parameters
        ----------
        data_dict: dict
          A nested dictionary of new data to add to self.data
        name: str
          The dictionary key to use for the new data
        update: bool
          Performs an update of the nested dictionary instead of replacing it

        """
        if isinstance(data_dict, dict):
            # Add the data to the active dictionary so we don't have to reload
            if update:
                self.data[name].update(data_dict)
            else:
                self.data[name] = data_dict

            # ...and add it to the pickle permanently
            try:
                pickle.dump(self.data, open(self.path, 'wb'))
                print('{} data added to {} pickle!'.format(name, self.path))
            except:
                print('Ut oh! {} data NOT added to {} pickle!'.format(name, self.path))

        else:
            print('The data input must be in the form of a Python dictionary.')

    def delete_source(self, name):
        """
        Removes the nested dictionary associated with the given source

        Parameters
        ----------
        name: str
          The dictionary key to delete from the data pickle

        """
        if name in self.data:
            # Remove this source from the active dictionary so we don't have to reload
            self.data.pop(name)

            # ...and remove it from the pickle permanently
            try:
                pickle.dump(self.data, open(self.path, 'wb'))
                print('{} data removed from {} pickle!'.format(name, self.path))
            except TypeError:
                print('Ut oh! {} data NOT removed from {} pickle!'.format(name, self.path))

        else:
            print('Source {} not found in {} pickle.'.format(name, self.data))

    def export_SED(self, name, filepath, key='SED_abs', header=True, 
                   wavelength_units='', flux_units=''):
        """
        Export an SEDs to an ascii file.

        Parameters
        ----------
        name: str
            The name of the object in the dictionary
        filepath: str
            The path to the directory for the file
        key: str
            The keyword to export
        header: bool
             Include the header
        flux_units: astropy.units.core.CompositeUnit
            The astropy units to convert to
        wavelength_units: astropy.units.core.CompositeUnit
            The astropy units to convert to
        """
        # Pull the object from the dictionary
        data = self.data[name]
        
        # Get the desired SED
        sed = data.get(key)
        
        # Convert to desired units
        sed[0] = sed[0].to(q.um)
        
        if flux_units==q.Jy:
            sed[1] = (sed[1]*sed[0]**2/ac.c).to(q.Jy)
            if len(sed)==3:
                sed[2] = (sed[2]*sed[0]**2/ac.c).to(q.Jy)
        else:
            sed[1] = sed[1].to(flux_units)
            if len(sed)==3:
                sed[2] = sed[2].to(flux_units)
        
        if sed:
            # Pull out all other values which are not arrays and place in the header
            head = sorted([['# {}'.format(k), str(v)] for k, v in data.items() 
                              if 'SED' not in k and 'RJ' not in k], key=lambda x: x[0])

            # Open the file
            fn = filepath + name.replace(' ', '_').replace('+', '%2B') + '.txt'

            # Write the header
            if head and header:
                try:
                    ascii.write([np.asarray(i) for i in np.asarray(head).T], fn, delimiter='=', format='no_header')
                except IOError:
                    pass

            # Write the data
            names = ['# wavelength [{}]'.format(str(wavelength_units)), 'flux [{}]'.format(str(flux_units))]
            if len(sed) == 3: 
                names += ['unc [{}]'.format(str(flux_units))]
            names[-1] = names[-1] + ' / {}'.format(key)

            with open(fn, mode='a') as f:
                ascii.write([np.asarray(i, dtype=np.float64) for i in sed], f, names=names, delimiter='\t')

        else:
            print('No {} for source {}'.format(key, name))

    def generate_mag_mag_relations(self, mag_mag_pickle=package + '/Data/Pickles/polynomial_relations.p', \
                                   pop=['TWA 27B', 'CD-35 2722b', 'HR8799b', '2MASS J11271382-3735076',
                                        'WISEA J182831.08+265037.6', '51 Eridani b']):
        """
        Generate estimated optical and MIR magnitudes for objects with NIR photometry based on magnitude-magnitude relations of the flux calibrated sample

        Parameters
        ----------
        mag_mag_pickle: str
          The path to the pickle which will store the magnitude-magnitude relations
        pop: sequence (ooptional)
          A list of sources to exclude when calculating mag-mag relations

        """
        bands, est_mags, rms_vals = ['M_' + i for i in RSR.keys()], [], []
        # pickle.dump({}, open(mag_mag_pickle,'wb'))
        Q = {}.fromkeys([b + '_fld' for b in bands] + [b + '_yng' for b in bands] + [b + '_all' for b in bands])

        # Photometry
        for b in bands:
            Q[b + '_fld'], Q[b + '_yng'], Q[b + '_all'] = {}.fromkeys(bands), {}.fromkeys(bands), {}.fromkeys(bands)

            # Iterate through and create polynomials for field, young, and all objects
            for name, groups in zip(['_fld', '_yng', '_all'], [('fld'), ('ymg', 'low-g'), ('fld', 'ymg', 'low-g')]):

                # Create band-band plots, fit polynomials, and store coefficients in polynomial_relations.p
                for c in bands:
                    # See how many objects qualify
                    sample = \
                        zip(*self.search(['SpT'], [c, b] + (['NYMG|gravity'] if name == '_yng' else [])) or ['none'])[0]

                    # If there are more than 10 objects across spectral types L0-T0, add the relation to the dictionary
                    if c != b and len(sample) > 20 and min(sample) <= 10 and max(sample) >= 20:

                        # Pop the band on interest from the dictionary, calculate polynomial, and add it as a nested dictionary
                        try:
                            P = \
                                self.mag_plot(c, b, pop=pop, fit=[(groups, 3, 'k', '-')], weighting=False,
                                              est_mags=False)[
                                    0][1:]
                            plt.close()
                            if P[1] != 0:
                                Q[b + name][c] = {'rms': P[1], 'min': P[0][0], 'max': P[0][1], 'yparam': b + name,
                                                  'xparam': c}
                                Q[b + name][c].update({'c{}'.format(n): p for n, p in enumerate(P[2:])})
                                print('{} - {} relation added!'.format(b + name, c))
                        except:
                            pass

        pickle.dump(Q, open(mag_mag_pickle, 'wb'))

    def mag_plot(self, xparam, yparam, zparam='', add_data={}, pct_lim=1000, est_mags=True, plot_field=True, \
                 identify=[], label_objects={}, pop=[], sources=[], binaries=False, allow_null_unc=False, \
                 fit=[], weighting=True, spt=['M', 'L', 'T', 'Y'], groups=['fld', 'low-g', 'ymg'], \
                 evo_model='hybrid_solar_age', biny=False, id_NYMGs=False, legend=True, add_text=False, \
                 xlabel='', xlims='', xticks=[], invertx='', xmaglimits='', \
                 ylabel='', ylims='', yticks=[], inverty='', ymaglimits='', \
                 zlabel='', zlims='', zticks=[], invertz='', zmaglimits='', \
                 border=['#FFA821', '#FFA821', 'k', 'r', 'r', 'k', '#2B89D6', '#2B89D6', 'k', '#7F00FF', '#7F00FF', 'k'], \
                 markers=['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'], \
                 colors=['#FFA821', 'r', '#2B89D6', '#7F00FF'], \
                 fill=['#FFA821', 'w', '#FFA821', 'r', 'w', 'r', '#2B89D6', 'w', '#2B89D6', '#7F00FF', 'w', '#7F00FF'], \
                 overplot=False, grid=False, fontsize=20, figsize=(10, 8), unity=False, save='', alpha=1, \
                 verbose=False, output_data=False, return_data='polynomials', save_polynomial=''):
        '''
        Plots the given parameters for all available objects in the given data_table

        Parameters
        ----------
        xparam: str
          The key for the given parameter in the data_table dictionary to serve as the x value.
          This can be a single key like 'J' or 'teff' or the difference of two keys, e.g. 'J-Ks')
        yparam: str
          The key for the given parameter in the data_table dictionary to serve as the y value.
          This can be a single key like 'J' or 'teff' or the difference of two keys, e.g. 'J-Ks')
        zparam: str (optional)
          The key for the given parameter in the data_table dictionary to serve as the z value.
          This can be a single key like 'J' or 'teff' or the difference of two keys, e.g. 'J-Ks')
        data_table: dict
          The nested dictionary of objects to potentially plot
        add_data: dict (optional)
          A nested dictionary of additional objects to plot that are not present in the data_table
        identify: list, tuple (optional)
          A sequence of the object names to identify with a star on the plot

        '''
        D = copy.deepcopy(self.data)
        x, y, z, data_out = xparam.split('-'), yparam.split('-'), zparam.split('-'), []
        NYMG_dict = {'TW Hya': 'k', 'beta Pic': 'c', 'Tuc-Hor': 'g', 'Columba': 'm', 'Carina': 'k', 'Argus': 'y',
                     'AB Dor': 'b', 'Pleiades': 'r'}

        # Set limits on x and y axis data
        xmaglimits, ymaglimits = xmaglimits or (-np.inf, np.inf), ymaglimits or (-np.inf, np.inf)

        # Add additional sources
        D.update(add_data)

        # Pop unwanted sources
        for name in pop:
            try:
                D.pop(name)
            except:
                pass

        # Include specific sources
        if sources:
            D = {k: v for k, v in D.items() if k in sources}

        def num(Q):
            ''' Strips alphanumeric strings and Quantities of chars and units and turns them into floats '''
            if Q and not isinstance(Q, (float, int)):
                if isinstance(Q, q.quantity.Quantity):
                    Q = Q.value
                elif isinstance(Q, str):
                    Q = float(re.sub(r'[a-zA-Z]', '', Q))
                else:
                    Q = 0
            return Q

        # =======================================================================================================================================================================
        # ============================================ Sorting ==================================================================================================================
        # =======================================================================================================================================================================

        # Iterate through data and add object to appropriate group
        M_all, M_fld, M_lowg, M_ymg = [], [], [], []
        L_all, L_fld, L_lowg, L_ymg = [], [], [], []
        T_all, T_fld, T_lowg, T_ymg = [], [], [], []
        Y_all, Y_fld, Y_lowg, Y_ymg = [], [], [], []
        beta, gamma, binary, circle = [], [], [], []
        labels, rejected = [], []

        for name, v in D.items():

            # Try to add an estimated mag to the dictionary
            for b in filter(None, x + y + z):

                # If synthetic magnitude exists but survey magnitude doesn't, use the synthetic magnitude
                if v.get(b) and not v.get(b + '_unc') and b not in ['source_id'] and not allow_null_unc:
                    v[b], v[b + '_unc'] = v.get('syn_' + b), v.get('syn_' + b + '_unc')

                # Estimate photometry based on linear mag-mag relations
                if est_mags:
                    if v.get('d'):
                        # Convert MKO JHK mags to 2MASS JHKs mags if necessary
                        TMS = ['M_2MASS_J', 'M_2MASS_H', 'M_2MASS_Ks', '2MASS_J', '2MASS_H', '2MASS_Ks']
                        MKO = ['M_MKO_J', 'M_MKO_H', 'M_MKO_K', 'MKO_J', 'MKO_H', 'MKO_K']
                        Lband = np.array(['M_IRAC_ch1', 'M_WISE_W1', "M_MKO_L'"])

                        for tms, mko in zip(TMS, MKO):
                            if b == tms and not v.get(b):
                                try:
                                    v[b], v[b + '_unc'], v[b + '_ref'] = mag_mag_relations(b, v, [mko], to_flux=False)
                                except:
                                    pass
                        for mko, tms in zip(MKO, TMS):
                            if b == mko and not v.get(b):
                                try:
                                    v[b], v[b + '_unc'], v[b + '_ref'] = mag_mag_relations(b, v, [tms], to_flux=False)
                                except:
                                    pass
                        for l in Lband:
                            if l[2:] in b and any([v.get(i) for i in Lband[Lband != l]]):
                                try:
                                    v[b], v[b + '_unc'], v[b + '_ref'] = mag_mag_relations(b, v, Lband[Lband != l],
                                                                                           to_flux=False)
                                except:
                                    pass
                else:
                    # Exclude all magnitudes that were estimated from polynomials, i.e. have an absolute magnitude as the reference
                    if str(v.get(b.replace('M_', '') + '_ref')).startswith('M_'):
                        v.pop(b)

            # Check to see if all the appropriate parameters are present from the SED
            if all([v.get(i) for i in filter(None, x + y + z)]):
                if 'SpT' not in [x, y, z] and not isinstance(v.get('SpT'), (int, float)): v['SpT'] = 13
                try:
                    # Pull out x, y, and z values and caluculate differences if necessary,
                    # e.g. 'J-W2' retrieves J and W2 mags separately and calculates color
                    i, j, k = [(np.diff(list(reversed([num(v.get(m, 0)) \
                                                           if all(
                        [num(v.get(n, 0)) and num(v.get(n + '_unc', 0)) for n in p]) \
                                                           else 0 for m in p]))) if len(p) == 2 \
                                    else [num(v.get(p[0], 0))])[0] for p in [x, y, z]]

                    # Pull out uncertainties and caluclate the sum of the squares if the parameter is a color
                    i_unc, j_unc, k_unc = [np.sqrt(sum([num(v.get(e + '_unc', 0)) ** 2 for e in p])) for p in [x, y, z]]

                    # Put all the applicable data into a list to pass through plotting criteria
                    data = [name, i, i_unc, j, j_unc, k if zparam else True, k_unc if zparam else True, \
                            v.get('gravity') or (True if v.get('age_min') < 500 else False), \
                            v.get('binary'), v.get('SpT'), v.get('SpT_unc', 0.5), v.get('NYMG')]

                    # If all the necessary data is there, drop it into the appropriate category for plotting
                    if all([data[1], data[3], data[5]]) \
                            and (binaries or (not binaries and not data[-4])) \
                            and (all([data[2], data[4], data[6]]) or allow_null_unc) \
                            and all([data[1] > xmaglimits[0], data[1] < xmaglimits[1], data[3] > ymaglimits[0],
                                     data[3] < ymaglimits[1]]) \
                            and all([(100 * ((np.e ** err - 1) if param in RSR.keys() or 'bol' in param \
                                                     else (err / val)) < pct_lim) if 'SpT' not in param else True \
                                     for param, val, err in
                                     zip([xparam, yparam], [data[1], data[3]], [data[2], data[4]])]):

                        # Is it a binary?
                        if v.get('binary') and binaries: binary.append(data)

                        # Sort through and put them in the appropriate *spt* and *groups*
                        for A, Y, L, F, low, high, S in zip([M_all, L_all, T_all, Y_all], \
                                                            [M_ymg, L_ymg, T_ymg, Y_ymg], \
                                                            [M_lowg, L_lowg, T_lowg, Y_lowg], \
                                                            [M_fld, L_fld, T_fld, Y_fld], \
                                                            range(0, 40, 10), range(10, 50, 10), \
                                                            ['M', 'L', 'T', 'Y']):

                            if data[-3] >= low and data[-3] < high and S in spt:

                                # Get total count
                                if all([data[-1], 'ymg' in groups]) \
                                        or all([data[7], 'low-g' in groups]) \
                                        or all(['fld' in groups, not data[-1], not data[7]]):
                                    A.append(data)

                                # Is the object field age, low gravity, or a NYMG member?
                                Y.append(data) if data[-1] and 'ymg' in groups \
                                    else L.append(data) if data[7] and 'low-g' in groups \
                                    else F.append(data) if not data[-1] and not data[7] and 'fld' in groups \
                                    else None

                                # Distinguish between beta and gamma
                                beta.append(data) if data[7] == 'b' and 'beta' in groups \
                                    else gamma.append(data) if data[7] == 'g' and 'gamma' in groups \
                                    else None

                        # Is it in the list of objects to identify or label?
                        if data[0] in identify:
                            circle.append(data)

                        if data[0] in label_objects.keys() and any([data in i for i in [M_all, L_all, T_all, Y_all]]):
                            labels.append([label_objects[data[0]].get('label', data[0]), data[1], data[3], \
                                           label_objects[data[0]]['dx'], label_objects[data[0]]['dy']])

                        # If object didn't make it into any of the bins, put it in a rejection table
                        if not any([data in i for i in [M_all, L_all, T_all, Y_all]]): rejected.append(data)

                    else:
                        rejected.append(data)
                except:
                    print('error: ', name)

        # =======================================================================================================================================================================
        # ============================================ Plotting =================================================================================================================
        # =======================================================================================================================================================================

        # Plot each group with specified formatting
        if any([M_fld, M_lowg, M_ymg, L_fld, L_lowg, L_ymg, T_fld, T_lowg, T_ymg, Y_fld, Y_lowg, Y_ymg, beta, gamma,
                binary]):
            fill = fill or [colors[0], 'w', colors[0], colors[1], 'w', colors[1], colors[2], 'w', colors[2], colors[3],
                            'w', colors[3]]
            border = border or [colors[0], colors[0], 'k', colors[1], colors[1], 'k', colors[2], colors[2], 'k',
                                colors[3], colors[3], 'k']
            if not overplot:
                if zparam: from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=figsize)
                ax = plt.subplot(111,
                                 projection='3d' if zparam else 'aitoff' if xparam == 'ra' and yparam == 'dec' else 'rectilinear')
                plt.rc('text', usetex=True, fontsize=fontsize)
                ax.set_xlabel(r'${}$'.format(xlabel or '\mbox{Spectral Type}' if 'SpT' in xparam else xparam),
                              labelpad=fontsize * 3 / 4, fontsize=fontsize + 4)
                ax.set_ylabel(r'${}$'.format(ylabel or yparam), labelpad=fontsize * 3 / 4, fontsize=fontsize + 4)
                if zparam: ax.set_zlabel(r'${}$'.format(zlabel or zparam), labelpad=fontsize, fontsize=fontsize + 4)
            else:
                ax = overplot if hasattr(overplot, 'figure') else plt.gca()

            if biny and 'SpT' in xparam:
                M_fld, M_lowg, M_ymg = [], [], []
                L_fld, L_lowg, L_ymg = [], [], []
                T_fld, T_lowg, T_ymg = [], [], []
                Y_fld, Y_lowg, Y_ymg = [], [], []

                for ungrp, grp in zip([M_all, L_all, T_all, Y_all], [M_fld, L_fld, T_fld, Y_fld]):
                    for k, g in groupby(sorted(ungrp, key=lambda x: int(x[1])), lambda y: int(y[1])):
                        G = list(g)
                        grp.append([str(k), k, 0.5,
                                    np.average(np.array([i[3] for i in G]), weights=1. / np.array([i[4] for i in G]) \
                                        if any([i[4] for i in G]) else None), \
                                    np.sqrt(sum(np.array([i[4] for i in G]) ** 2)) \
                                        if any([i[4] for i in G]) else np.std([i[3] for i in G]), 0, 0, 0, 0, k, 0.5,
                                    None])

            # Do polynomial fits of *degree* to data
            if fit:
                for grps, degree, c, ls in fit:
                    sample = (M_fld + L_fld + T_fld + Y_fld if 'fld' in grps else []) \
                             + (M_lowg + L_lowg + T_lowg + Y_lowg if 'low-g' in grps else []) \
                             + (M_ymg + L_ymg + T_ymg + Y_ymg if 'ymg' in grps else []) \
                             + (beta if 'beta' in grps else []) \
                             + (gamma if 'gamma' in grps else [])

                    if sample and not zparam:
                        N, X, Xsig, Y, Ysig, Z, Zsig, G, B, S, Ssig, NYMG = zip(*[i for i in sample if not i[8]])
                        if return_data == 'polynomials':
                            suffix = '_fld' if 'fld' in grps and 'ymg' not in grps and 'low-g' not in grps \
                                else '_yng' if 'fld' not in grps and 'ymg' in grps and 'low-g' in grps \
                                else '_all'
                            pr = u.output_polynomial(map(float, X), map(float, Y),
                                                     sig=map(float, Ysig) if weighting else '', \
                                                     title='{} | {}'.format(spt, grps), degree=degree, x=xparam,
                                                     y=yparam + suffix, verbose=verbose, \
                                                     c=c, ls=ls, legend=False, ax=ax)
                            if save_polynomial:
                                polynomial_relation(pr['xparam'], pr['yparam'], polynomial=pr,
                                                    pickle_path=save_polynomial)

            # Plot the data
            for z, l, m, c, e in zip(*[
                [M_fld, M_lowg, M_ymg, L_fld, L_lowg, L_ymg, T_fld, T_lowg, T_ymg, Y_fld, Y_lowg, Y_ymg, beta, gamma,
                 binary, circle], \
                    ['M Field', 'M ' + r'$\beta /\gamma$', 'M NYMG', 'L Field', 'L ' + r'$\beta /\gamma$', 'L NYMG', \
                     'T Field', 'T ' + r'$\beta /\gamma$', 'T NYMG', 'Y Field', 'Y ' + r'$\beta /\gamma$', 'Y NYMG', \
                     'Beta', 'Gamma', 'Binary', 'Interesting'], \
                            markers + ['d', 'd', 's', '*'], fill + ['w', 'w', 'none', 'k'],
                        border + ['b', 'g', 'g', 'k']]):

                if z:
                    if 'NYMG' in l and id_NYMGs:
                        for k, g in groupby(sorted(z, key=lambda x: x[-1]), lambda y: y[-1]):
                            G = list(g)
                            N, X, Xsig, Y, Ysig, Z, Zsig, G, B, S, Ssig, NYMG = zip(*G)
                            if zparam:
                                ax.scatter(X, Y, Z)
                            else:
                                ax.errorbar(X, Y, ls='none', marker='d', markerfacecolor=NYMG_dict[k],
                                            markeredgecolor='k', ecolor='k', \
                                            markeredgewidth=1,
                                            markersize=10 if l == 'Binary' else 20 if l == 'Interesting' else 7,
                                            label=l, \
                                            capsize=0,
                                            zorder=0 if 'Field' in l else 3 if l in ['Binary', 'Interesting'] else 2,
                                            alpha=alpha)

                    else:
                        N, X, Xsig, Y, Ysig, Z, Zsig, G, B, S, Ssig, NYMG = zip(*z)
                        if not plot_field and 'Field' in l:
                            pass
                        else:
                            if zparam:
                                ax.scatter(X, Y, Z, marker=m, facecolor=c, edgecolor=e,
                                           linewidth=0 if 'Field' in l else 2, s=10 if l == 'Binary' else 8, \
                                           label=l,
                                           zorder=0 if 'Field' in l else 3 if l in ['Binary', 'Interesting'] else 2,
                                           alpha=alpha)
                            else:
                                ax.errorbar(X, Y, xerr=None if l == 'Binary' else [Xsig, Xsig],
                                            yerr=None if l in ['Binary', 'Interesting'] else [Ysig, Ysig], \
                                            ls='none', marker=m, markerfacecolor=c, markeredgecolor=e, ecolor=e,
                                            markeredgewidth=0 if 'Field' in l else 1, \
                                            markersize=10 if l == 'Binary' else 20 if l == 'Interesting' else 8,
                                            label=l, capsize=0, \
                                            zorder=0 if 'Field' in l else 3 if l in ['Binary', 'Interesting'] else 2,
                                            alpha=alpha)
                                plt.connect('button_press_event', interact.AnnoteFinder(X, Y, N))

                    if verbose:
                        print('\n',l,':')
                        pdata = [N, S, X, Xsig, Y, Ysig, Z, Zsig, G, B, NYMG] if zparam else [N, S, X, Xsig, Y, Ysig, G, B, NYMG]
                        colnames = ['Name', 'spt', xparam, xparam+'_unc', yparam, yparam+'_unc', zparam, zparam+'_unc', 'Gravity', 'Binary', 'Age'] \
                                   if zparam else ['Name', 'spt', xparam, xparam+'_unc', yparam, yparam+'_unc', 'Gravity', 'Binary', 'Age']
                        at.Table(pdata, names=colnames).pprint(max_width=120, max_lines=300)

                if return_data == 'params': 
                    data_out.append(z)
                
                if output_data and output_data != 'polynomials':
                    u.printer(['Name', 'SpT', xparam, xparam + '_unc', yparam, yparam + '_unc', zparam, zparam + '_unc',
                               'Gravity', 'Binary', 'Age'], \
                              zip(*[N, S, X, Xsig, Y, Ysig, Z, Zsig, G, B, NYMG]), empties=True,
                              to_txt=package + '/Files/{} v {} v {}.txt'.format(xparam, yparam, zparam)) if zparam \
                        else u.printer(['Name', xparam, xparam + '_unc', yparam, yparam + '_unc', 'Gravity', 'Binary',
                                        'Age'] if xparam == 'SpT' \
                                           else ['Name', 'SpT', xparam, xparam + '_unc', yparam, yparam + '_unc',
                                                 'Gravity', 'Binary', 'Age'],
                                       zip(*[N, X, Xsig, Y, Ysig, G, B, NYMG]) if xparam == 'SpT' \
                                           else zip(*[N, S, X, Xsig, Y, Ysig, G, B, NYMG]), empties=True,
                                       to_txt='/Files/{} v {}.txt'.format(xparam, yparam))

            # Options to format axes, draw legend and save
            if 'SpT' in xparam and spt == ['M', 'L', 'T', 'Y'] and not xticks:
                ax.set_xlim(5, 33)
                xticks = ['M6', 'M8', 'L0', 'L2', 'L4', 'L6', 'L8', 'T0', 'T2', 'T4', 'T6', 'T8', 'Y0', 'Y2']
                ax.set_xticks([6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])

            if 'SpT' in xparam and spt == ['M', 'L', 'T'] and not xticks:
                ax.set_xlim(5, 31)
                xticks = ['M6', 'L0', 'L4', 'L8', 'T2', 'T6', 'Y0']
                ax.set_xticks([6, 10, 14, 18, 22, 26, 30])

            # Axis formatting
            if grid: ax.grid(True)
            if xticks: ax.set_xticklabels(xticks)
            if xlabel: ax.set_xlabel(xlabel, labelpad=20)
            if invertx: ax.invert_xaxis()
            if xlims:
                ax.set_xlim(xlims)
            elif not xlims and all([i in RSR.keys() for i in x]):
                ax.set_xlim((min([i[1] - i[2] for i in M_all + L_all + T_all + Y_all]) - 1,
                             max([i[1] + i[2] for i in M_all + L_all + T_all + Y_all]) + 1))

            if yticks: ax.set_yticklabels(yticks)
            if ylabel: ax.set_ylabel(ylabel, labelpad=20)
            if inverty: ax.invert_yaxis()
            if ylims:
                ax.set_ylim(ylims)
            elif not ylims and all([i in RSR.keys() for i in y]):
                ax.set_ylim((min([i[3] - i[4] for i in M_all + L_all + T_all + Y_all]) - 1,
                             max([i[3] + i[4] for i in M_all + L_all + T_all + Y_all]) + 1))

            if zparam:
                if zticks: ax.set_zticklabels(zticks)
                if zlabel: ax.set_zlabel(zlabel, labelpad=20)
                if invertz: ax.invery_zaxis()
                if zlims:
                    ax.set_zlim(zlims)
                elif not zlims and all([i in RSR.keys() for i in z]):
                    ax.set_zlim((min([i[5] - i[6] for i in M_all + L_all + T_all + Y_all]) - 1,
                                 max([i[5] + i[6] for i in M_all + L_all + T_all + Y_all]) + 1))

            # Axes text
            if not zparam:
                if add_text: ax.annotate(add_text[0], xy=add_text[1], xytext=add_text[2], fontsize=add_text[3])
                if labels:
                    for l, x, y, dx, dy in labels:
                        ax.annotate(l, xy=(x, y), xytext=(x + dx, y + dy), fontsize=14)
                if unity:
                    X, Y = ax.get_xlim(), ax.get_ylim()
                    ax.plot(X, Y, c='k', ls='--')
                    ax.set_xlim(X), ax.set_ylim(Y)

                plt.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.98)

            # Plot the legend (Breaks with 3D projection)
            if legend and not zparam:
                if binaries:
                    u.manual_legend(
                        ['M Field', 'L Field', 'T Field', 'Binary', r'M$\beta /\gamma$', r'L$\beta /\gamma$',
                         r'T$\beta /\gamma$', 'M NYMG', 'L NYMG', 'T NYMG'],
                        ['#FFA821', 'r', '#2B89D6', 'w', 'w', 'w', 'w', '#FFA821', 'r', '#2B89D6'], overplot=ax,
                        markers=['o', 'o', 'o', 's', 'o', 'o', 'o', 'o', 'o', 'o'],
                        sizes=[8, 8, 8, 12, 8, 8, 8, 8, 8, 8, 8, 8],
                        edges=['#FFA821', 'r', '#2B89D6', 'g', '#FFA821', 'r', '#2B89D6', 'k', 'k', 'k'],
                        errors=[True, True, True, False, True, True, True, True, True, True],
                        styles=['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'], ncol=3, loc=legend)
                else:
                    u.manual_legend(spt + (['Field', r'$\beta /\gamma$', 'NYMG'] if groups != ['fld'] else []),
                                    colors + ['0.5', 'w', '0.5'], sizes=[8, 8, 8, 8, 8, 8, 8], overplot=ax,
                                    markers=['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
                                    edges=['#FFA821', 'r', '#2B89D6', '#7F00FF', '0.5', '0.5', '0.3'],
                                    errors=[False, False, False, False, True, True, True],
                                    ncol=1 if groups == ['fld'] else 2, loc=0)

            # Saving, returning, and printing
            if save: 
                plt.savefig(save if '.png' in save else (save + '{} vs {}.png'.format(yparam, xparam)))

            view = [['Field', 'NYMG', 'low_g', 'Total'],
                    [len(M_fld), len(M_ymg), len(M_lowg), len(M_fld + M_ymg + M_lowg)],
                    [len(L_fld), len(L_ymg), len(L_lowg), len(L_fld + L_ymg + L_lowg)],
                    [len(T_fld), len(T_ymg), len(T_lowg), len(T_fld + T_ymg + T_lowg)],
                    [len(Y_fld), len(Y_ymg), len(Y_lowg), len(Y_fld + Y_ymg + Y_lowg)],
                    [len(M_fld + L_fld + T_fld + Y_fld), len(M_ymg + L_ymg + T_ymg + Y_ymg),
                     len(M_lowg + L_lowg + T_lowg + Y_lowg), 
                     len(M_fld + M_ymg + M_lowg + L_fld + L_ymg + L_lowg + T_fld + T_ymg + T_lowg + Y_fld + Y_ymg + Y_lowg)]]
            presults = at.Table(view, names=['-','M', 'L', 'T', 'Y', 'Total'])
            print('\n')
            presults.pprint()
            
            
            if rejected and verbose:
                print('\nRejected:')
                rej = at.Table(list(zip(*rejected)), dtype=[str,float,float,float,float,float,float,str,bool,float,float,str],
                        names=['name', xparam, xparam+'_unc', yparam, yparam+'_unc', zparam or 'z',
                        zparam+'_unc' if zparam else 'z_unc', 'gravity', 'binary', 'spt', 'spt_unc', 'NYMG'])
                rej.pprint(max_width=120, max_lines=300)
                
            if return_data and data_out: 
                return data_out

        else:
            print("No objects with {} and {} values.".format(xparam, yparam))

    def RESET(self):
        """
        Empties the data_pickle after a prompt
        """
        sure = raw_input("Are you sure you want to delete all data from {} pickle? [No,Yes]".format(self.path))
        if sure.lower() == 'yes':
            try:
                pickle.dump({}, open(self.path, 'wb'))
                print('All data removed from {} pickle!'.format(self.path))
            except:
                print('Ut oh! Could NOT delete all data from {} pickle!'.format(self.path))

        else:
            print("You must respond 'Yes' to delete all data from {} pickle!".format(self.path))

    def search(self, keys, requirements, sources=[], spt='', fmt='array', to_txt=False, delim='|', keysort='', \
               verbose=False):
        '''
        Returns list of all values for *keys* of objects that satisfy all *requirements*

        Parameters
        ----------
        keys: list, tuple
          A sequence of the dictionary keys to be returned
        requirements: list, tuple
          A sequence of the dictionary keys to be evaluated as True or False.
          | (or) and ~ (not) operators recognized, e.g ['WISE_W1|IRAC_ch1'] and ['~publication_shortname']
        sources: sequence (optional)
          A list of the sources to include exclusively
        fmt: str
          Returns an array, dictionary, or Astropy table of the results given 'array', 'dict', and 'table' respectively
        to_txt: bool
          Writes an ascii file with delimiter **delim to the path supplied by **to_txt
        delim: str
          The delimiter to use when writing data to a text file
        keysort: str (option)
          Sorts the columns by the given key
        verbose: bool
          Print the details

        Returns
        -------
        result: list, dict, table
          A container of the values for all objects that satisfy the given requirements

        '''
        L = copy.deepcopy(self.data)
        result, rejected = [], []

        # Option to provide a list of sources to include
        if sources:
            for k, v in L.items():
                if k not in sources: L.pop(k)

        # Fetch the data that satisfy the requirements
        for n, d in L.items():
            if all([any([True if (not d.get(j.replace('~', '')) and j.startswith('~')) \
                                 else d.get(j) for j in i.split('|')]) for i in requirements]):
                if spt:
                    if d.get('SpT') >= spt[0] and d.get('SpT') <= spt[1]:
                        result.append([d.get(k) for k in keys])
                else:
                    result.append([d.get(k) for k in keys])
            else:
                rejected.append(n)

        # Sort the columns
        if keysort in keys:
            result = np.asarray(sorted(result, key=lambda x: x[keys.index(keysort)]))

        if fmt == 'table' and not to_txt:
            result = at.Table(np.asarray(result), names=keys)
            if keysort: result.sort(keysort)
        if fmt == 'dict' and not to_txt:
            result = {n: {k: v for k, v in d.items() if k in keys} for n, d in L.items() \
                      if all([any([True if (not d.get(j.replace('~', '')) and j.startswith('~')) \
                                       else d.get(j) for j in i.split('|')]) for i in requirements])}

        print('{}/{} records found satisfying {}.'.format(len(result), len(L), requirements))
        if verbose: print(rejected)

        # Print to file or return data
        if to_txt:
            np.savetxt(to_txt, unitless(result), header=delim.join(keys), delimiter=delim, fmt='%s')
        else:
            return result

    def subsample(self, sources):
        """
        Select a subsample from the GetData() instace given a sequence of source keys

        Parameters
        ----------
        source_list: sequence
          The list of keys to include in the GetData() instance

        """
        subset = {}
        for source in sources:
            if source in self.data:
                subset[source] = self.data[source]
            else:
                print('{} not in {}'.format(source, self.path))

        if len(subset) != len(sources):
            if raw_input('Not all specified sources were included. Proceed anyway? [y/n] ') == 'y':
                self.data = subset
            else:
                print('Subsample not created.')
        else:
            self.data = subset

    def spec_plot(self, sources=[], um=(0.5, 14.5), spt=(5, 33), teff=(0, 9999), SNR=0.5,
                  groups=['fld', 'ymg', 'low-g'], plot_phot=True, norm_to='', app=False, 
                  add_nans=[1.4, 1.85, 2.6, 4.3, 5.05], binaries=False, pop=[], highlight=[], 
                  cmap=u.truncate_colormap(plt.cm.brg_r, 0.5, 1.), cbar=True, figsize=(12, 8),
                  fontsize=18, overplot=False, legend='None', save='', low_SNR=True, 
                  ylabel='', xlabel='', lw=1, plot_integrals=False, zorder=1, **kwargs):
        """
        Plot flux calibrated or normalized SEDs for visual comparison.

        Parameters
        ----------
        sources: sequence (optional)
          A list of sources names to include exclusively
        um: sequence
          The wavelength range in microns to include in the plot
        spt: sequence
          The numeric spectral type range to include
        teff: sequence
          The effective temperture range to include
        SNR: int, float
          The signal-to-noise ratio above which the spectra should be masked
        groups: sequence
          The gravity groups to include, including 'fld' for field gravity, 'low-g' for low gravity
          designations, and 'ymg' for members of nearby young moving groups
        norm_to: sequence (optional)
          The wavelength range in microns to which all spectra should be normalized
        app: bool
          Plot apparent fluxes instead of absolute
        plot_phot: bool
          Plot the photometry
        add_nans: sequence
          A sequence of wavelength positions in microns to insert NaN values for nicer plotting
        binaries: bool
          Include known binaries
        cmap: colormap object, list
          The matplotlib colormap to use or a list of manual colors
        pop: sequence (optional)
          The sources to exclude from the plot
        highlight: sequence (optional)
          The wavelength ranges to highlight to point out interesting spectral features,
          e.g. [(6.38,6.55),(11.7,12.8),(10.3,11.3)] for MIR spectra
        cbar: bool
          Plot the color bar
        legend: int
          The 0-9 location to plot the legend. Does not plot legend if 'None'
        figsize: sequence
          The (x,y) dimensions for the plot
        fontsize: int
          The size of the font
        overplot: matplotlib figure
          The axes to plot the figure on
        xlabel: str (optional)
          The x-axis label
        ylabel: str (optional)
          The y-axis label
        save: str (optional)
          The path to save the plot

        """
        if overplot:
            ax = overplot if hasattr(overplot, 'figure') else plt.gca()
        else:
            fig, ax = plt.subplots(figsize=figsize, **kwargs)
            if cbar:
                cbar = ax.contourf([[0, 0], [0, 0]], range(teff[0], teff[1], int((teff[1] - teff[0]) / 20.)), cmap=cmap)
                ax.cla()
        L = copy.deepcopy(self.data)

        # If you want to specify the sources to include
        if sources:
            for k in L.keys():
                if k not in sources: L.pop(k)

        # Iterate through the list and plot the sources that satisfy *kwarg criteria
        plots, ylims, cidx = [], [], 0
        for k, v in L.items():
            try:
                # Get the spectrum
                spec = np.asarray(
                    [i.value if hasattr(i, 'unit') else i for i in v.get('SED_spec_' + ('app' if app else 'abs'))])

                # Add NaN to gaps
                for w in add_nans:
                    spec = np.concatenate([spec.T, [[w, np.nan, np.nan]]])
                    spec = np.asarray(spec[np.argsort(spec.T)[0]])
                    idx = np.where(spec == [w, np.nan, np.nan])[0][0]
                    try:
                        spec[idx] = [spec[idx + 1][0] * 0.999, np.nan, np.nan]
                    except:
                        pass
                    spec = spec.T

                    # Make the spectrum mask
                mask = np.logical_and(spec[0] > um[0], spec[0] < um[1])
                norm_mask = np.logical_and(spec[0] > norm_to[0], spec[0] < norm_to[1]) if norm_to else mask
                norm = 1. / np.trapz(spec[1][norm_mask], x=spec[0][norm_mask]) if norm_to else 1.

                # Get temperature
                try:
                    Teff = v.get('teff', 0 * q.K).value
                except AttributeError:
                    Teff = 0

                # Plot the spectra which satisfy the criteria
                if any(mask) \
                        and (((app or (not app and Teff > teff[0] and Teff < teff[1])) \
                                      and (binaries or (not binaries and not v.get('binary'))) \
                                      and v.get('SpT') >= spt[0] and v.get('SpT') <= spt[1] \
                                      and any(['low-g' in groups and v.get('gravity'), \
                                               'ymg' in groups and v.get('NYMG'), \
                                               'fld' in groups and not v.get('gravity') and not v.get('NYMG')]) \
                                      and k not in pop)
                             or (k in sources)):
                    try:
                        # Pick the color
                        if isinstance(cmap, (list,tuple)):
                            color = cmap[cidx]
                        else:
                            color = cmap((1. * Teff - teff[0]) / (teff[1] - teff[0]), 1.) if Teff else '0.5'

                        # Plot the integral surface
                        if plot_integrals:
                            intgrl = np.asarray([i.value if hasattr(i, 'unit') else i for i in
                                                 v.get('SED_' + ('app' if app else 'abs'))])
                            ax.plot(intgrl[0], intgrl[1], ls='--', color=color, zorder=zorder)

                        # Plot the high SNR spectra
                        hiSNRflux = np.ma.masked_where(
                            np.logical_and((spec[1][mask] / spec[2][mask]) < SNR, spec[1][mask] != np.nan),
                            spec[1][mask] * norm)
                        ax.step(spec[0][mask], hiSNRflux, lw=2 if color=='r' else 1, color=color, zorder=zorder+(2 if color=='r' else 1))
                        
                        # Plot the low SNR spectra
                        if low_SNR:
                            lwSNRflux = np.ma.masked_where(
                                np.logical_and((spec[1][mask] / spec[2][mask]) > SNR, spec[1][mask] != np.nan),
                                spec[1][mask] * norm)
                            ax.step(spec[0][mask], lwSNRflux, lw=lw, color=color, alpha=0.2, zorder=zorder)

                        # Plot photometry as well
                        if plot_phot:
                            phot = np.asarray([i.value if hasattr(i, 'unit') else i \
                                               for i in v.get('SED_phot_' + ('app' if app else 'abs'))])
                            ax.errorbar(phot[0], phot[1] * norm, yerr=phot[2] * norm, marker='o', markersize=7,
                                        markeredgewidth=1, \
                                        markeredgecolor='k', ls='none', color=color, zorder=zorder)

                        # Print the results and store the count and max and min y-axis values
                        ylims += list(hiSNRflux)
                        plots.append(
                            ['{} ({}) {}'.format(k, v.get('spectral_type'), '{} K'.format(Teff) if Teff else '-'),
                             color, 1, \
                             [k, v.get('spectral_type'), v.get('teff') or '-']])

                        # Advance the index for manual colors
                        cidx += 1

                    except IOError:
                        pass
            except IOError:
                pass

        # Show bounds on wavelength range used to normalize
        # if norm_to: ax.axvline(x=norm_to[0], color='0.8'), ax.axvline(x=norm_to[1], color='0.8')

        if any(plots):

            # Sort the list by spectral type then Teff
            plots = sorted(plots, key=lambda x: x[-1][-1], reverse=True)
            to_print = [i.pop() for i in plots]

            # Labels
            if not overplot:
                plt.rc('text', usetex=True, fontsize=fontsize)
                ax.set_ylabel(ylabel or r'$f_\lambda$' + (r'$/F_\lambda ({}-{} \mu m)$'.format(
                    *norm_to) if norm_to else r'$(\lambda) [erg s^{-1} cm^{-2} A^{-1}]$'))
                ax.set_xlabel(r'$\lambda [\mu m]$')

            # Format y-axis
            ax.set_yscale('log'), ax.set_xscale('log')
            Y = (min(ylims) * 0.6, max(ylims) * 1.4)

            for x in highlight:
                ax.fill_between(x, [Y[0]] * 2, [Y[1]] * 2, color='k', alpha=0.1, zorder=zorder)

            ax.set_ylim(Y), ax.set_xlim(um)

            # Plot the colorbar
            if cbar and not overplot:
                C = fig.colorbar(cbar)
                C.ax.set_ylabel(r'$T_{eff}(K)$')

            # Draw the legend
            if legend != 'None':
                labels, colors, sizes = zip(*plots)
                u.manual_legend(labels, colors, fontsize=fontsize - 6, sizes=sizes, overplot=ax,
                                markers=['-'] * len(labels), styles=['l'] * len(labels), loc=legend)

            # Print the results, save the figure, and return the axes
            result = np.asarray([[n + 1] + r for n, r in enumerate(to_print)])
            result = at.Table(result, names=['#','Name','SpT','Teff'])
            result.pprint()

            if save: plt.savefig(save)
            return ax, result

        else:
            print('No spectra fulfilled that criteria.')
            return


class MakeSED(object):
    def __init__(self, source_id, db, spec_ids=[], dist='', pi='', age='', membership='', radius='', binary=False,
                 pop=[],
                 SNR_trim='', SNR='', split=[], trim='', SED_trim=[], smoothing=[], est_mags=True, any_mag_mag=False,
                 evo_model='hybrid_solar_age', fit=False, save=False, write=False, weighting=True, data_pickle='', \
                 diagnostics=False):
        """
    Pulls all available data from the BDNYC Data Archive, constructs an SED, and stores all calculations at *pickle_path*

    Parameters
    ----------
    source_id: int, str
      The *source_id*, *unum*, *shortname* or *designation* for any source in the database.
    db: database instance
      The database instance to retreive data from
    spec_ids: list, tuple (optional)
      A sequence of the ids from the SPECTRA table to plot. Uses any available spectra if no list is given. Uses no spectra if 'None' is given.
    dist: list, tuple (optional)
      A distance to the object with astropy.units
    pi: list, tuple (optional)
      A sequence of the parallax and uncertainty for the object in *mas*, e.g. (110.1,2.5). This overrides the parallax pulled from the database.
    age: list, tuple (optional)
      A sequence of the lower and upper limits of the object age range in Myr, e.g. (20,300). This overrides the assumed young or field age.
    membership: str (optional)
      The name of the nearby young moving group of which the source is a member, e.g. 'AB Dor' or 'Tuc-Hor'
    radius: list, tuple (optional)
      A sequence of the lower and upper limits of the object radius range in R_Jup, e.g. (1.25,2.35). This overrides the radius inferred from model isochrones.
    binary: bool (optional)
      Assumes the source is a binary if True and a single if False
    pop: sequence (optional)
      Photometric bands to exclude from the SED
    SNR_trim: sequence or float (optional)
      A float representing the signal-to-noise ratio used to trim spectra edges or a sequence of (spec_id,snr) pairs of the SNR value used to trim a particular spectrum, e.g. 15 or [(1580,15),(4012,25)] for spec_ids 1580 and 4012
    SNR:  sequence (optional)
      A sequence of (spec_id,snr) pairs of the signal-to-noise values to use for a given spectrum
    trim: sequence (optional)
      A sequence of (spec_id,x1,x2) tuples of the lower and upper wavelength values to trim from a given spectrum
    split: sequence (optiona)
      A sequence of wavelength positions in microns at which to split the SED spectra
    SED_trim: sequence (optional)
      The (x1,x2) values to trim the full SED by, e.g. [(0,1),(1.4,100)] for just J-band
    weighting: bool
      Weight the photometry by the width of the filter profile
    smoothing: sequence (optional)
      A sequence of the (spec_id,smooth) pairs to smooth a particular spectrum by, e.g. [(1580,2)] smooths most of the peaks and troughs from spectrum 1580
    est_mags: sequence (optional)
      The photometric bands to estimate from mag-mag relations if observational magnitude is unavailable
    any_mag_mag: bool (optional)
      Use any mag-mag relation to estimate missing mags, not just the relations with the tightest correllation
    evo_model: str
      The name of the evolutionary model isochrones to use for radius, logg, and mass estimations
    data_pickle: object (optional)
      The GetData() object to write new data to

    """
        self.data, self.model_fits = {}, []

        try:

            # =====================================================================================================================================
            # ======================================= METADATA ====================================================================================
            # =====================================================================================================================================
            
            # Retreive source metadata
            source = db.query("SELECT * FROM sources WHERE id=?", (source_id,), fetch='one', fmt='dict')
            
            if source.get('names'):
                self.name = self.data['name'] = source.get('names').split(',')[0]
            else:
                 self.name = self.data['name'] = source.get('shortname') or source.get('designation') or source.get('unum') or '-'
                                            
            # Add attributes
            for k in ['ra', 'dec', 'publication_shortname', 'shortname']:
                self.data[k] = source[k]
            self.data['source_id'], self.data['binary'] = source['id'], binary
            
            print(self.name, "=" * (116 - len(self.name)))
            
            # =====================================================================================================================================
            # ======================================= SPECTRAL TYPE ===============================================================================
            # =====================================================================================================================================

            # Retreive OPT and IR spectral type data but choose OPT for M+L and IR for T+Y
            OPT_SpT = dict(db.query(
                "SELECT * FROM spectral_types WHERE source_id={} AND regime like 'OPT' and adopted=1".format(
                    source['id']), fetch='one', fmt='dict') or db.query(
                "SELECT * FROM spectral_types WHERE source_id={} AND regime like 'OPT' AND gravity<>''".format(
                    source['id']), fetch='one', fmt='dict') or db.query(
                "SELECT * FROM spectral_types WHERE source_id={} AND regime like 'OPT'".format(source['id']),
                fetch='one', fmt='dict') or {'spectral_type': '', 'spectral_type_unc': '', 'gravity': '', 'suffix': ''})
            IR_SpT = dict(db.query(
                "SELECT * FROM spectral_types WHERE source_id={} AND regime like 'IR' and adopted=1".format(
                    source['id']), fetch='one', fmt='dict') or db.query(
                "SELECT * FROM spectral_types WHERE source_id={} AND regime like 'IR' AND gravity<>''".format(
                    source['id']), fetch='one', fmt='dict') or db.query(
                "SELECT * FROM spectral_types WHERE source_id={} AND regime like 'IR'".format(source['id']),
                fetch='one', fmt='dict') or {'spectral_type': '', 'spectral_type_unc': '', 'gravity': '', 'suffix': ''})
            opt_spec_type = "{}{}{}".format(u.specType(OPT_SpT.get('spectral_type')), OPT_SpT.get('suffix') or '',
                                            OPT_SpT.get('gravity').replace('d', r'$\delta$').replace('g',
                                                                                                     r'$\gamma$').replace(
                                                'b', r'$\beta$') if OPT_SpT.get('gravity') else '') if OPT_SpT.get(
                'spectral_type') else '-'
            ir_spec_type = "{}{}{}".format(u.specType(IR_SpT.get('spectral_type')), IR_SpT.get('suffix') or '',
                                           IR_SpT.get('gravity') or '') if IR_SpT.get('spectral_type') else '-'
            SpT = OPT_SpT if any([i in opt_spec_type for i in ['M', 'L']]) else IR_SpT if ir_spec_type else OPT_SpT
            spec_type = "{}{}{}".format(u.specType(SpT.get('spectral_type')), SpT.get('suffix') or '',
                                        SpT.get('gravity').replace('d', r'$\delta$').replace('g', r'$\gamma$').replace(
                                            'b', r'$\beta$') if SpT.get('gravity') else '') if SpT.get(
                'spectral_type') else '-'
            self.data['spectral_type'], self.data['SpT'], self.data['SpT_unc'], self.data['SpT_ref'], self.data[
                'gravity'], self.data['suffix'] = spec_type, SpT.get('spectral_type'), SpT.get(
                'spectral_type_unc') or 0.5, SpT.get('publication_shortname'), SpT.get('gravity'), SpT.get('suffix')

            # =====================================================================================================================================
            # ======================================= AGE and RADIUS ==============================================================================
            # =====================================================================================================================================

            # Retreive age data from input NYMG membership, input age range, or age estimate
            NYMG = NYMGs()
            self.data['age_min'], self.data['age_max'] = age if isinstance(age, tuple) \
                else (float(NYMG[membership]['age_min']), float(NYMG[membership]['age_max'])) if membership in NYMG \
                else (10, 150) if (SpT['gravity'] and not membership) \
                else (500, 10000)
            self.data['NYMG'] = membership
            self.data['youth_indicator'] = 'NYMG' if membership \
                else 'low-g' if (SpT['gravity'] and not membership) \
                else 'host age' if self.data['age_min'] < 500 \
                else ''

            # Use radius if given
            self.data['radius'], self.data['radius_unc'] = radius or ['', '']

            # =====================================================================================================================================
            # ======================================= DISTANCE ====================================================================================
            # =====================================================================================================================================
            
            try:
                # Retreive distance manually from *dist* argument or convert parallax into distance
                parallax = db.query("SELECT * FROM parallaxes WHERE source_id={} AND adopted=1".format(source['id']),
                                    fetch='one', fmt='dict') \
                           or db.query("SELECT * FROM parallaxes WHERE source_id={}".format(source['id']), fetch='one',
                                       fmt='dict') \
                           or {'parallax': '', 'parallax_unc': '', 'publication_shortname': '', 'comments': ''}
            except:
                parallax = {'parallax': '', 'parallax_unc': '', 'publication_shortname': '', 'comments': ''}
                
            self.parallax = dict(parallax)
            if pi or dist: self.parallax['parallax'], self.parallax['parallax_unc'] = u.pi2pc(dist[0], dist[1],
                                                                                              pc2pi=True) if dist else pi
            self.data['pi'], self.data['pi_unc'], self.data['pi_ref'] = parallax['parallax'], parallax['parallax_unc'], \
                                                                        parallax['publication_shortname']
            self.data['d'], self.data['d_unc'] = dist or (
                u.pi2pc(self.data['pi'], self.data['pi_unc']) if self.data['pi_unc'] else ['', ''])
            self.data['kinematic'] = True if self.parallax['comments'] and 'kinematic' in self.parallax[
                'comments'].lower() else False
            if not self.data['d']: print('No distance for flux calibration!')

            # =====================================================================================================================================
            # ======================================= PHOTOMETRY ==================================================================================
            # =====================================================================================================================================

            # Retreive all apparent photometry
            all_photometry = db.query(
                "SELECT * FROM photometry WHERE source_id=? AND magnitude_unc IS NOT NULL AND band NOT IN ('{}')".format(
                    "','".join([i.replace("'", "''") for i in map(str, pop)])), (source['id'],))
            self.data['phot_ids'] = list(all_photometry['id'])
            all_photometry = all_photometry[['band', 'magnitude', 'magnitude_unc', 'publication_shortname']]

            # Sort and homogenize it
            phot_data = []
            for k, g in groupby(sorted(all_photometry, key=lambda x: x[0]), lambda x: x[0]):
                band = list(g)

                # If it has multiple mags or some mags with uncertainties and others with
                # upper limits, use the mag with the lowest uncertainty
                band = sorted(band, key=lambda x: (x[2] is None, x[2]))[0]

                # Homogenize the magnitudes and add them to the dataframe
                try:
                    phot_data.append([k, RSR[k]['eff'], band[1] - RSR[k]['toVega'], band[2], band[3]])
                except:
                    print('{} photometry could not be included.'.format(k))

            # Add the data to the Data Frame and index by band
            self.photometry = pd.DataFrame(phot_data, columns=('band', 'eff', 'm', 'm_unc', 'ref'))
            self.photometry.set_index('band', inplace=True)

            if not self.photometry.empty:

                # Calculate apparent fluxes
                app_data = []
                for k in self.photometry.index.values:
                    app_data += [
                        [k] + u.mag2flux(k, self.photometry.loc[:, 'm'][k], sig_m=self.photometry.loc[:, 'm_unc'][k],
                                         photon=False, filter_dict=RSR)]
                app_fluxes = pd.DataFrame(app_data, columns=('band', 'm_flux', 'm_flux_unc')).set_index('band')

                # Calculate absolute mags and fluxes if distance is provided
                if self.data['d']:
                    abs_mags = pd.DataFrame([[k] + u.flux_calibrate(self.photometry.loc[:, 'm'][k], self.data['d'],
                                                                    sig_m=self.photometry.loc[:, 'm_unc'][k],
                                                                    sig_d=self.data['d_unc']) for k in
                                             self.photometry.index.values], columns=('band', 'M', 'M_unc')).set_index(
                        'band')
                    abs_fluxes = pd.DataFrame([[k] + u.mag2flux(k, abs_mags.loc[:, 'M'][k],
                                                                sig_m=abs_mags.loc[:, 'M_unc'][k], photon=False,
                                                                filter_dict=RSR) for k in self.photometry.index.values],
                                              columns=('band', 'M_flux', 'M_flux_unc')).set_index('band')
                else:
                    abs_mags = pd.DataFrame([[k, None, None] for k in self.photometry.index.values],
                                            columns=('band', 'M', 'M_unc')).set_index('band')
                    abs_fluxes = pd.DataFrame([[k, None, None] for k in self.photometry.index.values],
                                              columns=('band', 'M_flux', 'M_flux_unc')).set_index('band')

                # Add them to the dataframe
                self.photometry = self.photometry.join([app_fluxes, abs_mags, abs_fluxes])

                # Generate absolute flux dictionary for missing photometry estimation
                if self.data['d'] and est_mags:
                    # Create dictionary with band/abs_mag key/value pairs
                    phot_dict, abs_mag_dict = df_extract(self.photometry.reset_index(),
                                                         ['band', 'M', 'M_unc']), self.data.copy()
                    phot_dict = {"M_{}".format(i): j for i, j in zip(phot_dict[0], phot_dict[1])}.items() + {
                        "M_{}_unc".format(i): j for i, j in zip(phot_dict[0], phot_dict[2])}.items()
                    abs_mag_dict.update(phot_dict)

                    # Calculate the missing magnitudes using any mag-mag relation or just specific, well-corellated ones
                    relations = {k: RSR.keys() for k in RSR.keys()} if any_mag_mag \
                        else {'SDSS_u': ['2MASS_Ks', 'MKO_K'], 'SDSS_g': ['2MASS_J', 'MKO_J'],
                              'SDSS_r': ['2MASS_H', 'MKO_H'], \
                              'SDSS_i': ['2MASS_H', 'MKO_H'], 'SDSS_z': ['2MASS_J', 'MKO_J'],
                              'IRAC_ch1': ['WISE_W1', '2MASS_Ks'], \
                              'MKO_J': ['2MASS_J'], 'MKO_H': ['2MASS_H'], 'MKO_K': ['2MASS_Ks'],
                              "MKO_L'": ['IRAC_ch1', 'WISE_W1'], \
                              'IRAC_ch2': ['WISE_W2', '2MASS_Ks'], 'IRAC_ch3': ['WISE_W1'], 'IRAC_ch4': ['WISE_W1'], \
                              '2MASS_J': ['MKO_J'], '2MASS_H': ['MKO_H'], '2MASS_Ks': ['MKO_K'],
                              'WISE_W1': ['IRAC_ch2', "MKO_L'", '2MASS_Ks'], \
                              'WISE_W2': ['IRAC_ch2', "MKO_L'", '2MASS_Ks'], 'WISE_W3': ['IRAC_ch4', 'WISE_W2']}

                    # Get absolute magnitudes for missing bands only if there is an uncertainty
                    est_fluxes = [m for m in [[k, RSR[k]['eff']] + mag_mag_relations("M_{}".format(k), abs_mag_dict, \
                                                                                     [i for i in relations.get(k) if
                                                                                      abs_mag_dict.get(
                                                                                          "M_{}_unc".format(i))], \
                                                                                     mag_and_flux=True,
                                                                                     try_all=any_mag_mag) \
                                              for k in list(set(relations.keys()) - set(self.photometry.index.values))] \
                                  if m[2] and m[3] and m[0] not in pop]

                    if any(est_fluxes):
                        abs_phot = pd.DataFrame(est_fluxes, columns=(
                            'band', 'eff', 'm', 'm_unc', 'm_flux', 'm_flux_unc', 'M', 'M_unc', 'M_flux', 'M_flux_unc',
                            'ref'))
                        self.photometry = self.photometry.reset_index().merge(abs_phot, how='outer').set_index('band')
                    else:
                        print('Could not estimate missing photometry from mag-mag relations for this object.')

                    # Generate absolute flux dictionary for flux calibration
                    abs_fluxes = df_extract(self.photometry.reset_index(), ['band', 'M_flux', 'M_flux_unc'])
                    abs_fluxes = dict(
                        {i: j for i, j in zip(abs_fluxes[0], abs_fluxes[1])}.items() + {"{}_unc".format(i): j for i, j
                                                                                        in zip(abs_fluxes[0],
                                                                                               abs_fluxes[2])}.items())
                    self.abs_fluxes = abs_fluxes

                # Generate apparent flux dictionary for flux calibration
                app_fluxes = df_extract(self.photometry.reset_index(), ['band', 'm_flux', 'm_flux_unc'])
                app_fluxes = dict(
                    {i: j for i, j in zip(app_fluxes[0], app_fluxes[1])}.items() + {"{}_unc".format(i): j for i, j in
                                                                                    zip(app_fluxes[0],
                                                                                        app_fluxes[2])}.items())
                self.app_fluxes = app_fluxes

                self.data.update(dict(self.photometry['m_flux'].T.to_dict().items() + {"{}_unc".format(k): v for k, v in
                                                                                       self.photometry[
                                                                                           'm_flux_unc'].T.to_dict().items()}.items()))
                self.data.update({"{}_ref".format(k): v for k, v in self.photometry['ref'].to_dict().items()})

            else:
                print('No photometry available for SED.')

            # =====================================================================================================================================
            # ======================================= PROCESS SPECTRA =============================================================================
            # =====================================================================================================================================

            # Retreive spectra
            if spec_ids:
                spectra = db.query(
                    "SELECT * FROM spectra WHERE id IN ({}) AND source_id=?".format(','.join(['?'] * len(spec_ids))), \
                    list(spec_ids) + [source['id']], fmt='dict')
            else:
                spectra = list(filter(None, [
                    db.query("SELECT * FROM spectra WHERE source_id=? AND regime='OPT'", (source['id'],), fetch='one',
                             fmt='dict'), \
                    db.query("SELECT * FROM spectra WHERE source_id=? AND regime='NIR' AND wavelength_order=''",
                             (source['id'],), fetch='one', fmt='dict'), \
                    db.query("SELECT * FROM spectra WHERE source_id=? AND regime='MIR'", (source['id'],), fetch='one',
                             fmt='dict')]))
                if not spectra:
                    spectra = db.query("SELECT * FROM spectra WHERE source_id=? LIMIT 5", (source['id'],), fmt='dict')

            if spec_ids and len(spec_ids) != len(spectra):
                print('Check those spec_ids! One or more does not belong to source {}.'.format(source_id))

            # Make data frame columns
            # print(db.query("pragma table_info('spectra')", unpack=True))
            spec_cols = np.array(db.query("pragma table_info('spectra')", fmt='table')['name'])
            print(spec_cols)
            spec_cols = list(spec_cols[spec_cols != 'spectrum']) + ['wavelength', 'flux', 'unc']

            # Put spectrum into arrays
            for n, sp in enumerate(spectra):
                spectrum = spectra[n].pop('spectrum')
                spectra[n]['wavelength'], spectra[n]['flux'] = spectrum.data[:2]
                try:
                    spectra[n]['unc'] = spectrum.data[2]
                except:
                    spectra[n]['unc'] = None

            # Make the data frame
            self.spectra = pd.DataFrame(spectra, columns=spec_cols).set_index('id') if spectra else pd.DataFrame(columns=spec_cols)
            self.data['spec_ids'] = list(self.spectra.index)
            units, spec_coverage = [q.um, q.erg/q.s/q.cm**2/q.AA, q.erg/q.s/q.cm**2/q.AA], []

            # Add spectra metadata to SED object
            for r in ['OPT', 'NIR', 'MIR']:
                for key, item in zip(['id', 'telescope_id', 'instrument_id', 'mode_id', 'publication_shortname'], \
                                     ['_spec', '_scope', '_inst', '_mode', '_ref']):
                    try:
                        self.data[r + item] = map(int, [j for k in [i.split(',') for i in \
                                                                    map(str, db.query(
                                                                        "SELECT * FROM spectra WHERE id in ({}) and regime='{}'".format(
                                                                            ','.join(map(str, self.data['spec_ids'])),
                                                                            r), \
                                                                        use_converters=False)[key])] for j in k])
                    except (TypeError, ValueError):
                        self.data[r + item] = None

            # Create Rayleigh-Jeans tail for MIR estimates
            RJ = [np.arange(5, 500, 0.1) * q.um, u.blackbody(np.arange(5, 500, 0.1) * q.um, 1500),
                  (u.blackbody(np.arange(5, 500, 0.1) * q.um, 1800) - u.blackbody(np.arange(5, 500, 0.1) * q.um, 1200))]

            # Homogenize spectra units, clean up arrays, and smooth and trim as needed
            clean_spectra = []
            for (spec_id, spec) in self.spectra.iterrows():
                # Pull out spectrum
                w, f, e = spec['wavelength'], spec['flux'], spec['unc']

                # Convert log(F) units to linear
                try:
                    if spec.get('flux_units').startswith('log '):
                        f, e = 10 ** f, 10 ** e
                        spec['flux_units'] = spec['flux_units'].replace('log ', '')
                except:
                    print('No flux units.')

                # Force uncertainty array if none
                if e is None:
                    e = f / 10.
                    print('No uncertainty array for spectrum {}. Using SNR=10.'.format(spec_id))

                # Check that the wavelength_units are correct
                if (w[0] > 100 and spec['wavelength_units'] == 'um') or (
                                w[0] < 100 and spec['wavelength_units'] == 'A'):
                    print('Incorrect wavelength units for spectrum {}.'.format(spec_id))

                    # Convert wavelength array into microns if necessary
                w, w_units = (u.str2Q(spec['wavelength_units'], target='um') * w).value, 'um'

                # Insert uncertainty array of set SNR to force plotting
                for snr in SNR:
                    if snr[0] == spec_id: e = f / (1. * snr[1])

                # Convert any spectra in F_nu into F_lambda
                if spec['flux_units'] == 'Jy':
                    W_nu, F_nu, E_nu = w * q.um, u.str2Q(spec['flux_units']) * f, u.str2Q(spec['flux_units']) * e
                    f, e = (ac.c * F_nu / W_nu ** 2).to(q.erg / q.s / q.cm ** 2 / q.AA).value, (
                        ac.c * E_nu / W_nu ** 2).to(q.erg / q.s / q.cm ** 2 / q.AA).value
                    spec['flux_units'] = 'erg/s/cm2/A'

                # Remove NaNs, negatives and zeroes from flux array
                w, f, e = u.unc([w, f, e])

                # Trim spectra frist up to first point with SNR>SNR_trim then manually
                spec_coverage.append(w)
                if isinstance(SNR_trim, (float, int)):
                    snr_trim = SNR_trim
                elif SNR_trim and any([i[0] == spec_id for i in SNR_trim]):
                    snr_trim = [i[1] for i in SNR_trim if i[0] == spec_id][0]
                else:
                    snr_trim = 10

                if not SNR or not any([i[0] == spec_id for i in SNR]):
                    w, f, e = [i[np.where(f / e >= snr_trim)[0][0]:np.where(f / e >= snr_trim)[0][-1] + 1] for i in
                               [w, f, e]]
                if trim and any([i[0] == spec_id for i in trim]):
                    w, f, e = u.trim_spectrum([w, f, e], [i[1:] for i in trim if i[0] == spec_id])
                if not any(w):
                    spectra.pop(n)

                # Smoothing
                if isinstance(smoothing, (float, int)):
                    f = u.smooth(f, smoothing)
                elif smoothing and any([i[0] == spec_id for i in smoothing]):
                    f = u.smooth(f, i[1])

                clean_spectra.append([w, f, e])

            # Update the spectra in the SED object
            wav, flx, err = zip(*clean_spectra)
            self.spectra['wavelength'] = wav
            self.spectra['flux'] = self.spectra['flux_app'] = flx
            self.spectra['unc'] = self.spectra['unc_app'] = err

            if self.spectra.empty: print('No spectra available for SED.')

            # =====================================================================================================================================
            # ======================================= CONSTRUCT SED ===============================================================================
            # =====================================================================================================================================

            # Group overlapping spectra and make composites where possible to form peacewise spectrum for flux calibration
            if len(self.spectra) > 1:
                groups, peacewise = u.group_spectra(clean_spectra), []
                for group in groups:
                    composite = u.make_composite([[spec[0] * q.um, spec[1] * q.erg / q.s / q.cm ** 2 / q.AA,
                                                   spec[2] * q.erg / q.s / q.cm ** 2 / q.AA] for spec in group])
                    peacewise.append(composite)
            elif len(self.spectra) == 1:
                peacewise = map(list, self.spectra[['wavelength', 'flux_app', 'unc_app']].values)
            else:
                peacewise = []

            # Splitting
            keepers = []
            if split:
                for pw in peacewise:
                    wavs = filter(None, [np.where(pw[0] < i)[0][-1] if pw[0][0] < i and pw[0][-1] > i else None for i in
                                         split])
                    keepers += map(list, zip(*[np.split(i, wavs) for i in pw]))
            else:
                keepers = peacewise
            peacewise = keepers

            # Add composite spectra to SED object
            self.composites = pd.DataFrame([[i.value if hasattr(i, 'unit') else i for i in p] for p in peacewise], \
                                           columns=['wavelength', 'flux_app', 'unc_app']) if peacewise else \
                pd.DataFrame(columns=['wavelength', 'flux_app', 'unc_app'])

            # Flux calibrate composite to available apparent magnitudes
            for (n, spec) in self.composites.iterrows():
                self.composites.loc[n][['wavelength', 'flux_app', 'unc_app']] = [i.value for i in
                                                                                 norm_to_mags(spec.values, self.data)]

            # Concatenate pieces and finalize composite spectrum with units
            self.data['SED_spec_app'] = (W, F, E) = finalize_spec(
                [np.asarray(i) * Q for i, Q in zip(u.trim_spectrum([np.concatenate(j) \
                                                                    for j in zip(
                        *self.composites[['wavelength', 'flux_app', 'unc_app']].values)], SED_trim), units)]) \
                if not self.composites.empty else [np.array([])] * 3

            # Calculate all spectral indeces
            # if not self.spectra.empty:
            #   for sp_idx in ['IRS-CH4','IRS-NH3']:
            #     self.data[sp_idx.replace('IRS-','')], self.data[sp_idx.replace('IRS-','')+'_unc'] = spectral_index([W,F,E], sp_idx)

            # Create purely photometric SED to fill in spectral SED gaps
            self.data['SED_phot_app'] = (WP0, FP0, EP0) = [Q * np.array([i.value if hasattr(i, 'unit') else i \
                                                                         for i in self.photometry[
                                                                             ['eff', 'm_flux', 'm_flux_unc']].sort(
                    'eff')[self.photometry['m_flux_unc'] != ''][l].values]) \
                                                           for l, Q in zip(['eff', 'm_flux', 'm_flux_unc'],
                                                                           units)] if not self.photometry.empty else [
                                                                                                                         np.array(
                                                                                                                             [])] * 3

            # Normalize Rayleigh-Jeans tail to the IRS spectrum past 9.5um OR to the longest wavelength photometric point
            if not self.spectra.empty and any(W[W > 9 * q.um]):
                self.data['RJ'] = RJ = u.norm_spec(RJ, [W, F, E], exclude=[(0, 9.5), (14.5, 999999)])
            else:
                self.data['RJ'] = RJ = [RJ[0], RJ[1] * FP0[-1] / np.interp(WP0[-1].value, RJ[0].value, RJ[1].value), \
                                        RJ[2] * EP0[-1] / np.interp(WP0[-1].value, RJ[0].value,
                                                                    RJ[2].value)] if not self.photometry.empty else ''

            # Exclude photometric points with spectrum coverage
            if not self.composites.empty:
                covered = []
                for n, i in enumerate(WP0):
                    for (N, spec) in self.composites.iterrows():
                        if i < spec['wavelength'][-1] * q.um and i > spec['wavelength'][0] * q.um: covered.append(n)
                WP, FP, EP = [[i for n, i in enumerate(A) if n not in covered] * Q for A, Q in
                              zip(self.data['SED_phot_app'], units)]
            else:
                WP, FP, EP = WP0, FP0, EP0

            # Use zero flux at zero wavelength from bluest data point for Wein tail approximation
            wWein, fWein, eWein = np.array([0.00001]) * q.um, np.array(
                [1E-30]) * q.erg / q.s / q.cm ** 2 / q.AA, np.array([1E-30]) * q.erg / q.s / q.cm ** 2 / q.AA

            # Create spectra + photometry SED for model fitting
            if not self.spectra.empty or not self.photometry.empty:
                specPhot = finalize_spec([i * Q for i, Q in zip([j.value for j in [np.concatenate(i) for i in
                                                                                   [[pp, ss] for pp, ss in
                                                                                    zip([WP, FP, EP], [W, F, E])]]],
                                                                units)])
            else:
                specPhot = ''

            # Create full SED from Wien tail, spectra, linear interpolation between photometry, and Rayleigh-Jeans tail
            try:
                self.data['SED_app'] = finalize_spec([np.concatenate(i) for i in [
                    [ww[wWein < min([min(i) for i in [WP, specPhot[0] or [999 * q.um]] if any(i)])], sp,
                     bb[RJ[0] > max([max(i) for i in [WP, specPhot[0] or [-999 * q.um]] if any(i)])]] for ww, bb, sp in
                    zip([wWein, fWein, eWein], RJ, specPhot)]])
            except IOError:
                self.data['SED_app'] = ''

            # =====================================================================================================================================
            # ======================================= FLUX CALIBRATE EVERYTHING ===================================================================
            # =====================================================================================================================================

            # Flux calibrate if possible
            if self.data['d'] and not self.photometry.empty:
                nb = list(self.photometry.index)[0]
                self.data['norm'] = self.photometry['M_flux'][nb].value / self.photometry['m_flux'][nb].value
                self.data['SED_abs'] = [self.data['SED_app'][0], self.data['SED_app'][1] * self.data['norm'],
                                        self.data['SED_app'][2] * self.data['norm']]
                self.data['SED_spec_abs'] = [self.data['SED_spec_app'][0],
                                             self.data['SED_spec_app'][1] * self.data['norm'],
                                             self.data['SED_spec_app'][2] * self.data['norm']]
                self.data['SED_phot_abs'] = [self.data['SED_phot_app'][0],
                                             self.data['SED_phot_app'][1] * self.data['norm'],
                                             self.data['SED_phot_app'][2] * self.data['norm']]
                self.spectra[['flux_abs', 'unc_abs']] = self.spectra[['flux_app', 'unc_app']] * self.data['norm']
                self.composites[['flux_abs', 'unc_abs']] = self.composites[['flux_app', 'unc_app']] * self.data['norm']
            else:
                self.data['SED_abs'] = self.data['SED_spec_abs'] = self.data['SED_phot_abs'] = ''
                self.data['norm'] = 1.

            # =====================================================================================================================================
            # ======================================= MEASURE FUNDAMENTAL PARAMETERS ==============================================================
            # =====================================================================================================================================

            # Calculate all fundamental paramters without models
            self.data = fundamental_params(self.data, p='', plot=diagnostics)

            # =====================================================================================================================================
            # ======================================= PRINTING ====================================================================================
            # =====================================================================================================================================

            # db.inventory(source['id'])
            print(' ')
            print(at.Table([[opt_spec_type], [ir_spec_type]], names=['OPT_SpT', 'IR_SpT']))
            print(' ')
            if not self.photometry.empty: 
                p = np.asarray(df_extract(self.photometry.reset_index(), ['band', 'm', 'm_unc', 'M', 'M_unc', 'ref'])).T
                print(at.Table(p, names=['Band', 'm', 'm_unc', 'M', 'M_unc', 'Ref']))
            print(' ')
            if not self.spectra.empty: 
                p = np.asarray(df_extract(self.spectra.reset_index(), ['regime', 'instrument_id', 'mode_id', 'telescope_id', 'publication_shortname', 'filename','obs_date'])).T
                print(at.Table(p, names=['Regime', 'Instrument', 'Mode', 'Telescope', 'Publication', 'Filename', 'Obs Date']))
            print(' ')
            if self.data['d']:
                p = np.array(['{}({})'.format(self.data['Lbol'], self.data['Lbol_unc']),
                     '-' if binary else '{}({})'.format(self.data['teff'].value, self.data['teff_unc'].value),
                     '-' if binary else '{}({})'.format(self.data['radius'], self.data['radius_unc']),
                     '-' if binary else '{}({})'.format(self.data['mass'], self.data['mass_unc']),
                     '-' if binary else '{}({})'.format(self.data['logg'], self.data['logg_unc']),
                     '{}-{}'.format(self.data['age_min'], self.data['age_max']),
                     '{}({})'.format(self.data['d'].value, self.data['d_unc'].value)]).T
                print(at.Table(p, names=['Lbol', 'Teff', 'R_Jup', 'M_Jup', 'logg', 'Age', 'Dist']))
            print(' ')

            # =====================================================================================================================================
            # ======================================= OUTPUT ======================================================================================
            # =====================================================================================================================================

            if not self.photometry.empty:
                # Add the photometry to the SED object data
                self.data.update(dict(self.photometry['m'].T.to_dict().items() + {k + '_unc': v for k, v in
                                                                                  self.photometry[
                                                                                      'm_unc'].T.to_dict().items()}.items()))
                self.data.update(dict(
                    {'M_' + k: v for k, v in self.photometry['M'].T.to_dict().items()}.items() + {'M_' + k + '_unc': v
                                                                                                  for k, v in
                                                                                                  self.photometry[
                                                                                                      'M_unc'].T.to_dict().items()}.items()))
                self.data.update(dict(
                    {k + '_flux': v for k, v in self.photometry['m_flux'].T.to_dict().items()}.items() + {
                        k + '_flux_unc': v for k, v in self.photometry['m_flux_unc'].T.to_dict().items()}.items()))
                self.data.update(dict(
                    {'M_' + k + '_flux': v for k, v in self.photometry['M_flux'].T.to_dict().items()}.items() + {
                        'M_' + k + '_flux_unc': v for k, v in
                        self.photometry['M_flux_unc'].T.to_dict().items()}.items()))

                # Send object data to the data pickle for mag_plots
            if data_pickle and (not self.spectra.empty or not self.photometry.empty):
                data_pickle.add_source(self.data, self.name)
            else:
                if self.spectra.empty and self.photometry.empty:
                    print("No spectra or photometry to build SED. No data saved.")
                else:
                    print("No data_pickle to save data!")

            # Auto-fit and use Teff +/- 300K as the model grid range
            if fit and (not self.spectra.empty or not self.photometry.empty):
                try:
                    self.fit_SED(fit, param_lims=[('teff', (round(self.data['teff'].value / 50.) * 50.) - 300.,
                                                   (round(self.data['teff'].value / 50.) * 50.) + 300., 50),
                                                  ('logg', 4.0, 5.5, 0.5)])
                except:
                    plt.close();
                    print("Couldn't perform MCMC fit to this SED.")

            # Auto-plot and save them
            if save and (not self.spectra.empty or not self.photometry.empty):
                try:
                    figpath = save + self.name.replace(' ', '_') + '.png'
                    self.plot(integrals=True, save=figpath)
                    print('SED saved as ' + figpath)
                except:
                    plt.close()
                    print("Couldn't plot this SED.")

            # Write to file
            if write and not self.spectra.empty:
                try:
                    datapath = self.name.replace(' ', '_').replace('+', '%2B') + '.txt'
                    self.write(write + datapath)
                    print('Data saved to ' + datapath)
                except:
                    print('Could not write this SED to file.')

        except IOError:
            print("Could not build SED for source {}.".format(source['id']))
        print('\n')
        
    def synthetic_mags(self, bands=RSR.keys(), data_pickle=''):
        data = self.data

        # Get the apparent SED and calculate all the synthetic mags
        spec = self.data['SED_app']
        mags = s.all_mags(spec, bands=bands, photon=False, Flam=False)
        data.update(mags)
        
        # Get the apparent SED and calculate all the synthetic mags
        spec = self.data['SED_abs']
        mags = s.all_mags(spec, bands=bands, photon=False, Flam=False)
        Mags = {'M_'+k:v for k,v in mags.items()}
        data.update(Mags)
        
        # Add it to the dictionary
        self.data = data
        
        if data_pickle:
            print("Mags added to SED:",mags.keys())
            data_pickle.add_source(self.data, self.name, update=True)

    def fit_SED(self, model_db_path, model_fits=[('bt_settl_2013', 2, 2)], mask=[(1.12, 1.16), (1.35, 1.42)],
                param_lims=[], fit_spec=True, fit_phot=False, data_pickle='', save=''):
        '''
        Perform MCMC fit of model atmosphere spectra and photometry to SED data

        Parameters
        ----------
        model_db_path: str
          The path to model_atmospheres.db
        model_fits: sequence (optional)
          A list of the MCMC fits to perform on the SED in the format (model_grid,walkers,steps), e.g. [('bt_settl_2013',100,1000)]
        mask: sequence (optional)
          Wavelength regions to exclude in the model fits
        param_lims: sequence (optional)
          A sequence of tuples to constrain the model grid by, e.g. [('teff',400,1000,50),('logg',4,5,0.5)] sets the lower limits,
          upper limits, and increments on the 'teff' and 'logg' parameters
        spec_fit: bool
          Fit model grid to spectra
        phot_fit: bool
          Fit model grid to photometry
        data_pickle: object (optional)
          The GetData() object to write new data to
        save: str (optional)
          The directory path to save the plots in
        '''
        if model_fits:
            for model, walkers, steps in model_fits:
                try:

                    # Run MCMC analysis on both the photometry and the spectra separately
                    for desc, dicn, data in filter(None, [
                        ['spec', self.spectra, self.data['SED_spec_app']] if fit_spec else '',
                        ['phot', self.photometry, self.data['SED_phot_app']] if fit_phot else '']):
                        try:
                            print('\nStarting {} {} model fit at {}'.format(model, desc,
                                                                            datetime.datetime.now().strftime(
                                                                                '%Y-%m-%d %I:%M:%S')))
                            model_grid = mcmc_fit.make_model_db(model, model_db_path, grid_data=desc,
                                                                param_lims=param_lims, bands=self.photometry.index,
                                                                fill_holes=False)

                            if not dicn.empty:
                                # Run the MCMC analysis on the data and get errors
                                fit = mcmc_fit.fit_spectrum(data, model_grid, walkers, steps, mask=mask, plot=True,
                                                            outfile=None)
                                fit_data = fit.get_error_and_unc()

                                # Save the marginalized distribution plot
                                if save:
                                    md = plt.gcf()
                                    md.savefig(
                                        save + '{} - {} {} {}_{}_{}.png'.format(self.data['spectral_type'], self.name,
                                                                                desc, model, walkers, steps))
                                    plt.close()

                                # Add the model fit params to the data pickle
                                for idx, suf in enumerate(['_unc_lower', '', '_unc_lower']):
                                    self.data.update({model + '_' + desc + '_' + p + suf: v for p, v in
                                                      zip(fit.all_params, fit_data.T[idx])})

                                # Use maximum uncertainty value as a conservative uncertainty estimate for plotting
                                self.data.update({model + '_' + desc + '_' + p + '_unc': v for p, v in
                                                  zip(fit.all_params,
                                                      [max(i, j) for i, j in zip(fit_data.T[0], fit_data.T[2])])})

                                # Add the interpolated best fit model to the data pickle
                                self.data[model + '_' + desc + '_SED_app'] = norm_to_mags(fit.best_fit_spectrum,
                                                                                          self.app_fluxes)

                                # Flux calibrate the best fit model if possible
                                if self.data['d']:
                                    self.data[model + '_' + desc + '_SED_abs'] = [
                                        self.data[model + '_' + desc + '_SED_app'][0],
                                        self.data[model + '_' + desc + '_SED_app'][1] * self.data['norm']]
                                else:
                                    self.data[model + '_' + desc + '_SED_abs'] = ''

                                # Print the best fit parameters and save the successful fit input
                                u.printer(fit.all_params,
                                          [['{:.2f}-/+({:.2f},{:.2f})'.format(fd[1], fd[0], fd[2]) for fd in fit_data]])
                                self.model_fits += [(model, desc, walkers, steps)]

                                print('Model fit completed at {}\n'.format(
                                    datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S')))

                        except IOError:
                            print('Model fit failed at {}\n'.format(
                                datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S')))

                except IOError:
                    print("Could not perform model fits to {}.".format(model))

            # Add the model fit data to the data object
            if data_pickle: data_pickle.add_source(self.data, self.name, update=True)

        else:
            print("Please specify a sequence of all (model_grid,walkers,steps) to run.")

    def complete_with_model_fits(self, data_pickle=''):
        """
        Create an SED filling in the gaps with an averaged model spectrum and recalculate all fundamental parameters

        Parameters
        ----------
        data_pickle: object (optional)
          The GetData() object to write new data to

        """
        (W, F, E), RJ, units = self.data['SED_spec_app'], self.data['RJ'], [q.um, q.erg / q.s / q.cm ** 2 / q.AA,
                                                                            q.erg / q.s / q.cm ** 2 / q.AA]

        if self.model_fits:
            try:
                # Make SED with averaged model fits
                modelSEDs = []
                for model, desc, walkers, steps in self.model_fits:
                    try:
                        # Normalize the best fit model spectrum to the apparent observational spectrum
                        mW, mF, mE = self.data[model + '_' + desc + '_SED_app']

                        # Create full SED from model Wien tail, spectrum, model MIR, and Rayleigh-Jeans tail
                        model_app = [np.concatenate(
                            [m[mW < W[0]].value, S.value, m[mW > W[-1]].value, B[RJ[0] > mW[-1]].value]) * Q for
                                     m, Q, S, B in zip([mW, mF, mE], units, [W, F, E], RJ)]
                        model_app = finalize_spec(model_app)
                        modelSEDs.append(model_app)
                    except IOError:
                        pass

                # Make mean SED_app from all spectrum+model SEDs
                model_SED_W = modelSEDs[0][0]
                if len(modelSEDs) > 1:
                    # Interpolate all model SEDs to same wavelength range
                    model_SED_F = [modelSEDs[0][1]] + [np.interp(model_SED_W, w, f) * q.erg / q.s / q.cm ** 2 / q.AA for
                                                       w, f, e in modelSEDs[1:]]
                    model_SED_E = [modelSEDs[0][2]] + [np.interp(model_SED_W, w, e) * q.erg / q.s / q.cm ** 2 / q.AA for
                                                       w, f, e in modelSEDs[1:]]

                    # Use mean flux and max and min uncertainty at each wavelength point
                    model_SED_E = np.asarray([max(n) - min(n) for n in
                                              zip(*[i.value for i in model_SED_F])]) * q.erg / q.s / q.cm ** 2 / q.AA
                    model_SED_F = np.asarray(np.add(*model_SED_F) / len(model_SED_F)) * q.erg / q.s / q.cm ** 2 / q.AA
                else:
                    model_SED_F, model_SED_E = modelSEDs[0][1:]

                # Create new SED and flux calibrate if possible
                self.data['m_SED_app'] = [model_SED_W, model_SED_F, model_SED_E]
                if self.data['d']:
                    self.data['m_SED_abs'] = [model_SED_W, model_SED_F * self.data['norm'],
                                              model_SED_E * self.data['norm']]
                else:
                    self.data['m_SED_abs'] = ''

                # Calculate fundamental parameters if model atmospheres are used to fill in the gaps
                self.data = fundamental_params(self.data, p='m_')

            except IOError:
                self.data['m_SED_app'], self.data['m_SED_abs'] = '', ''

        if self.data['d']:
            if print_fits: u.printer(['Lbol', 'Teff', 'SpT', 'R_Jup', 'M_Jup', 'logg', 'Age', 'Dist', 'Binary'], [
                ['{}({})'.format(self.data['m_Lbol'], self.data['m_Lbol_unc']),
                 '-' if binary else '{}({})'.format(self.data['m_teff'], self.data['m_teff_unc']),
                 '{}/{}'.format(opt_spec_type, ir_spec_type),
                 '-' if binary else '{}({})'.format(self.data['m_radius'], self.data['m_radius_unc']),
                 '-' if binary else '{}({})'.format(self.data['m_mass'], self.data['m_mass_unc']),
                 '-' if binary else '{}({})'.format(self.data['m_logg'], self.data['m_logg_unc']),
                 '{}-{}'.format(self.data['age_min'], self.data['age_max']),
                 '{}({})'.format(self.data['d'].value, self.data['d_unc'].value), 'Yes' if binary else '-']],
                                     empties=True, title='Using Model Fits')

        # Add the model fit data to the data object
        if data_pickle: data_pickle.add_source(self.data, self.name, update=True)

    def plot(self, Flam=False, app=False, photometry=True, spectra=False, composites=True, \
             models=True, integrals=False, model_syn_photometry=True, model_integrals=False, \
             figsize=(12, 8), xaxis='', yaxis='', scale=['log', 'log'], legend=True, overplot=False, \
             zorder=0, colors=['k', 'k', 'k'], save=''):
        '''
        Plot the SED

        Parameters
        ----------
        Flam: bool
          Plots the SED in units of erg/s/cm2 instead of erg/s/cm2/A
        app: bool
          Plot the apparent SED even if it can be flux calibrated
        photometry: bool
          Plot the photometry
        spectra: bool
          Plot the spectra
        models: bool
          Plot the atmospheric model fits
        integrals: bool
          Plot the integral surface used to calculate Lbol
        syn_photometry: bool
          Plot the synthetic photometry calculated from the spectra
        model_syn_photometry: bool or 'all'
          Plot the synthetic magnitudes from the model spectra that corespond to real photometry. Plot all synthetic mags if 'all'.
        model_integrals: bool
          Plot the SED completed with the averaged best fit models instead of linear interpolation
        save: str (optional)
          The path to save the image to
        figsize: sequence (optional)
          The (x,y) dimensions of the image
        xaxis: sequence (optional)
          The (x_min,x_max) range to plot
        yaxis: sequence (optional)
          The (y_min,y_max) range to plot
        scale: sequence (optional)
          The (x,y) axis scales to plot, e.g. ['log','linear']
        legend: bool
          Plot the legend
        overplot: bool or plt.axis object
          Plot the SED on an existing plot or specified axis instead of creating a new figure
        zorder: int
          The zorder of the plot
        colors: sequence
          The colors to use in the plot

        Returns
        -------
        None
        '''
        # Draw the figure and load the axes
        plt.rc('text', usetex=True), plt.rc('text', fontsize=22)
        if not overplot: fig = plt.figure(figsize=figsize)
        ax = overplot if hasattr(overplot, 'figure') else plt.gca()

        # Choose apparent or absolute flux calibration
        pre, suf = ['m', 'app'] if app or not self.data['d'] else ['M', 'abs']

        # Convert to lambda*F_lambda
        def lam(l, idx=1):
            return (l[idx] * l[0] * (q.um * q.erg / q.s / q.cm ** 2 / q.AA if not hasattr(l[0], 'unit') else 1.)).to(
                q.erg / q.s / q.cm ** 2).value \
                if Flam else l[idx].value if hasattr(l[0], 'unit') else l[idx]

        # Plot the spectra
        if spectra:
            for (spec_id, spec) in self.spectra.iterrows():
                plt.step(spec['wavelength'], lam(spec['flux_' + suf]), where='mid', color=colors[0], alpha=0.9)

        # Plot the composite spectra
        if composites:
            for (n, spec) in self.composites.iterrows():
                plt.step(spec['wavelength'], spec['flux_' + suf], where='mid', color=colors[1])

        # Plot the surfaces in integration
        if integrals:
            # Plot the SED with linear interpolation completion
            plt.plot(self.data['SED_' + suf][0].value, lam(self.data['SED_' + suf]), color='k', alpha=0.5, ls='--')
            plt.fill_between(self.data['SED_' + suf][0].value,
                             lam(self.data['SED_' + suf]) - lam(self.data['SED_' + suf], idx=2), \
                             lam(self.data['SED_' + suf]) + lam(self.data['SED_' + suf], idx=2), color='k', alpha=0.1)

            # Plot the SED with model atmosphere completion
            if self.data.get('m_SED_' + suf):
                plt.step(self.data['m_SED_' + suf][0].value, lam(self.data['m_SED_' + suf]), where='mid')
                plt.fill_between(self.data['m_SED_' + suf][0].value,
                                 lam(self.data['m_SED_' + suf]) - lam(self.data['m_SED_' + suf], idx=2), \
                                 lam(self.data['m_SED_' + suf]) + lam(self.data['m_SED_' + suf], idx=2), color='k',
                                 alpha=0.1)

        # Plot the photometry
        if photometry:

            # Observational photometry with uncertainties
            w, f, e = df_extract(self.photometry[(self.photometry[pre + '_flux_unc'] != '') & (
                type(self.photometry[pre + '_flux_unc']) != str)], ['eff', pre + '_flux', pre + '_flux_unc'])
            ax.errorbar(w, lam([w, f, e]), yerr=lam([w, f, e], idx=2), fmt='o', color=colors[0], markeredgecolor='k',
                        markeredgewidth=1, markersize=10, zorder=zorder + 10, capsize=3)

            # Observational photometry upper limits
            w, f = df_extract(self.photometry[(self.photometry[pre + '_flux_unc'] == '')], ['eff', pre + '_flux'])
            if w: ax.errorbar(w, lam([w, f, e]), fmt='v', color=colors[0], markeredgecolor='k', markeredgewidth=1,
                              lolims=[True] * len(w), markersize=10, zorder=zorder + 10, capsize=3)

            # Photometry estimated from mag-mag relations
            w, f, e = df_extract(self.photometry[(self.photometry[pre + '_flux_unc'] != '') & (
                type(self.photometry[pre + '_flux_unc']) == str)], ['eff', pre + '_flux', pre + '_flux_unc'])
            if w: ax.errorbar(w, lam([w, f, e]), yerr=lam([w, f, e], idx=2), fmt='o', color='w', markeredgecolor='k',
                              markeredgewidth=1, markersize=10, zorder=zorder, capsize=3)

        # Plot the best fit model spectrum inferred from fits
        if models:
            for model in self.model_fits:
                # Plot the model best fit to the spectra
                if model[1] == 'spec':
                    plt.step(self.data[model[0] + '_spec_SED_' + suf][0].value,
                             lam(self.data[model[0] + '_spec_SED_' + suf]), alpha=0.5, where='mid')

                # Plot the model best fit to the photometry
                if model[1] == 'phot' and model_syn_photometry:
                    plt.scatter(self.data[model[0] + '_phot_SED_' + suf][0],
                                lam(self.data[model[0] + '_phot_SED_' + suf]), marker='s', alpha=0.6)

        # Format the axes
        ax.set_xlim(xaxis or (0.3, 30)), ax.set_xscale(scale[0]), ax.set_yscale(scale[1], nonposy='clip')
        if yaxis: ax.set_ylim(yaxis)
        ax.set_xlabel(r"$\displaystyle\lambda\mbox{ (}\mu\mbox{m)}$", labelpad=10)
        ax.set_ylabel(
            r"$\displaystyle" + ('\lambda' if Flam else '') + " F_\lambda\mbox{ (erg s}^{-1}\mbox{ cm}^{-2}" + (
                '' if Flam else '\mbox{ A}^{-1}') + ")$", labelpad=20)
        if legend: ax.legend(loc=8, frameon=False, fontsize=16,
                             title='({}) {}'.format(self.data['spectral_type'], self.name))
        plt.ion()

        # Save the image, then close it
        if save: plt.savefig(save if save.endswith('.png') else '{}{} - {}.png'.format(save, self.data['spectral_type'],
                                                                                       self.name)), plt.close()

    def write(self, dirpath, app=False, spec=True, phot=False):
        """
        Exports a file of photometry and a file of the composite spectra with minimal data headers

        Parameters
        ----------
        dirpath: str
          The directory path to place the file
        app: bool
          Write apparent SED data
        spec: bool
          Write a file for the spectra with wavelength, flux and uncertainty columns
        phot: bool
          Write a file for the photometry with

        """
        if spec:
            try:
                sed = self.data['SED_spec_' + ('app' if app else 'abs')]
                if not dirpath.endswith('.txt'): specpath = dirpath + '{} ({}) SED.txt'.format(self.data['shortname'],
                                                                                               self.data[
                                                                                                   'spectral_type'])
                header = '{} {} spectrum (erg/s/cm2/A) as a function of wavelength (um)'.format(self.name,
                                                                                                'apparent' if app else 'flux calibrated')
                np.savetxt(specpath, np.asarray(sed).T, header=header)
            except:
                print("Couldn't print spectra.")

        if phot:
            try:
                phot = np.asarray([np.asarray([i.value if hasattr(i, 'unit') else i for i in j]) for j in
                                   self.photometry.reset_index()[['band', 'eff', 'm_flux' if app else 'M_flux',
                                                                  'm_flux_unc' if app else 'M_flux_unc']].values])
                if not dirpath.endswith('.txt'): photpath = dirpath + '{} ({}) phot.txt'.format(self.data['shortname'],
                                                                                                self.data[
                                                                                                    'spectral_type'])
                header = '{} {} spectrum (erg/s/cm2/A) as a function of wavelength (um)'.format(self.name,
                                                                                                'apparent' if app else 'flux calibrated')
                np.savetxt(photpath, phot, header=header, fmt='%s')
            except IOError:
                print("Couldn't print photometry.")


def fundamental_params(D, p='', plot=False):
    '''
    Calculates all possible fundamental parameters given a dictionary of data

    Parameters
    ----------
    D: dict
    A dictionary containing the object's SED and (optionally) distance and radius
    p: str
    A prefix for the new dictionary keys
    plot: bool
    Plot the model isochrones with the ranges estimated for this object

    Returns
    -------
    D: dict
    The input dictionary updated with fundamental parameter key/value pairs

    '''
    try:

        # Calculate fbol, mbol
        D[p + 'fbol'], D[p + 'fbol_unc'] = (np.trapz(D[p + 'SED_app'][1], x=D[p + 'SED_app'][0])).to(
            q.erg / q.s / q.cm ** 2), np.sqrt(
            np.sum((D[p + 'SED_app'][2] * np.gradient(D[p + 'SED_app'][0])).to(q.erg / q.s / q.cm ** 2) ** 2))
        D[p + 'mbol'], D[p + 'mbol_unc'] = -2.5 * np.log10(D[p + 'fbol'].value) - 11.482, (2.5 / np.log(10)) * (
            D[p + 'fbol_unc'] / D[p + 'fbol']).value

        # if D.get('d') and D.get(p+'mbol')>5:
        if D.get('d'):
            # Calculate Mbol, Lbol in solar units, and Lbol in Watts
            D[p + 'Mbol'], D[p + 'Mbol_unc'] = D[p + 'mbol'] - 5 * np.log10((D['d'] / 10 * q.pc).value), np.sqrt(
                D[p + 'mbol_unc'] ** 2 + ((2.5 / np.log(10)) * (D['d_unc'] / D['d']).value) ** 2)
            D[p + 'Lbol'], D[p + 'Lbol_unc'] = get_Lbol(D[p + 'SED_app'], D['d'], sig_d=D['d_unc'], solar_units=True)
            D[p + 'Lbol_W'], D[p + 'Lbol_W_unc'] = get_Lbol(D[p + 'SED_app'], D['d'], sig_d=D['d_unc'])

            if D.get('binary'):
                # Can't calculate these if it's a binary!
                for k in ['teff', 'mass', 'radius', 'logg']: D[p + k], D[p + k + '_unc'] = '', ''

            else:
                # Get radius from *radius* argument or radius interpolation of evolutionary model isochrones, then calculate Teff
                D[p + 'radius'], D[p + 'radius_unc'] = (D['radius'], D['radius_unc']) if D[
                                                                                             'radius'] != '' else isochrone_interp(
                    D[p + 'Lbol'], D[p + 'Lbol_unc'], D['age_min'], D['age_max'], plot=plot)
                D[p + 'teff'], D[p + 'teff_unc'] = get_teff(D[p + 'Lbol_W'], D[p + 'Lbol_W_unc'], D[p + 'radius'],
                                                            D[p + 'radius_unc'])

                # Also calculate model mass and logg
                D[p + 'logg'], D[p + 'logg_unc'] = isochrone_interp(D[p + 'Lbol'], D[p + 'Lbol_unc'], D['age_min'],
                                                                    D['age_max'], yparam='logg', plot=plot)
                D[p + 'mass'], D[p + 'mass_unc'] = isochrone_interp(D[p + 'Lbol'], D[p + 'Lbol_unc'], D['age_min'],
                                                                    D['age_max'], yparam='mass', plot=plot)

        else:
            pass

    except IOError:
        pass

    return D


def df_extract(df, keys):
    '''
    Turns a pandas DataFrame into a list of arrays

    Parameters
    ----------
    df: DataFrame
    The DataFrame to be converted
    keys: list
    The list of keys to extract from the DataFrame, sorted by the first element

    Returns
    -------
    new_format: list or dict
    A list of arrays
    '''
    new_format = [np.array([i.value if hasattr(i, 'unit') else i for i in df[keys].sort(keys[0])[l].values]) for l in
                  keys]
    return new_format


def norm_to_mags(spec, to_mags, weighting=True, reverse=False, plot=False):
    '''
    Normalize the given spectrum to the given dictionary of magnitudes

    Parameters
    ----------
    spec: sequence
    The [W,F,E] to be normalized
    to_mags: dict
    The dictionary of magnitudes to normalize to, e.g {'W2':12.3, 'W2_unc':0.2, ...}

    Returns
    -------
    spec: sequence
    The normalized [W,F,E]
    '''
    spec = u.unc(spec)
    spec = [spec[0] * (q.um if not hasattr(spec[0], 'unit') else 1.), \
            spec[1] * (q.erg / q.s / q.cm ** 2 / q.AA if not hasattr(spec[1], 'unit') else 1.), \
            spec[2] * (q.erg / q.s / q.cm ** 2 / q.AA if not hasattr(spec[2], 'unit') else 1.)]

    # Force JHK coverage if close enough
    blue, red = spec[0][0], spec[0][-1]

    # Blue side of spectrum
    if blue > 1.08 * q.um and blue < 1.12 * q.um:
        spec[0][0] *= 1.08 / blue.value
    elif blue > 1.47 * q.um and blue < 1.55 * q.um:
        spec[0][0] *= 1.47 / blue.value
    elif blue > 1.95 * q.um and blue < 2.00 * q.um:
        spec[0][0] *= 1.95 / blue.value
    else:
        pass

    # Red side of spectrum
    if red > 1.3 * q.um and red < 1.41 * q.um:
        spec[0][-1] *= 1.41 / red.value
    elif red > 1.76 * q.um and red < 1.825 * q.um:
        spec[0][-1] *= 1.825 / red.value
    elif red > 2.3 * q.um and red < 2.356 * q.um:
        spec[0][-1] *= 2.356 / red.value
    else:
        pass

    # Calculate all synthetic magnitudes for flux calibration then fix end points if necessary
    mags = s.all_mags(spec,
                      bands=[b for b in to_mags if to_mags.get(b) and to_mags.get(b + '_unc') and b in RSR.keys()], \
                      Flam=False, to_flux=True, photon=False)

    # Return red and blue wavelength positions to original values
    spec[0][0], spec[0][-1] = blue, red

    try:
        # Get list of all bands in common and pull out flux values
        bands, data = [b for b in list(set(mags).intersection(set(to_mags))) if '_unc' not in b], []
        for b in bands:
            if all([mags.get(b), mags.get(b + '_unc'), to_mags.get(b), to_mags.get(b + '_unc')]):
                data.append([RSR[b]['eff'].value, mags[b].value if hasattr(mags[b], 'unit') else mags[b], \
                             mags[b + '_unc'].value if hasattr(mags[b + '_unc'], 'unit') else mags[b + '_unc'], \
                             to_mags[b].value if hasattr(to_mags[b], 'unit') else to_mags[b], \
                             to_mags[b + '_unc'].value if hasattr(to_mags[b + '_unc'], 'unit') else to_mags[b + '_unc'], \
                             (RSR[b]['max'] - RSR[b]['min']).value if weighting else 1.])

        # Make arrays of values and calculate normalization factor that minimizes the function
        w, f2, e2, f1, e1, weight = [np.array(i, np.float) for i in np.array(data).T]
        norm = sum(weight * f1 * f2 / (e1 ** 2 + e2 ** 2)) / sum(weight * f2 ** 2 / (e1 ** 2 + e2 ** 2))

        # Plotting test
        if plot:
            plt.loglog(spec[0].value, spec[1].value, label='old', color='g')
            plt.loglog(spec[0].value, spec[1].value * norm, label='new', color='b')
            plt.scatter(w, f1, c='g')
            plt.scatter(w, f2, c='b')
            plt.legend()

        return [spec[0], spec[1] / norm, spec[2] / norm] if reverse else [spec[0], spec[1] * norm, spec[2] * norm]

    except:
        print('No overlapping photometry for normalization!')
        return spec


def polynomial_relation(xparam, yparam, polynomial={}, pickle_path=package + '/Data/Pickles/polynomial_relations.p'):
    """
    Store or retrieve a polynomial relation for this GetData() instance.

    Parameters
    ----------
    xparam: str
    The x parameter of the relation, e.g. 'MKO_J-MKO_K', 'SpT'
    yparam: str
    The y parameter of the relation, e.g. 'M_MKO_J', 'Lbol'
    relation: dict
    A dictionary of a polynomial to write to the pickle.
    Must contain the keys 'rms', 'min', 'max', and 'c0'.
    Additional orders should be called 'c1', 'c2', etc.
    pickle: str
    The path to the pickle

    """
    D = pickle.load(open(pickle_path, 'rb'))

    if polynomial:
        if all([k in polynomial for k in ['rms', 'min', 'max', 'c0']]):
            if not D.get(yparam):
                D[yparam] = {}
            D[yparam][xparam] = polynomial
            pickle.dump(D, open(pickle_path, 'wb'))
            print('Polynomial of {} as a function of {} saved to {}.\n'.format(yparam, xparam, pickle_path))
        else:
            print("Polynomial relation must have keys 'rms','min','max', and 'c0'.")

    else:
        try:
            return D[yparam][xparam]
        except:
            print('No polynomial of {} as a function of {}.'.format(yparam, xparam))


def finalize_spec(spec):
    '''
  Sort by wavelength and remove nans, negatives and zeroes

  Parameters
  ----------
  spec: sequence
    The [W,F,E] to be cleaned up

  Returns
  -------
  spec: sequence
    The cleaned and ordered [W,F,E]
  '''
    spec = list(zip(*sorted(zip(*map(list, [[i.value if hasattr(i, 'unit') else i for i in j] for j in spec])), key=lambda x: x[0])))
    return u.scrub([spec[0]*q.um, spec[1]*q.erg/q.s/q.cm**2/q.AA, spec[2]*q.erg/q.s/q.cm**2/q.AA])


def SpT_relations(L, yparam, pop=[], identify=[], colors=False, ylabel='', inverty=False):
    """
  Make nice dual plot of spectral type vs. Teff for field and young objects
  """
    F, a, b = u.multiplot(1, 2, figsize=(12, 6), fontsize=18)
    pop += ['2MASS J16262034+3925190', 'WISEPA J164715.59+563208.2', 'TWA 27B', 'WISEA J182831.08+265037.6',
            '0718-6415', '0556-0927', '0619-2127']
    ymin = -7.5 if yparam == 'Lbol' else 300

    # Field Sequence
    L.mag_plot('SpT', yparam, overplot=a, fit=[(['fld'], 6, 'k', '-')], pop=pop, identify=identify, inverty=inverty,
               legend=True, groups=['fld'], xlabel=r'$\mbox{Spectral Type}$',
               add_text=('(a) Field Age Sequence', (7, ymin), (7, ymin), 16), weighting=True)

    # Young Sequence
    L.mag_plot('SpT', yparam, overplot=b, fit=[(['fld'], 6, 'k', '-')], pop=pop, verbose=True, identify=identify,
               inverty=inverty, legend=False, xlabel=r'$\mbox{Spectral Type}$',
               add_text=('(b) Low Gravity Objects\nwith Field Age Sequence', (7, ymin), (7, ymin), 16), weighting=True,
               plot_field=False)
    u.manual_legend([r'$\beta /\gamma$', 'NYMG'], ['w', '0.5'], overplot=b, markers=['o', 'o'], sizes=[8, 8],
                    edges=['0.5', 'k'], errors=[True, True], ncol=1)

    a.set_ylabel(ylabel or yparam, fontsize=28, labelpad=8)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.98), plt.draw()


def M_v_Lbol(L, pct_lim=35, pop=[]):
    F, ax, bx, cx = u.multiplot(1, 3, figsize=(12, 6), fontsize=22, hspace=0.5, sharex=False, sharey=True)
    colors = ['#FFA821', 'r', '#2B89D6']  # ,'#7F00FF']
    pop += ['2MASS J16262034+3925190', 'WISEPA J164715.59+563208.2', 'TWA 27B', 'WISEA J182831.08+265037.6',
            '0718-6415', '0556-0927', '0619-2127', 'HR8799b', 'HR8799c']

    # J band ==============================================================================================================
    band, coords = 'M_J', (17.5, -2.9)
    L.mag_plot(band, 'Lbol', pct_lim=pct_lim, spt=['M', 'L'], overplot=ax, fit=[(['fld'], 3, 'k', '-')],
               plot_field=False, weighting=False, pop=pop, legend=False)
    L.mag_plot(band, 'Lbol', pct_lim=pct_lim, spt=['T'], overplot=ax, fit=[(['fld'], 3, 'k', '-')], plot_field=False,
               weighting=False, pop=pop, legend=False)
    ax.set_xlim((8.1, 17.5)), ax.set_ylabel(r'$L_{bol}$', fontsize=28, labelpad=8), ax.set_xlabel(
        r'$M_{' + band.replace('M_', '') + r'}$', fontsize=30, labelpad=8)

    # Ks band ==============================================================================================================
    band, coords = 'M_Ks', (17, -2.9)
    L.mag_plot(band, 'Lbol', pct_lim=pct_lim, overplot=bx, fit=[(['fld'], 3, 'k', '-')], plot_field=False,
               weighting=True, pop=pop, legend=False)
    bx.set_xlim((7, 17.5)), bx.set_xlabel(r'$M_{' + band.replace('M_', '') + r'}$', fontsize=28, labelpad=8)

    # W2 band ==============================================================================================================
    band, coords = 'M_W2', (13.5, -2.9)
    L.mag_plot(band, 'Lbol', pct_lim=pct_lim, overplot=cx, fit=[(['fld'], 5, 'k', '-')], plot_field=False,
               xmaglimits=(8.82, 14.3), weighting=True, pop=pop, legend=False)
    cx.set_xlabel(r'$M_{' + band.replace('M_', '') + r'}$', fontsize=28, labelpad=8), cx.set_xlim(
        (8.5, 14.5)), cx.set_ylim((-5.9, -2.3))

    u.manual_legend(['M', 'L', 'T'], colors, sizes=[8, 8, 8], overplot=bx, markers=['o', 'o', 'o'], edges=colors,
                    errors=[False, False, False], styles=['p', 'p', 'p'], ncol=1, loc=0)
    u.manual_legend(['Field Sequence', r'$\beta /\gamma$', 'NYMG'], ['k', 'w', '0.5'], sizes=[2, 8, 8], overplot=cx,
                    markers=['-', 'o', 'o'], edges=['k', '0.5', '0.3'], errors=[False, True, True],
                    styles=['l', 'p', 'p'], ncol=1, loc=0)

    plt.subplots_adjust(right=0.98, top=0.98, bottom=0.16, left=0.11, hspace=0)
