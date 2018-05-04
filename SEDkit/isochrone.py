#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
import os
import numpy as np
from glob import glob
from pkg_resources import resource_filename
import astropy.units as q
from itertools import chain, groupby
import scipy.interpolate as si


def DMEstar(ages, xparam='Lbol', yparam='radius'):
    """
    Retrieve the DMEstar isochrones

    Parameters
    ----------
    ages: sequence
        The ages at which the isochrones should be evaluated
    xparam: str

    """
    D, data = [], glob(resource_filename('SEDkit', 'data/models/evolutionary/DMESTAR/*.txt'))
    for f in data:
        age = int(os.path.basename(f).split('_')[1][:-5]) / 1000.
        if age in ages:
            mass, teff, Lbol, logg, radius = np.genfromtxt(f, usecols=(1, 3, 4, 2, 5), unpack=True)
            teff, radius, mass = 10 ** teff, 9.72847 * 10 ** radius, 1047.2 * mass
            D.append([age,
                      mass if xparam == 'mass' else logg if xparam == 'logg' else radius if xparam == 'radius' else Lbol,
                      mass if yparam == 'mass' else logg if yparam == 'logg' else radius if yparam == 'radius' else Lbol])
    return D
    
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


def isochrone_interp(z, age, xparam='Lbol', yparam='radius', xlabel='', ylabel='', xlims='',
                     ylims='', evo_model='hybrid_solar_age', plot=False, ages=[0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 10],
                     title=''):
    """
    Interpolates the model isochrones to obtain a range in y given a range in x
    """
    min_age = (age[0]-age[1]).to(q.Gyr).value
    max_age = (age[0]+age[1]).to(q.Gyr).value
    
    if max_age>10:
        print('Cannot interpolate past 10Gyr.')
        return
    
    if hasattr(z[0], 'unit'):
        z, z_unc = z
        z = z.value
        z_unc = z_unc.value
    else:
        z, z_unc = z
    
    # Grab and plot the desired isochrones
    D = isochrones(evo_model=evo_model, xparam=xparam, yparam=yparam, ages=ages, plot=plot)
    Q = {d[0]: {'x': d[1], 'y': d[2]} for d in D}

    # Pull out isochrones which lie just above and below *min_age* and *max_age*
    A = np.array(list(zip(*D))[0])
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

# def isochrone_interp(z, z_unc, min_age, max_age, xparam='Lbol', yparam='radius', xlabel='', ylabel='', xlims='',
#                      ylims='', evo_model='hybrid_solar_age', plot=False, ages=[0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 10],
#                      title=''):
#     """
#     Interpolates the model isochrones to obtain a range in y given a range in x
#     """
#
#     # Grab and plot the desired isochrones
#     D = isochrones(evo_model=evo_model, xparam=xparam, yparam=yparam, ages=ages, plot=plot)
#     Q = {d[0]: {'x': d[1], 'y': d[2]} for d in D}
#
#     # Convert to Gyr in necessary
#     if max_age > 10: min_age, max_age = min_age / 1000., max_age / 1000.
#
#     # Pull out isochrones which lie just above and below *min_age* and *max_age*
#     A = np.array(zip(*D)[0])
#     min1, min2, max1, max2 = A[A <= min_age][-1] if min_age > 0.01 else 0.01, A[A >= min_age][0], A[A <= max_age][-1], \
#                              A[A >= max_age][0]
#
#     # Create a high-res x-axis in region of interest and interpolate isochrones horizontally onto new x-axis
#     x = np.linspace(z - z_unc, z + z_unc, 20)
#     for k, v in Q.items(): v['x'], v['y'] = x, np.interp(x, v['x'], v['y'])
#
#     # Create isochrones interpolated vertically to *min_age* and *max_age*
#     min_iso, max_iso = [np.interp(min_age, [min1, min2], [r1, r2]) for r1, r2 in zip(Q[min1]['y'], Q[min2]['y'])], [
#         np.interp(max_age, [max1, max2], [r1, r2]) for r1, r2 in zip(Q[max1]['y'], Q[max2]['y'])]
#
#     # Pull out least and greatest y value of interpolated isochrones in x range of interest
#     y_min, y_max = min(min_iso + max_iso), max(min_iso + max_iso)
#
#     if plot:
#         ax = plt.gca()
#         ax.set_ylabel(r'${}$'.format(ylabel or yparam), fontsize=22, labelpad=5), ax.set_xlabel(
#             r'${}$'.format(xlabel or xparam), fontsize=22, labelpad=15), plt.grid(True, which='both'), plt.title(
#             evo_model.replace('_', '-') if title else '')
#         # ax.axvline(x=z-z_unc, ls='-', color='0.7', zorder=-3), ax.axvline(x=z+z_unc, ls='-', color='0.7', zorder=-3)
#         ax.add_patch(plt.Rectangle((z - z_unc, 0), 2 * z_unc, 10, color='0.7', zorder=-3))
#         xlims, ylims = ax.get_xlim(), ax.get_ylim()
#         ax.fill_between([-100, 100], y_min, y_max, color='#99e6ff', zorder=-3)
#         # plt.plot(x, min_iso, ls='--', color='r'), plt.plot(x, max_iso, ls='--', color='r')
#         plt.xlim(xlims), plt.ylim(ylims)
#
#     return [round(np.mean([y_min, y_max]), 2), round(abs(y_min - np.mean([y_min, y_max])), 2)]


def isochrones(evo_model='hybrid_solar_age', xparam='Lbol', yparam='radius',
               ages=[0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 10], plot=False, overplot=False):
    if plot:
        if overplot:
            ax = plt.gca()
        else:
            fig = plt.figure()
            ax = plt.subplot(111)
    DME = DMEstar(ages, xparam=xparam, yparam=yparam)
    evo_file = resource_filename('SEDkit', 'data/models/evolutionary/{}.txt'.format(evo_model))
    D = []
    # data = [d for d in np.genfromtxt(evo_file, delimiter=',', usecols=range(6)) if d[0] in ages and d[0] in zip(*DME)[0]]
    data = [d for d in np.genfromtxt(evo_file, delimiter=',', usecols=range(6)) if d[0] in ages and d[0] in list(zip(*DME))[0]]

    for k, g in groupby(data, key=lambda y: y[0]):
        age, mass, teff, Lbol, logg, radius = [np.array(i) for i in list(zip(*[list(i) for i in list(g)]))[:6]]
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