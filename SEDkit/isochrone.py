#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
A module to estimate fundamental parameters from model isochrones
"""
import os
import numpy as np
from glob import glob
from pkg_resources import resource_filename
import astropy.units as q
from itertools import chain, groupby
import scipy.interpolate as si
from bokeh.plotting import figure, show
from bokeh.models import Range1d
from .spectrum import COLORS


# A dictionary of all supported moving group ages
NYMG_AGES = {'TW Hya': (14*q.Myr, 6*q.Myr), 'beta Pic': (17*q.Myr, 5*q.Myr),
             'Tuc-Hor': (25*q.Myr, 15*q.Myr), 'Columba': (25*q.Myr, 15*q.Myr),
             'Carina': (25*q.Myr, 15*q.Myr), 'Argus': (40*q.Myr, 10*q.Myr),
             'AB Dor': (85*q.Myr, 35*q.Myr), 'Pleiades': (120*q.Myr, 10*q.Myr)}


def avg_param(yparam, z, z_unc, min_age, max_age, spt, xparam='Lbol', plot=False):
    """
    Get the average parameter value across multiple evolutionary models

    Parameters
    ----------
    yparam: str
        The y-axis value
    z: float
        The value to interpolate to
    z_unc: float
        The uncertainty
    min_age: float
        The lower age limit in [Gyr]
    max_age: float
        The upper age limit in [Gyr]
    spt: float
        The numeric spectral type, 0-29 for M0-T9 dwarfs
    xparam: str
        The x-axis value
    plot: bool
        Plot the isochrone

    Returns
    -------
    tuple
        The average interpolate value
    """
    # Get the late type models
    late_models = ['nc_solar_age', 'COND03' if z > -5.1 else None]
    late_models = (filter(None, late_models) if spt > 17 else [])

    # Get the even later type models
    later_models = ['f2_solar_age', 'DUSTY00' if z > -5.1 else None]
    later_models = (filter(None, later_models) if spt < 23 else [])

    # Get the final list
    models = ['hybrid_solar_age'] + late_models + later_models

    # Get interpolated values forom all models
    results = []
    for m in models:
        result = isochrone_interp(z, z_unc, min_age, max_age, xparam=xparam,
                                  yparam=yparam, evo_model=m, plot=plot)
        results.append(result)

    # Make into arrays
    x, sig_x = [np.array(i) for i in zip(*results)]

    # Convert (value, unc) to (max, min) range
    min_x = min(x - sig_x)
    max_x = max(x + sig_x)
    X, X_unc = [(max_x + min_x) / 2., (max_x - min_x) / 2.]

    # # Plot it
    # if plot:
    #     plt.axhline(y=X, color='k', ls='-', lw=2)
    #     plt.axhline(y=X - X_unc, color='k', ls='--', lw=2)
    #     plt.axhline(y=X + X_unc, color='k', ls='--', lw=2)

    return [X, X_unc]


def DMEstar(ages, xparam='Lbol', yparam='radius', jupiter=False):
    """
    Retrieve the DMEstar isochrones

    Parameters
    ----------
    ages: sequence
        The ages at which the isochrones should be evaluated
    xparam: str
        The x-axis parameter
    yapram: str
        The y-axis parameter
    jupiter: bool
        Use Jupiter units instead of solar units

    Returns
    -------
    list
        The isochrones for the given ages
    """
    # Fetch the data
    path_str = 'data/models/evolutionary/DMESTAR/*.txt'
    data = glob(resource_filename('SEDkit', path_str))

    # Add them to the list
    D = []
    for f in data:

        # Convert age to Gyr
        age = int(os.path.basename(f).split('_')[1][:-5]) / 1000.

        if age in ages:

            # Get the data
            dat = np.genfromtxt(f, usecols=(1, 3, 4, 2, 5), unpack=True)
            mass, teff, Lbol, logg, radius = dat

            # Convert to linear
            teff = 10**teff
            radius = 10**radius

            # Jupiuter units if necessary
            if jupiter:
                radius *= 9.72847
                mass *= 1047.2

            # Add to the list
            xp = mass if xparam == 'mass' else logg if xparam == 'logg' else radius if xparam == 'radius' else teff if xparam == 'teff' else Lbol
            yp = mass if yparam == 'mass' else logg if yparam == 'logg' else radius if yparam == 'radius' else teff if yparam == 'teff' else Lbol
            D.append([age, xp, yp])

    return D


def isochrone_interp(xval, age, xparam='Lbol', yparam='radius', jupiter=False,
                     ages=[0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 10],
                     xlabel=None, ylabel=None, xlims=None, ylims=None,
                     evo_model='hybrid_solar_age', title=None, plot=False):
    """
    Interpolates the model isochrones to obtain a range in y given an age range

    Parameters
    ----------
    xval: sequence
        The (value, uncertainty) of the xparam to interpolate to
    age: sequence
        The (min_age, max_age) in astropy units to interpolate to
    xparam: str
        The x-axis parameter
    yapram: str
        The y-axis parameter
    ages: sequence
        The ages at which the isochrones should be evaluated
    xlabel: str
        The x-axis label for the plot
    ylabel: str
        The y-axis label for the plot
    xlims: sequence
        The x-axis limits for the plot
    ylims: sequence
        The y-axis limits for the plot
    evo_model: str
        The name of the evolutionary model
    title: str
        The plot title
    plot: bool
        Plot the figure

    Returns
    -------
    tuple
        The value and uncertainty of the interpolated value
    """
    # Convert (age, unc) into age range
    min_age = (age[0]-age[1]).to(q.Gyr).value
    max_age = (age[0]+age[1]).to(q.Gyr).value

    if max_age > 10 or min_age < 0.01:
        raise ValueError('Please provide an age range within 0.01-10 Gyr')

    # Get xval floats
    if hasattr(xval[0], 'unit'):
        xval, xval_unc = xval
        xval = xval.value
        xval_unc = xval_unc.value
    else:
        xval, xval_unc = xval

    # Grab and plot the desired isochrones
    D, fig = isochrones(evo_model=evo_model, xparam=xparam, yparam=yparam,
                        ages=ages, jupiter=jupiter, plot=plot)
    Q = {d[0]: {'x': d[1], 'y': d[2]} for d in D}

    # Pull out isochrones which lie just above and below min_age and max_age
    A = np.array(list(zip(*D))[0])
    min1 = A[A <= min_age][-1]
    min2 = A[A >= min_age][0]
    max1 = A[A <= max_age][-1]
    max2 = A[A >= max_age][0]

    # Create a high-res x-axis in region of interest and interpolate
    # isochrones horizontally onto new x-axis
    x = np.linspace(xval - xval_unc, xval + xval_unc, 20)
    for k, v in Q.items():
        v['y'] = np.interp(x, v['x'], v['y'])
        v['x'] = x

    # Create isochrones interpolated vertically to *min_age* and *max_age*
    mn = zip(Q[min1]['y'], Q[min2]['y'])
    mx = zip(Q[max1]['y'], Q[max2]['y'])
    min_iso = [np.interp(min_age, [min1, min2], [r1, r2]) for r1, r2 in mn]
    max_iso = [np.interp(max_age, [max1, max2], [r1, r2]) for r1, r2 in mx]

    # Pull out least and greatest y value of interpolated isochrones in
    # x range of interest
    y_min, y_max = min(min_iso + max_iso), max(min_iso + max_iso)

    if plot:
        fig.patch([-10, -10, 10, 10], [y_min, y_max, y_max, y_min], alpha=0.3,
                  line_width=0, legend='{} range'.format(yparam))
        xmin = min([min(i[1]) for i in D])*0.75
        xmax = max([max(i[1]) for i in D])*1.25
        fig.x_range = Range1d(xmin, xmax)
        show(fig)

    # Round the values
    val = round(np.mean([y_min, y_max]), 2)
    unc = round(abs(y_min - np.mean([y_min, y_max])), 2)
        
    # Set the units of the output
    if yparam == 'radius':
        units = q.Rjup if jupiter else q.Rsun
    elif yparam == 'teff':
        units = q.K
    elif yparam == 'mass':
        units = q.Mjup if jupiter else q.Msun
    else:
        units = 1.

    return np.array([val, unc])*units


def isochrones(evo_model='hybrid_solar_age', xparam='Lbol', yparam='radius',
               ages=[0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 10], jupiter=False,
               plot=False, fig=None):
    """Generate model isochrones for the given ages

    Parameters
    ----------
    evo_model: str
        The name of the evolutionary model
    xparam: str
        The x-axis parameter
    yapram: str
        The y-axis parameter
    ages: sequence
        The ages at which the isochrones should be evaluated
    jupiter: bool
        Use Jupiter units instead of solar units
    plot: bool
        Plot the isochrones
    fig: bokeh.plotting.figure (optional)
        An existing figure to plot to

    Returns
    -------
    list
        The isochrones for the given ages
    """
    if plot:
        if fig is not None:
            fig = fig
        else:
            fig = figure()

    # Get the DMEStar and BD evo tracks
    DME = DMEstar(ages, xparam=xparam, yparam=yparam, jupiter=jupiter)
    evo_path = 'data/models/evolutionary/{}.txt'.format(evo_model)
    evo_file = resource_filename('SEDkit', evo_path)
    
    # Stitch them together using cubic spline
    D = []
    models = np.genfromtxt(evo_file, delimiter=',', usecols=range(6))
    data = [d for d in models if d[0] in ages and d[0] in list(zip(*DME))[0]]

    for k, g in groupby(data, key=lambda y: y[0]):
        dat = [np.array(i) for i in list(zip(*[list(i) for i in list(g)]))[:6]]
        age, mass, teff, Lbol, logg, radius = dat
        color = next(COLORS)

        # Convert to Jupier mass and radius
        if jupiter:
            mass *= 1047.2
            radius *= 9.72847

        # Get the data for the chosen parameters
        x = mass if xparam == 'mass' else logg if xparam == 'logg' else radius if xparam == 'radius' else teff if xparam == 'teff' else Lbol
        y = mass if yparam == 'mass' else logg if yparam == 'logg' else radius if yparam == 'radius' else teff if yparam == 'teff' else Lbol

        # Stitch together isochrones
        for idx, m in zip([15, 25, 0, 30, 28, 20, 20, 20], DME):
            if m[0] == k:
                (x1, y1) = (x, y) if x[0] < m[1][0] else (m[1], m[2])
                (x3, y3) = (x, y) if x[-1] > m[1][-1] else (m[1], m[2])
                x2 = np.arange(x1[0], x3[-1], 0.05)
                y2 = si.interp1d(np.concatenate([x1, x3]),
                                 np.concatenate([y1, y3]),
                                 kind='cubic')

                x2 = x2[np.logical_and(x2 > x1[-1], x2 < x3[0])]
                y2 = y2(x2)[np.logical_and(x2 > x1[-1], x2 < x3[0])]
                xnew = np.concatenate([x1, x2, x3])
                ynew = np.concatenate([y1, y2, y3])

                # Plot the isochrones
                if plot:
                    fig.line(xnew, ynew, color=color, legend='{} Gyr'.format(k))

        D.append([k, xnew, ynew])

    if plot:
        fig.xaxis.axis_label = xparam
        fig.yaxis.axis_label = yparam
        fig.legend.location = 'top_left'
        fig.legend.click_policy = "hide"

    return D, fig
