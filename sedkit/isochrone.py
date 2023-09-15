#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
# !python3
"""
A module to estimate fundamental parameters from model isochrones
"""

import os
import glob
from pkg_resources import resource_filename

import astropy.units as q
import astropy.constants as ac
from astropy.io.ascii import read
from bokeh.plotting import figure, show
from bokeh.models import LinearColorMapper, BasicTicker, ColorBar, Text
import numpy as np

from . import utilities as u
from . import uncertainties as un

# A dictionary of all supported moving group ages from Bell et al. (2015)
NYMG_AGES = {'AB Dor': (149 * q.Myr, 51 * q.Myr, '2015MNRAS.454..593B'),
             'beta Pic': (24 * q.Myr, 3 * q.Myr, '2015MNRAS.454..593B'),
             'Carina': (45 * q.Myr, 11 * q.Myr, '2015MNRAS.454..593B'),
             'Columba': (42 * q.Myr, 6 * q.Myr, '2015MNRAS.454..593B'),
             'eta Cha': (11 * q.Myr, 3 * q.Myr, '2015MNRAS.454..593B'),
             'Tuc-Hor': (45 * q.Myr, 4 * q.Myr, '2015MNRAS.454..593B'),
             'TW Hya': (10 * q.Myr, 3 * q.Myr, '2015MNRAS.454..593B'),
             '32 Ori': (22 * q.Myr, 4 * q.Myr, '2015MNRAS.454..593B')}

# A list of all supported evolutionary models
try:
    EVO_MODELS = [os.path.basename(m).replace('.txt', '') for m in glob.glob(resource_filename('sedkit', 'data/models/evolutionary/*'))]
# Fails RTD build for some reason
except:
    EVO_MODELS = ['COND03', 'dmestar_solar', 'DUSTY00', 'f2_solar_age', 'hybrid_solar_age', 'nc+03_age', 'nc-03_age', 'nc_solar_age', 'parsec12_solar', 'ATMO_NEQ_strong']

class Isochrone:
    """A class to handle model isochrones"""
    def __init__(self, name, units=None, verbose=True, **kwargs):
        """Initialize the isochrone object

        Parameters
        ----------
        name: str
            The name of the model set
        """
        self.verbose = verbose
        if name not in EVO_MODELS:
            raise ValueError(name, 'No evolutionary model by this name. Try', EVO_MODELS)

        # Set the path
        self.name = name
        self.path = resource_filename('sedkit', 'data/models/evolutionary/{}.txt'.format(self.name))
        self._age_units = None
        self._mass_units = None
        self._radius_units = None
        self._teff_units = None

        # Read in the data
        self.data = read(self.path)

        # Convert log to linear
        for col in self.data.colnames:
            if col.startswith('log_'):
                self.data[col[4:]] = 10**self.data[col]

        # Convert years to Gyr if necessary
        if min(self.data['age']) > 100000:
            self.data['age'] *= 1E-9

        # Calculate radii if not in the table (R = sqrt(GM/g))
        if 'radius' not in self.data.colnames:
            radius = np.sqrt((ac.G * (self.data['mass'] * q.M_sun)) / ((10**self.data['logg']) * q.m / q.s**2)).to(q.R_sun)
            self.data.add_column(radius, name='radius')

        # Get the units
        if units is None or not isinstance(units, dict):
            units = {'age': q.Gyr, 'mass': q.Msun, 'Lbol': q.Lsun, 'teff': q.K, 'radius': q.Rsun}

        # Set the initial units
        for uname, unit in units.items():
            try:
                setattr(self, '{}_units'.format(uname), unit)
            except KeyError:
                print("No '{}' parameter in {} models. Skipping.".format(uname, self.name))

        # Save the raw data for resampling
        self.raw_data = self.data

    @property
    def age_units(self):
        """A getter for the age units"""
        return self._age_units

    @age_units.setter
    def age_units(self, unit):
        """A setter for the age age_units

        Parameters
        ----------
        unit: astropy.units.quantity.Quantity
            The desired units of the age column
        """
        # Make sure the values are in time units
        if not u.equivalent(unit, q.Gyr):
            raise TypeError("{}: Age units must be time units, e.g. 'Gyr'".format(unit))

        # Define the age_units...
        self._age_units = unit
        if self.data['age'].unit is None:
            self.data['age'] *= self.age_units

        # ...or convert them
        else:
            self.data['age'] = self.data['age'].to(self.age_units)

        # Get the min and max ages
        self.ages = np.array(np.unique(self.data['age']))*self.age_units

    def evaluate(self, xval, age, xparam, yparam, plot=False):
        """Interpolate the value and uncertainty of *yparam* given an
        x-value and age range

        Parameters
        ----------
        xval: float, int, astropy.units.quantity.Quantity, sequence
            The value of the x-axis (or value and uncertainty) to evaluate
        age: astropy.units.quantity.Quantity, sequence
            The age or (age, uncertainty) of the source
        xparam: str
            The name of the parameter on the x-axis
        yparam: str
            The name of the parameter on the y-axis
        plot: bool
            Plot all isochrones and the interpolated value

        Returns
        -------
        float, int, astropy.units.quantity.Quantity, sequence
            The interpolated result
        """
        # Check if the age has an uncertainty
        if not isinstance(age, (tuple, list)):
            age = (age, age * 0)

        # Make sure the age has units
        if not u.equivalent(age[0], q.Gyr) or not u.equivalent(age[1], q.Gyr):
            raise ValueError("'age' argument only accepts a sequence of the nominal age and associated uncertainty with astropy units of time.")

        # Make sure age uncertainty is the same unit as age
        age = age[0], age[1].to(age[0].unit)

        # Check if the xval has an uncertainty
        if not isinstance(xval, (tuple, list)):
            xval = (xval, 0)

        # Convert (age, unc) into age range
        min_age = age[0] - age[1]
        max_age = age[0] + age[1]

        # Test the age range is inbounds
        if age[0] < self.ages.min() or age[0] > self.ages.max():
            args = age[0], self.ages.min(), self.ages.max(), yparam, self.name
            self.message('{}: age must be between {} and {} to infer {} from {} isochrones.'.format(*args))
            return None

        # Monte Carlo Approach
        mu, sigma = xval[0], xval[1]  # mean and standard deviation for values on the x-axis
        mu_a, sigma_a = age[0].value, age[1].value  # mean and standard deviation for the age range provided
        xsample = np.random.normal(mu, sigma, 1000)
        ysample = np.random.normal(mu_a, sigma_a, 1000)
        values_list = []
        nan_counter = 0

        for x, y in zip(xsample, ysample):
            result = self.interpolate(x, y * age[0].unit, xparam, yparam)
            if result is not None:
                values_list.append(result.value if hasattr(result, 'unit') else result)
            elif result is None:
                print("Interpolate resulted in NaN")
                nan_counter = nan_counter + 1

        # Get final values
        unit = self.data[yparam].unit or 1
        average = np.mean(np.array(values_list) * unit)
        std = np.std(values_list)
        error = std * unit

        # Plot the figure and evaluated point
        if plot:
            val = average.value if hasattr(average, 'unit') else average
            err = error.value if hasattr(error, 'unit') else error
            fig = self.plot(xparam, yparam)
            legend = '{} = {:.3f} ({:.3f})'.format(yparam, val, err)
            fig.circle(xval[0], val, color='red', legend_label=legend)
            u.errorbars(fig, [xval[0]], [val], xerr=[xval[1]*2], yerr=[err], color='red')

            show(fig)

        # Balk at nans
        if np.isnan(average):
            raise ValueError("I got a nan from {} for some reason.".format(self.name))

        return average, error, self.name

    def interpolate(self, xval, age, xparam, yparam):
        """Interpolate a value between two isochrones

        Parameters
        ----------
        xval: float, int, astropy.units.quantity.Quantity
            The value of the x-axis to evaluate
        age: astropy.units.quantity.Quantity
            The age of the source
        xparam: str
            The name of the parameter on the x-axis
        yparam: str
            The name of the parameter on the y-axis

        Returns
        -------
        float, int, astropy.units.quantity.Quantity
            The interpolated result
        """
        # Get the neighboring ages
        try:
            lower_age = self.ages[self.ages < age].max()
        except ValueError:
            lower_age =  self.ages.min()
        try:
            upper_age = self.ages[self.ages > age].min()
        except ValueError:
            upper_age = self.ages.max()

        # Get the neighboring isochrones
        lower_iso = self.data[self.data['age'] == lower_age.value][[xparam, yparam]].as_array()
        upper_iso = self.data[self.data['age'] == upper_age.value][[xparam, yparam]].as_array()

        # Test the xval is inbounds
        min_x = min(lower_iso[xparam].min(), upper_iso[xparam].min())
        max_x = max(lower_iso[xparam].max(), upper_iso[xparam].max())
        if xval < min_x or xval > max_x:
            args = round(xval, 3), xparam, min_x, max_x, yparam, self.name
            self.message('{}: {} must be between {} and {} to infer {} from {} isochrones.'.format(*args))
            return None

        # Get the neighboring interpolated values
        lower_val = np.interp(xval, lower_iso[xparam], lower_iso[yparam])
        upper_val = np.interp(xval, upper_iso[xparam], upper_iso[yparam])

        # Take the weighted mean of the two points to find the single value
        weights = (lower_age/age).value, (upper_age/age).value
        result = np.average([lower_val, upper_val], weights=weights)
        unit = self.data[yparam].unit or 1

        return result * unit

    def message(self, msg, pre='[sedkit]'):
        """
        Only print message if verbose=True

        Parameters
        ----------
        msg: str
            The message to print
        pre: str
            The stuff to print before
        """
        if self.verbose:
            if pre is None:
                print(msg)
            else:
                print("{} {}".format(pre, msg))

    def plot(self, xparam, yparam, draw=False, **kwargs):
        """Plot an evaluated isochrone, isochrone, or set of isochrones

        Parameters
        ----------
        xparam: str
            The column name to plot on the x-axis
        yparam: str
            The column name to plot on the y-axis

        Returns
        -------
        bokeh.figure
            The figure
        """
        # Make the figure
        try:
            xlabel = '{} [{}]'.format(xparam, getattr(self, '{}_units'.format(xparam)))
        except AttributeError:
            xlabel = xparam
        try:
            ylabel = '{} [{}]'.format(yparam, getattr(self, '{}_units'.format(yparam)))
        except AttributeError:
            ylabel = yparam
        fig = figure(title=self.name, x_axis_label=xlabel, y_axis_label=ylabel,
                     **kwargs)

        # Generate a colorbar
        colors = u.color_gen(colormap='viridis', n=len(self.ages))
        color_mapper = LinearColorMapper(palette='Viridis256',
                                         low=self.ages.min().value,
                                         high=self.ages.max().value)
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                             label_standoff=5, border_line_color=None,
                             title='Age [{}]'.format(self.age_units),
                             location=(0, 0))

        # Plot a line for each isochrone
        for age in self.ages.value:
            data = self.data[self.data['age'] == age][[xparam, yparam]].as_array()
            fig.line(data[xparam], data[yparam], color=next(colors))

        # Add the colorbar
        fig.add_layout(color_bar, 'right')

        if draw:
            show(fig)
        else:
            return fig

    @property
    def mass_units(self):
        """A getter for the mass units"""
        return self._mass_units

    @mass_units.setter
    def mass_units(self, unit):
        """A setter for the mass mass_units

        Parameters
        ----------
        unit: astropy.units.quantity.Quantity
            The desired units of the mass column
        """
        # Make sure the values are in mass units
        if not u.equivalent(unit, q.Msun):
            raise TypeError("{}: Mass units must be mass units, e.g. 'Msun'".format(unit))

        # Define the mass_units...
        self._mass_units = unit
        if self.data['mass'].unit is None:
            self.data['mass'] *= self.mass_units

        # ...or convert them
        else:
            self.data['mass'] = self.data['mass'].to(self.mass_units)

    @property
    def radius_units(self):
        """A getter for the radius units"""
        return self._radius_units

    @radius_units.setter
    def radius_units(self, unit):
        """A setter for the radius radius_units

        Parameters
        ----------
        unit: astropy.units.quantity.Quantity
            The desired units of the radius column
        """
        # Make sure the values are in distance units
        if not u.equivalent(unit, q.Rsun):
            raise TypeError("{}: Radius units must be distance units, e.g. 'Rsun'".format(unit))

        # Define the radius_units...
        self._radius_units = unit
        if self.data['radius'].unit is None:
            self.data['radius'] *= self.radius_units

        # ...or convert them
        else:
            self.data['radius'] = self.data['radius'].to(self.radius_units)

    @property
    def teff_units(self):
        """A getter for the teff units"""
        return self._teff_units

    @teff_units.setter
    def teff_units(self, unit):
        """A setter for the teff teff_units

        Parameters
        ----------
        unit: astropy.units.quantity.Quantity
            The desired units of the teff column
        """
        # Make sure the values are in temperature units
        if not u.equivalent(unit, q.K):
            raise TypeError("{}: Teff units must be temperature units, e.g. 'K'".format(unit))

        # Define the teff_units...
        self._teff_units = unit
        if self.data['teff'].unit is None:
            self.data['teff'] *= self.teff_units

        # ...or convert them
        else:
            self.data['teff'] = self.data['teff'].to(self.teff_units)
