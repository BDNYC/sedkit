#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
A module to estimate fundamental parameters from model isochrones
"""
import os
from glob import glob
from pkg_resources import resource_filename

import astropy.units as q
from astropy.io.ascii import read
import astropy.table as at
from bokeh.plotting import figure, show
from bokeh.models import Range1d, LinearColorMapper, BasicTicker, ColorBar
import numpy as np

from .utilities import filter_table, color_gen

# A dictionary of all supported moving group ages from Bell et al. (2015)
NYMG_AGES = {'AB Dor': (149*q.Myr, 51*q.Myr),
             'beta Pic': (24*q.Myr, 3*q.Myr),
             'Carina': (45*q.Myr, 11*q.Myr),
             'Columba': (42*q.Myr, 6*q.Myr),
             'eta Cha': (11*q.Myr, 3*q.Myr),
             'Tuc-Hor': (45*q.Myr, 4*q.Myr),
             'TW Hya': (10*q.Myr, 3*q.Myr),
             '32 Ori': (22*q.Myr, 4*q.Myr)}

UNIT_DTYPES = (q.quantity.Quantity, q.core.PrefixUnit, q.core.Unit,
               q.core.CompositeUnit, q.core.IrreducibleUnit)


class Isochrone:
    """A class to handle model isochrones"""
    def __init__(self, path, name=None, units=None, **kwargs):
        """Initialize the isochrone object

        Parameters
        ----------
        path: str
            The path to the isochrone files
        name: str
            The name of the model set
        """
        self.name = name or os.path.basename(path)
        self.path = path
        self._age_units = None
        self._mass_units = None
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

        # Get the units
        if units is None or not isinstance(units, dict):
            units = {'age': q.Gyr, 'mass': q.Msun, 'teff': q.K}

        # Set the initial units
        for uname, unit in units.items():
            setattr(self, '{}_units'.format(uname), unit)

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
        # Make sure it's a quantity
        if not isinstance(unit, UNIT_DTYPES):
            raise TypeError('Age units must be astropy.units.quantity.Quantity')

        # Make sure the values are in flux density age_units
        if not unit.is_equivalent(q.Gyr):
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
            age = (age, age*0)

        # Make sure the age has units
        if not isinstance(age[0], UNIT_DTYPES) or not isinstance(age[1], UNIT_DTYPES):
            raise ValueError("'age' argument only accepts a sequence of the nominal age and associated uncertainty with astropy units of time.")

        # Check if the xval has an uncertainty
        if not isinstance(xval, (tuple, list)):
            xval = (xval, 0)

        # Convert (age, unc) into age range
        min_age = age[0]-age[1]
        max_age = age[0]+age[1]

        # Test the age range is inbounds
        if max_age > self.ages.max() or min_age < self.ages.min():
            raise ValueError('Please provide an age range within {} and {}'.format(self.ages.min(), self.ages.max()))

        # Get the lower, nominal, and upper values
        lower = self.iso_interp(xval[0]-xval[1], age[0]-age[1], xparam, yparam)
        nominal = self.iso_interp(xval[0], age[0], xparam, yparam)
        upper = self.iso_interp(xval[0]+xval[1], age[0]+age[1], xparam, yparam)

        # Caluclate the symmetric error
        error = max(abs(nominal-lower), abs(nominal-upper))*2

        # Plot the figure and evaluated point
        if plot:
            fig = self.plot(xparam, yparam)
            fig.circle(xval[0], nominal, color='red')
            fig.ellipse(x=xval[0], y=nominal, width=xval[1]*2, height=error,
                        color='red', alpha=0.1)
            show(fig)

        return nominal, error

    def iso_interp(self, xval, age, xparam, yparam):
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
        lower_age = self.ages[self.ages < age].max()
        upper_age = self.ages[self.ages > age].min()

        # Get the neighboring isochrones
        lower_iso = self.data[self.data['age'] == lower_age.value][[xparam, yparam]].as_array()
        upper_iso = self.data[self.data['age'] == upper_age.value][[xparam, yparam]].as_array()

        # Get the neighboring interpolated values
        lower_val = np.interp(xval, lower_iso[xparam], lower_iso[yparam])
        upper_val = np.interp(xval, upper_iso[xparam], upper_iso[yparam])

        # Take the weighted mean of the two points to find the single value
        weights = (lower_age/age).value, (upper_age/age).value
        result = np.average([lower_val, upper_val], weights=weights)

        return result

    def plot(self, xparam, yparam, **kwargs):
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
        fig = figure(title=self.name, x_axis_label=xparam, y_axis_label=yparam,
                     **kwargs)

        # Generate a colorbar
        colors = color_gen(colormap='viridis', n=len(self.ages))
        color_mapper = LinearColorMapper(palette='Viridis256',
                                         low=self.ages.min().value,
                                         high=self.ages.max().value)
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                             label_standoff=5, border_line_color=None,
                             title='Age [{}]'.format(self.age_units),
                             location=(0,0))

        # Plot a line for each isochrone
        for age in self.ages.value:
            data = self.data[self.data['age'] == age][[xparam, yparam]].as_array()
            fig.line(data[xparam], data[yparam], color=next(colors))

        # Add the colorbar
        fig.add_layout(color_bar, 'right')

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
        # Make sure it's a quantity
        if not isinstance(unit, UNIT_DTYPES):
            raise TypeError('Age units must be astropy.units.quantity.Quantity')

        # Make sure the values are in flux density mass_units
        if not unit.is_equivalent(q.Msun):
            raise TypeError("{}: Mass units must be mass units, e.g. 'Msun'".format(unit))

        # Define the mass_units...
        self._mass_units = unit
        if self.data['mass'].unit is None:
            self.data['mass'] *= self.mass_units

        # ...or convert them
        else:
            self.data['mass'] = self.data['mass'].to(self.mass_units)

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
        # Make sure it's a quantity
        if not isinstance(unit, UNIT_DTYPES):
            raise TypeError('Teff units must be astropy.units.quantity.Quantity')

        # Make sure the values are in flux density teff_units
        if not unit.is_equivalent(q.K):
            raise TypeError("{}: Teff units must be temperature units, e.g. 'K'".format(unit))

        # Define the teff_units...
        self._teff_units = unit
        if self.data['teff'].unit is None:
            self.data['teff'] *= self.teff_units

        # ...or convert them
        else:
            self.data['teff'] = self.data['teff'].to(self.teff_units)


class PARSEC(Isochrone):
    """A class for the PARSEC 1.2 model isochrones

    Data described in Bressan et al. (2012)
    """
    def __init__(self, Z='solar', **kwargs):
        """Initialize the model isochrone instance"""
        # Set the init parameters
        path = resource_filename('sedkit', 'data/models/evolutionary/parsec12_{}.txt'.format(Z))

        # Inherit from Isochrone class
        super().__init__(name='PARSEC v1.2 - Solar Metallicity', path=path, **kwargs)
