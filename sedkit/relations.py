#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
# !python3
"""
This is the code used to generate the polynomial relations
used in sedkit's calculations
"""
from pkg_resources import resource_filename

import astropy.io.ascii as ii
import astropy.units as q
import astropy.table as at
from astroquery.vizier import Vizier
from bokeh.plotting import figure, show
import numpy as np

from . import utilities as u


V = Vizier(columns=["**"])


class Relation:
    """A base class to store raw data, fit a polynomial, and evaluate quickly"""
    def __init__(self, file, xparam=None, yparam=None, order=None, add_columns=None, **kwargs):
        """Load the data

        Parameters
        ----------
        file: str
            The file to load
        """
        # Load the file into a table
        self.data = ii.read(file, **kwargs)

        # Fill in masked values
        self.data = self.data.filled(np.nan)

        # Add additional columns
        if isinstance(add_columns, dict):
            for colname, values in add_columns.items():
                self.add_column(colname, values)

        # Set attributes
        self.xparam = xparam
        self.yparam = yparam
        self.order = order
        self.x = None
        self.y = None
        self.coeffs = None
        self.C_p = None
        self.matrix = None
        self.yi = None
        self.C_yi = None
        self.sig_yi = None
        self.derived = False

        # Try to derive
        if self.xparam is not None and self.yparam is not None and self.order is not None:
            self.derive(self.xparam, self.yparam, self.order)

    def add_column(self, colname, values):
        """
        Add the values to the data table

        Parameters
        ----------
        colname: str
            The column name
        values: sequence
            The values for the column
        """
        # Check the colname
        if colname in self.data.colnames:
            raise KeyError("{}: column name already exists!".format(colname))

        # Check the length
        if len(values) != len(self.data):
            raise ValueError("{} != {}: number of values must match number of data rows.".format(len(values), len(self.data)))

        # Add the column
        self.data[colname] = values

    def derive(self, xparam, yparam, order, xrange=None):
        """
        Create a polynomial of the given *order* for *yparam* as a function of *xparam*
        which can be evaluated at any x value

        Parameters
        ----------
        xparam: str
            The x-axis parameter
        yparam: str
            The y-axis parameter
        order: int
            The order of the polynomial fit
        """
        # Make sure params are in the table
        if xparam not in self.data.colnames or yparam not in self.data.colnames:
            raise NameError("{}, {}: Make sure both parameters are in the data, {}".format(xparam, yparam, self.data.colnames))

        # Grab data
        self.xparam = xparam
        self.yparam = yparam
        self.order = order
        self.x = self.data[xparam]
        self.y = self.data[yparam]

        # Set x range for fit
        if xrange is not None:
            idx = np.where(np.logical_and(self.x > xrange[0], self.x < xrange[1]))
            self.x = self.x[idx]
            self.y = self.y[idx]

        # Remove masked and NaN values
        self.x, self.y = np.asarray([(x, y) for x, y in zip(self.x, self.y) if not hasattr(x, 'mask') and not np.isnan(x) and not hasattr(y, 'mask') and not np.isnan(y)]).T

        # Determine monotonicity
        self.monotonic = u.monotonic(self.x)

        # Set weighting
        self.weight = np.ones_like(self.x)
        if '{}_unc'.format(yparam) in self.data.colnames:
            self.weight = 1. / self.data['{}_unc'.format(yparam)]

        # Try to fit a polynomial
        try:

            # Fit polynomial
            self.coeffs, self.C_p = np.polyfit(self.x, self.y, self.order, w=self.weight, cov=True)

            # Matrix with rows 1, spt, spt**2, ...
            self.matrix = np.vstack([self.x**(order-i) for i in range(order + 1)]).T

            # Matrix multiplication calculates the polynomial values
            self.yi = np.dot(self.matrix, self.coeffs)

            # C_y = TT*C_z*TT.T
            self.C_yi = np.dot(self.matrix, np.dot(self.C_p, self.matrix.T))

            # Standard deviations are sqrt of diagonal
            self.sig_yi = np.sqrt(np.diag(self.C_yi))

            # Set as derived
            self.derived = True

        except Exception as exc:
            print(exc)
            print("Could not fit a polynomial to [{}, {}, {}, {}]. Try different values.".format(xparam, yparam, order, xrange))

    def estimate(self, xval, plot=False):
        """
        Estimate the y-value given the xvalue

        Parameters
        ----------
        x_val: float, int
            The xvalue to evaluate

        Returns
        -------
        y_val, y_unc
            The value and uncertainty
        """
        # Check to see if the polynomial has been derived
        if not self.derived:
            print("Please run the derive method before trying to evaluate.")
            return

        # Find the nearest
        y_val = np.polyval(self.coeffs, x_val)
        y_unc = np.interp(x_val, self.x, self.sig_yi)

        if plot:
            plt = self.plot()
            plt.circle([x_val], [y_val], color='red', size=10, legend='f({})'.format(x_val))
            show(plt)

        return y_val, y_unc

    def evaluate(self, x_val, plot=False):
        """
        Evaluate the derived polynomial at the given xval

        Parameters
        ----------
        x_val: float, int
            The xvalue to evaluate

        Returns
        -------
        y_val, y_unc
            The value and uncertainty
        """
        # Check to see if the polynomial has been derived
        if not self.derived:
            print("Please run the derive method before trying to evaluate.")
            return

        try:

            # Evaluate the polynomial
            y_val = np.polyval(self.coeffs, x_val)
            y_unc = np.interp(x_val, self.x, self.sig_yi)

            if plot:
                plt = self.plot()
                plt.circle([x_val], [y_val], color='red', size=10, legend='f({})'.format(x_val))
                show(plt)

            return y_val, y_unc

        except ValueError as exc:

            print(exc)
            print("Could not evaluate the {}({}) relation at {}".format(self.yparam, self.xparam, x_val))

            return None

    def plot(self, xparam=None, yparam=None, **kwargs):
        """
        Plot the data for the given parameters
        """
        # If no param, use stored
        if xparam is None:
            xparam = self.xparam

        if yparam is None:
            yparam = self.yparam

        # Make sure there is data to plot
        if xparam is None or yparam is None:
            raise ValueError("{}, {}: Not enough data to plot.".format(xparam, yparam))

        # Make the figure
        fig = figure(x_axis_label=xparam, y_axis_label=yparam)
        fig.circle(self.data[xparam], self.data[yparam], legend='Data', **kwargs)

        if self.derived and xparam == self.xparam and yparam == self.yparam:

            # Plot polynomial values
            xaxis = np.linspace(self.x.min(), self.x.max(), 100)
            evals = [self.evaluate(i)[0] for i in xaxis]
            fig.line(xaxis, evals, color='black', legend='Fit')

            # Plot polynomial uncertainties
            xunc = np.append(self.x, self.x[::-1])
            yunc = np.append(self.yi-self.sig_yi, (self.yi+self.sig_yi)[::-1])
            fig.patch(xunc, yunc, fill_alpha=0.1, line_alpha=0, color='black')

        return fig


class DwarfSequence(Relation):
    """A class to evaluate the Main Sequence in arbitrary parameter spaces"""
    def __init__(self, **kwargs):
        """
        Initialize a Relation object with the Dwarf Sequence data
        """
        # Get the file
        file = resource_filename('sedkit', 'data/dwarf_sequence.txt')

        # Replace '...' with NaN
        fill_values = [('...', np.nan), ('....', np.nan), ('.....', np.nan)]

        # Initialize Relation object
        super().__init__(file, fill_values=fill_values, **kwargs)

        self.add_column('spt', [u.specType(i)[0] for i in self.data['SpT']])


class SpectralTypeRadius:
    def __init__(self, orders=(5, 3), name='Spectral Type vs. Radius'):
        """Initialize the object

        Parameters
        ----------
        order: int
            The order polynomial to fit to the spt-radius data
        """
        self.name = name
        self.generate(orders)

    def get_radius(self, spt, plot=False):
        """Get the radius for the given spectral type

        Parameters
        ----------
        spt: str, int
            The alphanumeric (e.g. 'A0') or integer (0-99 => O0-Y9) spectral
            type
        plot: bool
            Generate a plots

        Returns
        -------
        tuple
            The radius and uncertainty in solar radii
        """
        # Convert to integer
        if isinstance(spt, (str, bytes)):
            spt = u.specType(spt)[0]

        # Test valid ranges
        if not isinstance(spt, (int, float)) or not 30 <= spt <= 99:
            raise ValueError("Please provide a spectral type within [30, 99]")

        # Evaluate the polynomials
        if spt > 64:
            data = self.MLTY
        else:
            data = self.AFGK
        radius = np.polyval(data['coeffs'], spt)*q.Rsun
        radius_unc = np.interp(spt, data['spt'], data['sig_yi'])*q.Rsun

        if plot:
            fig = self.plot()
            fig.triangle([spt], [radius.value], color='red', size=15, legend=u.specType(spt))
            show(fig)

        return radius.round(3), radius_unc.round(3)

    def generate(self, orders):
        """
        Generate a polynomial that describes the radius as a function of
        spectral type for empirically measured AFGKM main sequence stars
        (Boyajian+ 2012b, 2013) and MLTY model isochrone interpolated stars
        (Filippazzoet al. 2015, 2016)

        Parameters
        ----------
        orders: sequence
            The order polynomials to fit to the MLTY and AFGK data
        generate: bool
            Generate the polynomials
        """
        # ====================================================================
        # Boyajian AFGKM data
        # ====================================================================

        afgk = resource_filename('sedkit', 'data/AFGK_radii.txt')
        afgk_data = ii.read(afgk, format='csv', comment='#')

        # ====================================================================
        # Filippazzo MLTY data
        # ====================================================================

        # Get the data
        cat1 = V.query_constraints('J/ApJ/810/158/table1')[0]
        cat2 = V.query_constraints('J/ApJ/810/158/table9')[0]

        # Join the tables to getthe spectral types and radii in one table
        mlty_data = at.join(cat1, cat2, keys='ID', join_type='outer')

        # Only keep field age
        mlty_data = mlty_data[mlty_data['b_Age'] >= 0.5]

        # Rename columns
        mlty_data.rename_column('SpT', 'spectral_type')
        mlty_data.rename_column('Rad', 'radius')
        mlty_data.rename_column('e_Rad', 'radius_unc')

        # Make solar radii units
        mlty_data['radius'] = mlty_data['radius'].to(q.Rsun)
        mlty_data['radius_unc'] = mlty_data['radius_unc'].to(q.Rsun)

        # ====================================================================
        # Fit and save the data
        # ====================================================================

        for data, name, order, ref, rng in zip([afgk_data, mlty_data],
                                               ['AFGK', 'MLTY'], orders,
                                               ['Boyajian+ 2012b, 2013', 'Filippazzo+ 2015'],
                                               [(30, 65), (65, 99)]):

            # Container for data
            container = {}

            # Translate string SPT to numbers
            spts = []
            keep = []
            for n,i in enumerate(data['spectral_type']):
                try:
                    spt = u.specType(i)
                    spts.append(spt)
                    keep.append(n)
                except:
                    pass

            # Filter bad spectral types
            data = data[keep]

            # Add the number to the table
            num, *_, lum = np.array(spts).T
            data['spt'] = num.astype(float)
            data['lum'] = lum

            # Filter out sub-giants
            data = data[(data['spt'] > rng[0]) & (data['spt'] < rng[1])]
            data = data[data['lum'] == 'V']
            data = data[((data['radius'] < 1.8) & (data['spt'] > 37)) | (data['spt'] <= 37)]

            # Filter out nans
            data = data[data['radius'] < 4]
            data = data[data['radius'] > 0]
            data = data[data['radius_unc'] > 0]
            container['data'] = data[data['spt'] > 0]
            container['rng'] = rng

            # Fit polynomial
            container['coeffs'], container['C_p'] = np.polyfit(data['spt'], data['radius'], order, w=1./data['radius_unc'], cov=True)

            # Do the interpolation for plotting
            container['spt'] = np.arange(np.nanmin(data['spt'])-3, np.nanmax(data['spt'])+1)

            # Matrix with rows 1, spt, spt**2, ...
            container['sptT'] = np.vstack([container['spt']**(order-i) for i in range(order+1)]).T

            # Matrix multiplication calculates the polynomial values
            container['yi'] = np.dot(container['sptT'], container['coeffs'])

            # C_y = TT*C_z*TT.T
            container['C_yi'] = np.dot(container['sptT'], np.dot(container['C_p'], container['sptT'].T))

            # Standard deviations are sqrt of diagonal
            container['sig_yi'] = np.sqrt(np.diag(container['C_yi']))

            # Store the new order
            container['order'] = order

            # Set the reference
            container['ref'] = ref

            # Add the container as an attribute
            setattr(self, name, container)

    def plot(self, draw=False):
        """Plot the relation

        Parameters
        ----------
        draw: bool
            Draw the figure, else return it

        Returns
        -------
        bokeh.plotting.figure
            The plotted figure
        """
        AFGK_color = '#1f77b4'
        MLTY_color = '#2ca02c'

        # Configure plot
        TOOLS = ['pan', 'reset', 'box_zoom', 'wheel_zoom', 'save']
        xlab = 'Spectral Type'
        ylab = 'Solar Radii'
        fig = figure(plot_width=800, plot_height=500, title=self.name,
                          x_axis_label=xlab, y_axis_label=ylab,
                          tools=TOOLS)

        # Plot the fit
        for n, (data, color) in enumerate(zip([self.AFGK, self.MLTY], [AFGK_color, MLTY_color])):

            # Add the data
            if n == 0:
                fig.circle(data['data']['spt'], data['data']['radius'], size=8,
                           color=color, legend=data['ref'])
            else:
                fig.square(data['data']['spt'], data['data']['radius'], size=8,
                           color=color, legend=data['ref'])

            # Add the fit line and uncertainty
            fig.line(data['spt'], data['yi'], color=color,
                     legend='Order {} Fit'.format(data['order']))
            x = np.append(data['spt'], data['spt'][::-1])
            y = np.append(data['yi']-data['sig_yi'], (data['yi']+data['sig_yi'])[::-1])
            fig.patch(x, y, fill_alpha=0.1, line_alpha=0, color=color)

        if draw:
            show(fig)
        else:
            return fig
