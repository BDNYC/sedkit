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
from .uncertainties import Unum


V = Vizier(columns=["**"])


class Relation:
    """A base class to store raw data, fit a polynomial, and evaluate quickly"""
    def __init__(self, file, add_columns=None, ref=None, **kwargs):
        """Load the data

        Parameters
        ----------
        file: str
            The file to load
        """
        # Load the file into a table
        self.data = ii.read(file, **kwargs)
        self.ref = ref

        # Fill in masked values
        self.data = self.data.filled(np.nan)

        # Dict of relations
        self.relations = {}

        # Add additional columns
        if isinstance(add_columns, dict):
            for colname, values in add_columns.items():
                self.add_column(colname, values)

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
        if colname in self.parameters:
            raise KeyError("{}: column name already exists!".format(colname))

        # Check the length
        if len(values) != len(self.data):
            raise ValueError("{} != {}: number of values must match number of data rows.".format(len(values), len(self.data)))

        # Add the column
        self.data[colname] = values

    def add_relation(self, rel_name, order, xrange=None, xunit=None, yunit=None, plot=True):
        """
        Create a polynomial of the given *order* for *yparam* as a function of *xparam*
        which can be evaluated at any x value

        Parameters
        ----------
        rel_name: str
            The relation name, i.e. 'yparam(xparam)'
        order: int
            The order of the polynomial fit
        xrange: sequence
            The range of x-values to consider
        xunit: astropy.units.quantity.Quantity
            The units of the x parameter values
        yunit: astropy.units.quantity.Quantity
            The units of the y parameter values
        """
        # Get params
        xparam, yparam = self._parse_rel_name(rel_name)

        # Make sure params are in the table
        if xparam not in self.parameters or yparam not in self.parameters:
            raise NameError("{}, {}: Make sure both parameters are in the data, {}".format(xparam, yparam, self.data.colnames))

        # Grab data
        rel = {'xparam': xparam, 'yparam': yparam, 'order': order, 'x': np.array(self.data[xparam]), 'y': np.array(self.data[yparam]),
               'coeffs': None, 'C_p': None, 'matrix': None, 'yi': None, 'C_yi': None, 'sig_yi': None, 'xunit': xunit or 1, 'yunit': yunit or 1}

        # Set x range for fit
        if xrange is not None:
            idx = np.where(np.logical_and(rel['x'] > xrange[0], rel['x'] < xrange[1]))
            rel['x'] = rel['x'][idx]
            rel['y'] = rel['y'][idx]

        # Remove masked and NaN values
        rel['x'], rel['y'] = self.validate_data(rel['x'], rel['y'])

        # Determine monotonicity
        rel['monotonic'] = u.monotonic(rel['x'])

        # Set weighting
        rel['weight'] = np.ones_like(rel['x'])
        if '{}_unc'.format(yparam) in self.data.colnames:
            rel['weight'] = 1. / self.data['{}_unc'].format(yparam)

        # Try to fit a polynomial
        try:

            # Fit polynomial
            rel['coeffs'], rel['C_p'] = np.polyfit(rel['x'], rel['y'], rel['order'], w=rel['weight'], cov=True)

            # Matrix with rows 1, spt, spt**2, ...
            rel['matrix'] = np.vstack([rel['x']**(order-i) for i in range(order + 1)]).T

            # Matrix multiplication calculates the polynomial values
            rel['yi'] = np.dot(rel['matrix'], rel['coeffs'])

            # C_y = TT*C_z*TT.T
            rel['C_yi'] = np.dot(rel['matrix'], np.dot(rel['C_p'], rel['matrix'].T))

            # Standard deviations are sqrt of diagonal
            rel['sig_yi'] = np.sqrt(np.diag(rel['C_yi']))

        except Exception as exc:
            print(exc)
            print("Could not fit a polynomial to [{}, {}, {}, {}]. Try different values.".format(xparam, yparam, order, xrange))

        # Add relation to dict
        self.relations['{}({})'.format(yparam, xparam)] = rel

        if plot:
            show(self.plot(rel_name))

    def evaluate(self, rel_name, x_val, plot=False):
        """
        Evaluate the given relation at the given xval

        Parameters
        ----------
        rel_name: str
            The relation name, i.e. 'yparam(xparam)'
        x_val: float, int
            The xvalue to evaluate

        Returns
        -------
        y_val, y_unc, ref
            The value, uncertainty, and reference
        """
        # Check to see if the polynomial has been derived
        if not rel_name in self.relations:
            print("Please run 'add_relation' method for {} before trying to evaluate.".format(rel_name))
            return

        if x_val is None:

            return None

        else:

            try:

                # Get the relation
                rel = self.relations[rel_name]

                # Evaluate the polynomial
                if isinstance(x_val, (list, tuple)):

                    # With uncertainties
                    x = Unum(*x_val)
                    y = x.polyval(rel['coeffs'])
                    x_val = x.nominal
                    y_val = y.nominal * rel['yunit']
                    y_upper = y.upper * rel['yunit']
                    y_lower = y.lower * rel['yunit']

                else:

                    # Without uncertainties
                    x_val = x_val.value if hasattr(x_val, 'unit') else x_val
                    y_val = np.polyval(rel['coeffs'], x_val) * rel['yunit']
                    y_lower = y_upper = None

                if plot:
                    print(y_val, y_lower, y_upper)
                    plt = self.plot(rel_name)
                    plt.circle([x_val], [y_val], color='red', size=10, legend='{}({})'.format(rel['yparam'], x_val))
                    if y_upper:
                        plt.line([x_val, x_val], [y_val - y_lower, y_val + y_upper], color='red')
                    show(plt)

                if y_upper:
                    return y_val, y_upper, y_lower, self.ref
                else:
                    return y_val, self.ref

            except ValueError as exc:

                print(exc)
                print("Could not evaluate the {} relation at {}".format(rel_name, x_val))

                return None

    @property
    def parameters(self):
        """
        List of parameters in the data table
        """
        return self.data.colnames

    def _parse_rel_name(self, rel_name):
        """
        Parse the rel_name into xparam and yparam

        Parameters
        ----------
        rel_name: str
            The relation name, i.e. 'yparam(xparam)'

        Returns
        -------
        str, str
            The xparam and yparam of the relation
        """
        return rel_name.replace(')', '').split('(')[::-1]

    def plot(self, rel_name, **kwargs):
        """
        Plot the data for the given parameters
        """
        # Get params
        xparam, yparam = self._parse_rel_name(rel_name)

        if not xparam in self.parameters or not yparam in self.parameters:
            raise ValueError("{}, {}: Both parameters need to be in the relation. Try {}".format(xparam, yparam, self.relations))

        # Make the figure
        fig = figure(x_axis_label=xparam, y_axis_label=yparam)
        x, y = self.validate_data(self.data[xparam], self.data[yparam])
        fig.circle(x, y, legend='Data', **kwargs)

        if rel_name in self.relations:

            # Get the relation
            rel = self.relations[rel_name]

            # Plot polynomial values
            xaxis = np.linspace(rel['x'].min(), rel['x'].max(), 100)
            evals = np.polyval(rel['coeffs'], xaxis)
            fig.line(xaxis, evals, color='black', legend='Fit')

        return fig

    def validate_data(self, X, Y):
        """
        Validate the data for onlu numbers

        Parameters
        ----------
        X: sequence
            The x-array
        Y: sequence
            The y-array

        Returns
        -------
        sequence
            The validated arrays
        """
        valid = np.asarray([(float(x), float(y)) for x, y in zip(X, Y) if u.isnumber(x) and u.isnumber(y)]).T

        if len(valid) == 0:
            raise ValueError("No valid data in the arrays")
        else:
            return valid


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
        super().__init__(file, fill_values=fill_values, ref='2013ApJS..208....9P', **kwargs)

        self.add_column('spt', [u.specType(i)[0] for i in self.data['SpT']])

        # Add well-characterized relations
        self.add_relation('Teff(spt)', 12, yunit=q.K, plot=False)
        self.add_relation('Teff(Lbol)', 9, yunit=q.K, plot=False)
        self.add_relation('radius(Lbol)', 9, yunit=q.R_sun, plot=False)
        self.add_relation('radius(spt)', 11, yunit=q.R_sun, plot=False)
        self.add_relation('radius(M_J)', 9, yunit=q.R_sun, plot=False)
        self.add_relation('radius(M_Ks)', 9, yunit=q.R_sun, plot=False)
        self.add_relation('mass(Lbol)', 9, yunit=q.M_sun, plot=False)
        self.add_relation('mass(M_Ks)', 9, yunit=q.M_sun, plot=False)
        self.add_relation('mass(M_J)', 9, yunit=q.M_sun, plot=False)


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
