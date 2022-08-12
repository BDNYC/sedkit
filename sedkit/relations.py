#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
# !python3
"""
This is the code used to generate the polynomial relations
used in sedkit's calculations
"""
import os
from pkg_resources import resource_filename

import astropy.io.ascii as ii
import astropy.units as q
import astropy.table as at
from astroquery.vizier import Vizier
from bokeh.plotting import figure, show
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline, UnivariateSpline
from bokeh.models.glyphs import Patch
from bokeh.models import ColumnDataSource
import numpy as np

from . import utilities as u
from .uncertainties import Unum


V = Vizier(columns=["**"])


class Relation:
    """A base class to store raw data, fit a polynomial, and evaluate quickly"""
    def __init__(self, table, add_columns=None, ref=None, **kwargs):
        """Load the data

        Parameters
        ----------
        table: str, astropy.table.Table
            The file or table to load
        """
        # Load the file into a table
        if isinstance(table, str):
            if os.path.exists(table):
                table = ii.read(table, **kwargs)

        # Make sure it's a table
        if not isinstance(table, at.Table):
            raise TypeError("{} is not a valid table of data. Please provide a astropy.table.Table or path to an ascii file to ingest.".format(type(table)))

        # Store the data
        self.data = table
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

    def add_relation(self, rel_name, order, xrange=None, xunit=None, yunit=None, reject_outliers=False, plot=True):
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
        reject_outliers: bool
            Use outlier rejection in the fit if polynomial
            is of order 3 or less
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
            idx = np.where(np.logical_and(rel['x'] > np.nanmin(xrange), rel['x'] < np.nanmax(xrange)))
            rel['x'] = rel['x'][idx]
            rel['y'] = rel['y'][idx]

        # Remove masked and NaN values
        rel['x'], rel['y'], rel['weight'] = self.validate_data(rel['x'], rel['y'])

        # Set weighting
        if '{}_unc'.format(yparam) in self.data.colnames:
            y_unc = np.array(self.data['{}_unc'.format(yparam)])
            rel['x'], rel['y'], y_unc = self.validate_data(rel['x'], rel['y'], y_unc)
            rel['weight'] = 1. / y_unc

        # Determine monotonicity
        rel['monotonic'] = u.monotonic(rel['x'])

        # Try to fit a polynomial
        try:

            # X array
            rel['x_fit'] = np.linspace(rel['x'].min(), rel['x'].max(), 1000)

            if reject_outliers:

                def f(x, *c):
                    """Generic polynomial function"""
                    result = 0
                    for coeff in c:
                        result = x * result + coeff
                    return result

                def residual(p, x, y):
                    """Residual calulation"""
                    return y - f(x, *p)

                def errFit(hess_inv, resVariance):
                    return np.sqrt(np.diag(hess_inv * resVariance))

                # TODO: This fails for order 4 or more
                # Fit polynomial to data
                p0 = np.ones(rel['order'] + 1)
                res_robust = least_squares(residual, p0, loss='soft_l1', f_scale=0.1, args=(rel['x'], rel['y']))
                rel['coeffs'] = res_robust.x
                rel['jac'] = res_robust.jac
                rel['y_fit'] = f(rel['x_fit'], *rel['coeffs'])

                # Calculate errors on coefficients
                rel['sig_coeffs'] = errFit(np.linalg.inv(np.dot(rel['jac'].T, rel['jac'])), (residual(rel['coeffs'], rel['x'], rel['y']) ** 2).sum() / (len(rel['y']) - len(p0)))
                rel['sig_coeffs2'] = errFit(np.linalg.inv(2 * np.dot(rel['jac'].T, rel['jac'])), (residual(rel['coeffs'], rel['x'], rel['y']) ** 2).sum() / (len(rel['y']) - len(p0)))

                # Calculate upper and lower bounds on the fit
                coeff_err = rel['coeffs'] - rel['sig_coeffs']
                rel['y_fit_err'] = f(rel['x_fit'], *coeff_err)

            else:

                # Fit polynomial
                rel['coeffs'], rel['C_p'] = np.polyfit(rel['x'], rel['y'], rel['order'], w=rel['weight'], cov=True)

                # Matrix with rows 1, spt, spt**2, ...
                rel['matrix'] = np.vstack([rel['x'] ** (order - i) for i in range(order + 1)]).T

                # Matrix multiplication calculates the polynomial values
                rel['yi'] = np.dot(rel['matrix'], rel['coeffs'])

                # C_y = TT*C_z*TT.T
                rel['C_yi'] = np.dot(rel['matrix'], np.dot(rel['C_p'], rel['matrix'].T))

                # Standard deviations are sqrt of diagonal
                rel['sig_yi'] = np.sqrt(np.diag(rel['C_yi']))

                # Plot polynomial values
                rel['y_fit'] = np.polyval(rel['coeffs'], rel['x_fit'])

        except Exception as exc:
            print(exc)
            print("Could not fit a polynomial to [{}, {}, {}, {}]. Try different values.".format(xparam, yparam, order, xrange))

        # Add relation to dict
        self.relations['{}({})'.format(yparam, xparam)] = rel

        if plot:
            show(self.plot(rel_name))

    def evaluate(self, rel_name, x_val, xunits=None, yunits=None, fit_local=False, plot=False):
        """
        Evaluate the given relation at the given xval

        Parameters
        ----------
        rel_name: str
            The relation name, i.e. 'yparam(xparam)'
        x_val: float, int
            The xvalue to evaluate
        xunits: astropy.units.quantity.Quantity
            The output x units
        yunits: astropy.units.quantity.Quantity
            The output y units

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
                full_rel = self.relations[rel_name]
                out_xunits = full_rel['xunit'].to(xunits) * xunits if xunits is not None else full_rel['xunit'] or 1
                out_yunits = full_rel['yunit'].to(yunits) * yunits if yunits is not None else full_rel['yunit'] or 1

                # Use local points for relation
                if isinstance(fit_local, int) and fit_local is not False:

                    # Trim relation data to nearby points, refit with low order polynomial, and evaluate
                    idx = np.argmin(np.abs(full_rel['x'] - (x_val[0] if isinstance(x_val, (list, tuple)) else x_val)))
                    x_min, x_max = full_rel['x'][max(0, idx - fit_local)], full_rel['x'][min(idx + fit_local, len(full_rel['x'])-1)]
                    self.add_relation(rel_name, 2, xrange=[x_min, x_max], yunit=full_rel['yunit'], reject_outliers=True, plot=False)
                    rel = self.relations[rel_name]

                # ... or use the full relation
                else:
                    rel = full_rel

                # Evaluate the polynomial
                if isinstance(x_val, (list, tuple)):

                    # With uncertainties
                    x = Unum(*x_val)
                    y = x.polyval(rel['coeffs'])
                    x_val = x.nominal * out_xunits
                    y_val = y.nominal * out_yunits
                    y_upper = y.upper * out_yunits
                    y_lower = y.lower * out_yunits

                else:

                    # Without uncertainties
                    x_val = x_val * out_xunits
                    y_val = np.polyval(rel['coeffs'], x_val) * out_yunits
                    y_lower = y_upper = None

                if plot:
                    plt = self.plot(rel_name, xunits=xunits, yunits=yunits)
                    plt.circle([x_val.value if hasattr(x_val, 'unit') else x_val], [y_val.value if hasattr(y_val, 'unit') else y_val], color='red', size=10, legend='{}({})'.format(rel['yparam'], x_val))
                    if y_upper:
                        plt.line([x_val, x_val], [y_val - y_lower, y_val + y_upper], color='red')
                    show(plt)

                # Restore full relation
                self.relations[rel_name] = full_rel

                if y_upper:
                    return y_val, y_upper, y_lower, self.ref
                else:
                    return y_val, self.ref

            except IndexError as exc:

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

    def plot(self, rel_name, xunits=None, yunits=None, **kwargs):
        """
        Plot the data for the given parameters

        Parameters
        ----------
        rel_name: str
            The name of the relation
        xunits: astropy.units.quantity.Quantity
            The units to display
        yunits: astropy.units.quantity.Quantity
            The units to display
        """
        # Get params
        xparam, yparam = self._parse_rel_name(rel_name)

        if not xparam in self.parameters or not yparam in self.parameters:
            raise ValueError("{}, {}: Both parameters need to be in the relation. Try {}".format(xparam, yparam, self.relations))

        # Make the figure
        fig = figure(x_axis_label=xparam, y_axis_label=yparam)
        x, y, _ = self.validate_data(self.data[xparam], self.data[yparam])

        xu = 1
        yu = 1
        if rel_name in self.relations:

            # Get the relation
            rel = self.relations[rel_name]

            # Plot polynomial values
            xu = rel['xunit'].to(xunits) if xunits is not None else rel['xunit'] or 1
            yu = rel['yunit'].to(yunits) if yunits is not None else rel['yunit'] or 1
            fig.line(rel['x_fit'] * xu, rel['y_fit'] * yu, color='black', legend='Fit')

            # # Plot relation error
            # xpat = np.hstack((rel['x_fit'], rel['x_fit'][::-1]))
            # ypat = np.hstack((rel['y_fit'] + rel['y_fit_err'], (rel['y_fit'] - rel['y_fit_err'])[::-1]))
            # err_source = ColumnDataSource(dict(xaxis=xpat, yaxis=ypat))
            # glyph = Patch(x='xaxis', y='yaxis', fill_color='black', line_color=None, fill_alpha=0.1)
            # fig.add_glyph(err_source, glyph)

            # Update axis labels
            fig.xaxis.axis_label = '{}{}'.format(xparam, '[{}]'.format(xunits or rel['xunit']))
            fig.yaxis.axis_label = '{}{}'.format(yparam, '[{}]'.format(yunits or rel['yunit']))

        # Draw points
        fig.circle(x * xu, y * yu, legend='Data', **kwargs)

        return fig

    def validate_data(self, X, Y, Y_unc=None):
        """
        Validate the data for onlu numbers

        Parameters
        ----------
        X: sequence
            The x-array
        Y: sequence
            The y-array
        Y_unc: sequence
            The uncertainty of the y-array

        Returns
        -------
        sequence
            The validated arrays
        """
        if Y_unc is None:
            Y_unc = np.ones_like(Y)

        # Check for valid numbers to plot
        valid = np.asarray([(float(x), float(y), float(y_unc)) for x, y, y_unc in zip(X, Y, Y_unc) if u.isnumber(x) and u.isnumber(y) and u.isnumber(y_unc)]).T

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
        radius = np.polyval(data['coeffs'], spt)*q.R_sun
        radius_unc = np.interp(spt, data['spt'], data['sig_yi'])*q.R_sun

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
        mlty_data['radius'] = (mlty_data['radius'].value * q.Rjup).to(q.R_sun)
        mlty_data['radius_unc'] = (mlty_data['radius_unc'].value * q.Rjup).to(q.R_sun)

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
