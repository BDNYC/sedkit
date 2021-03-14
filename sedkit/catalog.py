#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
A module to produce a catalog of spectral energy distributions
"""

import os
import pickle
from copy import copy
import shutil

from astropy.io import ascii
import astropy.table as at
import astropy.units as q
import numpy as np
from bokeh.models import HoverTool, ColumnDataSource, LabelSet
from bokeh.plotting import figure, show
from bokeh.models.glyphs import Patch

from .sed import SED
from . import utilities as u


class Catalog:
    """An object to collect SED results for plotting and analysis"""
    def __init__(self, name='SED Catalog', marker='circle', color='blue', verbose=True, **kwargs):
        """Initialize the Catalog object"""
        # Metadata
        self.verbose = verbose
        self.name = name
        self.marker = marker
        self.color = color
        self.wave_units = q.um
        self.flux_units = q.erg/q.s/q.cm**2/q.AA

        # List all the results columns
        self.cols = ['name', 'ra', 'dec', 'age', 'age_unc', 'distance', 'distance_unc',
                     'parallax', 'parallax_unc', 'radius', 'radius_unc',
                     'spectral_type', 'spectral_type_unc', 'SpT',
                     'membership', 'reddening', 'fbol', 'fbol_unc', 'mbol',
                     'mbol_unc', 'Lbol', 'Lbol_unc', 'Lbol_sun',
                     'Lbol_sun_unc', 'Mbol', 'Mbol_unc', 'logg', 'logg_unc',
                     'mass', 'mass_unc', 'Teff', 'Teff_unc', 'Teff_evo',
                     'Teff_evo_unc', 'Teff_bb', 'SED']

        # A master table of all SED results
        self.results = self.make_results_table(self)

        # Try to set attributes from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __add__(self, other, name=None):
        """Add two catalogs together

        Parameters
        ----------
        other: sedkit.catalog.Catalog
            The Catalog to add

        Returns
        -------
        sedkit.catalog.Catalog
            The combined catalog
        """
        if not type(other) == type(self):
            raise TypeError('Cannot add object of type {}'.format(type(other)))

        # Make a new catalog
        new_cat = Catalog(name=name or self.name)

        # Combine results
        new_results = at.vstack([at.Table(self.results), at.Table(other.results)])
        new_cat.results = new_results

        return new_cat

    def add_column(self, name, data, unc=None):
        """
        Add a column of data to the results table

        Parameters
        ----------
        name: str
            The name of the new column
        data: sequence
            The data array
        unc: sequence (optional)
            The uncertainty array
        """
        # Make sure column doesn't exist
        if name in self.results.colnames:
            raise ValueError("{}: Column already exists.".format(name))

        # Make sure data is the right length
        if len(data) != len(self.results):
            raise ValueError("{} != {}: Data is not the right size for this catalog.".format(len(data), len(self.results)))

        # Add the column
        self.results.add_column(data, name=name)

        # Add uncertainty column
        if unc is not None:

            # Uncertainty name
            name = name + '_unc'

            # Make sure column doesn't exist
            if name in self.results.colnames:
                raise ValueError("{}: Column already exists.".format(name))

            # Make sure data is the right length
            if len(unc) != len(self.results):
                raise ValueError(
                    "{} != {}: Data is not the right size for this catalog.".format(len(unc), len(self.results)))

            # Add the column
            self.results.add_column(unc, name=name)

    def add_SED(self, sed):
        """Add an SED to the catalog

        Parameters
        ----------
        sed: sedkit.sed.SED
            The SED object to add
        """
        # Turn off print statements
        sed.verbose = False

        # Check the units
        sed.wave_units = self.wave_units
        sed.flux_units = self.flux_units

        # Run the SED
        sed.make_sed()

        # Add the values and uncertainties if applicable
        new_row = {}
        for col in self.cols[:-1]:

            if col + '_unc' in self.cols:
                if isinstance(getattr(sed, col), tuple):
                    val = getattr(sed, col)[0]
                else:
                    val = None
            elif col.endswith('_unc'):
                if isinstance(getattr(sed, col.replace('_unc', '')), tuple):
                    val = getattr(sed, col.replace('_unc', ''))[1]
                else:
                    val = None
            else:
                val = getattr(sed, col)

            val = val.to(self.results[col.replace('_unc', '')].unit).value if hasattr(val, 'unit') else val

            new_row[col] = val

        # Add the SED
        new_row['SED'] = sed

        # Append apparent and absolute photometry
        for row in sed.photometry:

            # Add the column to the results table
            if row['band'] not in self.results.colnames:
                self.results.add_column(at.Column([np.nan] * len(self.results), dtype=np.float16, name=row['band']))
                self.results.add_column(at.Column([np.nan] * len(self.results), dtype=np.float16, name=row['band'] + '_unc'))
                self.results.add_column(at.Column([np.nan] * len(self.results), dtype=np.float16, name='M_' + row['band']))
                self.results.add_column(at.Column([np.nan] * len(self.results), dtype=np.float16, name='M_' + row['band'] + '_unc'))

            # Add the apparent magnitude
            new_row[row['band']] = row['app_magnitude']

            # Add the apparent uncertainty
            new_row[row['band'] + '_unc'] = row['app_magnitude_unc']

            # Add the absolute magnitude
            new_row['M_' + row['band']] = row['abs_magnitude']

            # Add the absolute uncertainty
            new_row['M_' + row['band'] + '_unc'] = row['abs_magnitude_unc']

        # Add the new row
        self.results.add_row(new_row)

        self.message("Successfully added SED '{}'".format(sed.name))

    def export(self, parentdir='.', dirname=None, format='ipac', sources=True, zipped=False):
        """
        Exports the results table and a directory of all SEDs

        Parameters
        ----------
        parentdir: str
            The parent directory for the folder or zip file
        dirname: str (optional)
            The name of the exported directory or zip file, default is SED name
        format: str
            The format of the output results table
        sources: bool
            Export a directory of all source SEDs too
        zipped: bool
            Zip the directory
        """
        # Check the parent directory
        if not os.path.exists(parentdir):
            raise IOError('No such target directory', parentdir)

        # Check the target directory
        name = self.name.replace(' ', '_')
        dirname = dirname or name
        dirpath = os.path.join(parentdir, dirname)

        # Remove '.' from column names
        final = at.Table(self.results).filled(np.nan)
        for col in final.colnames:
            final.rename_column(col, col.replace('.', '_').replace('/', '_'))

        # Write a directory of results and all SEDs...
        if sources:

            # Make a directory
            if not os.path.exists(dirpath):
                os.system('mkdir {}'.format(dirpath))
            else:
                raise IOError('Directory already exists:', dirpath)

            # Export the results table
            resultspath = os.path.join(dirpath, '{}_results.txt'.format(name))
            final.write(resultspath, format=format)

            # Make a sources directory
            sourcedir = os.path.join(dirpath,'sources')
            os.system('mkdir {}'.format(sourcedir))

            # Export all SEDs
            for source in self.results['SED']:
                source.export(sourcedir)

            # zip if desired
            if zipped:
                shutil.make_archive(dirpath, 'zip', dirpath)
                os.system('rm -R {}'.format(dirpath))

        # ...or just write the results table
        else:
            resultspath = dirpath + '_results.txt'
            final.write(resultspath, format=format)

    def filter(self, param, value):
        """Retrieve the filtered rows

        Parameters
        ----------
        param: str
            The parameter to filter by, e.g. 'Teff'
        value: str, float, int, sequence
            The criteria to filter by, 
            which can be single valued like 1400
            or a range with operators [<,<=,>,>=],
            e.g. (>1200,<1400), ()

        Returns
        -------
        sedkit.sed.Catalog
            The filtered catalog
        """
        # Make a new catalog
        cat = Catalog()
        cat.results = u.filter_table(self.results, **{param: value})

        return cat

    def from_file(self, filepath, run_methods=['find_2MASS'], delimiter=','):
        """Generate a catalog from a file of source names and coordinates

        Parameters
        ----------
        filepath: str
            The path to an ASCII file
        run_methods: list
            A list of methods to run
        delimiter: str
            The column delimiter of the ASCII file
        """
        # Get the table of sources
        data = ascii.read(filepath, delimiter=delimiter)

        self.message("Generating SEDs for {} sources from {}".format(len(data), filepath))

        # Iterate over table
        for row in data:

            # Make the SED
            s = SED(row['name'], verbose=False)
            if 'ra' in row and 'dec' in row:
                s.sky_coords = row['ra']*q.deg, row['dec']*q.deg

            # Run the desired methods
            s.run_methods(run_methods)

            # Add it to the catalog
            self.add_SED(s)

    def get_data(self, *args):
        """Fetch the data for the given columns
        """
        results = []

        for x in args:

            # Get the data
            if '-' in x:
                x1, x2 = x.split('-')
                if self.results[x1].unit != self.results[x2].unit:
                    raise TypeError('Columns must be the same units.')

                xunit = self.results[x1].unit
                xdata = self.results[x1] - self.results[x2]
                xerror = np.sqrt(self.results['{}_unc'.format(x1)]**2 + self.results['{}_unc'.format(x2)]**2)

            else:
                xunit = self.results[x].unit
                xdata = self.results[x]
                xerror = self.results['{}_unc'.format(x)]

            # Append to results
            results.append([xdata, xerror, xunit])

        return results

    def get_SED(self, name_or_idx):
        """Retrieve the SED for the given object

        Parameters
        ----------
        idx_or_name: str, int
            The name or index of the SED to get
        """
        # Add the index
        self.results.add_index('name')

        # Get the rows
        if isinstance(name_or_idx, str) and name_or_idx in self.results['name']:
            return copy(self.results.loc[name_or_idx]['SED'])

        elif isinstance(name_or_idx, int) and name_or_idx <= len(self.results):
            return copy(self.results[name_or_idx]['SED'])

        else:
            self.message('Could not retrieve SED {}'.format(name_or_idx))

            return

    def load(self, file):
        """Load a saved Catalog"""
        if os.path.isfile(file):

            f = open(file)
            cat = pickle.load(f)
            f.close()

            f = open(file, 'rb')
            cat = pickle.load(f)
            f.close()

            self.results = cat

    @staticmethod
    def make_results_table(self):
        """Generate blank results table"""
        results = at.QTable(names=self.cols, dtype=['O'] * len(self.cols))
        results.add_index('name')

        # Set the units
        results['age'].unit = q.Gyr
        results['age_unc'].unit = q.Gyr
        results['distance'].unit = q.pc
        results['distance_unc'].unit = q.pc
        results['parallax'].unit = q.mas
        results['parallax_unc'].unit = q.mas
        results['radius'].unit = q.Rsun
        results['radius_unc'].unit = q.Rsun
        results['fbol'].unit = q.erg / q.s / q.cm ** 2
        results['fbol_unc'].unit = q.erg / q.s / q.cm ** 2
        results['Lbol'].unit = q.erg / q.s
        results['Lbol_unc'].unit = q.erg / q.s
        results['mass'].unit = q.Msun
        results['mass_unc'].unit = q.Msun
        results['Teff'].unit = q.K
        results['Teff_unc'].unit = q.K
        results['Teff_bb'].unit = q.K
        results['Teff_evo'].unit = q.K
        results['Teff_evo_unc'].unit = q.K

        return results

    def message(self, msg, pre='[sedkit.Catalog]'):
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

    def plot(self, x, y, marker=None, color=None, scale=['linear','linear'],
             xlabel=None, ylabel=None, fig=None, order=None, identify=None,
             id_color='red', label_points=False, exclude=None, draw=True, **kwargs):
        """Plot parameter x versus parameter y

        Parameters
        ----------
        x: str
             The name of the x axis parameter, e.g. 'SpT'
        y: str
             The name of the y axis parameter, e.g. 'Teff'
        marker: str (optional)
             The name of the method for the desired marker
        color: str (optional)
             The color to use for the points
        scale: sequence
             The (x,y) scale for the plot
        xlabel: str
             The label for the x-axis
        ylable : str
             The label for the y-axis 
        fig: bokeh.plotting.figure (optional)
             The figure to plot on
        order: int
             The polynomial order to fit
        identify: idx, str, sequence
             Names of sources to highlight in the plot
        id_color: str
             The color of the identified points
        label_points: bool
             Print the name of the object next to the point

        Returns
        -------
        bokeh.plotting.figure.Figure
             The figure object
        """
        # Grab the source and valid params
        source = copy(self.source)
        params = [k for k in source.column_names if not k.endswith('_unc')]

        # If no uncertainty column for parameter, add it
        if '{}_unc'.format(x) not in source.column_names:
            _ = source.add([None] * len(self.source.data['name']), '{}_unc'.format(x))
        if '{}_unc'.format(y) not in source.column_names:
            _ = source.add([None] * len(self.source.data['name']), '{}_unc'.format(y))

        # Check if the x parameter is a color
        if '-' in x and all([i in params for i in x.split('-')]):
            colordata = self.get_data(x)[0]
            if len(colordata) == 3:
                _ = source.add(colordata[0], x)
                _ = source.add(colordata[1], '{}_unc'.format(x))
                params.append(x)

        # Check if the y parameter is a color
        if '-' in y and all([i in params for i in y.split('-')]):
            colordata = self.get_data(y)[0]
            if len(colordata) == 3:
                _ = source.add(colordata[0], y)
                _ = source.add(colordata[1], '{}_unc'.format(y))
                params.append(y)

        # Check the params are in the table
        if x not in params:
            raise ValueError("'{}' is not a valid x parameter. Please choose from {}".format(x, params))
        if y not in params:
            raise ValueError("'{}' is not a valid y parameter. Please choose from {}".format(y, params))

        # Make the figure
        if fig is None:

            # Tooltip names can't have '.' or '-'
            xname = source.add(source.data[x], x.replace('.', '_').replace('-', '_'))
            yname = source.add(source.data[y], y.replace('.', '_').replace('-', '_'))

            # Set up hover tool
            tips = [('Name', '@name'), (x, '@{}'.format(xname)), (y, '@{}'.format(yname))]
            hover = HoverTool(tooltips=tips, names=['points'])

            # Make the plot
            TOOLS = ['pan', 'reset', 'box_zoom', 'wheel_zoom', 'save', hover]
            title = '{} v {}'.format(x, y)
            fig = figure(plot_width=800, plot_height=500, title=title, y_axis_type=scale[1], x_axis_type=scale[0], tools=TOOLS)

        # # Exclude sources
        # if exclude is not None:
        #     exc_idx = [i for i, v in enumerate(source.data['name']) if v in exclude]
        #     patches = {x : [(i, np.nan) for i in exc_idx],
        #                y : [(i, np.nan) for i in exc_idx]}
        #
        #     source.patch(patches)

        # Get marker class
        size = kwargs.get('size', 8)
        kwargs['size'] = size
        marker = getattr(fig, marker or self.marker)
        color = color or self.color
        marker(x, y, source=source, color=color, fill_alpha=0.7, name='points', **kwargs)

        # Plot y errorbars
        yval, yerr = source.data[y], source.data['{}_unc'.format(y)]
        yval[yval == None] = np.nan
        yerr[yerr == None] = np.nan
        y_err_x = [(i, i) for i in source.data[x]]
        y_err_y = [(i, j) for i, j in zip(yval - yerr, yval + yerr)]
        fig.multi_line(y_err_x, y_err_y, color=color)

        # Plot x errorbars
        xval, xerr = source.data[x], source.data['{}_unc'.format(x)]
        xval[xval == None] = np.nan
        xerr[xerr == None] = np.nan
        x_err_y = [(i, i) for i in source.data[y]]
        x_err_x = [(i, j) for i, j in zip(xval - xerr, xval + xerr)]
        fig.multi_line(x_err_x, x_err_y, color=color)

        # Label points
        if label_points:
            labels = LabelSet(x=x, y=y, text='name', level='glyph', x_offset=5, y_offset=5, source=source, render_mode='canvas')
            fig.add_layout(labels)

        # Fit polynomial
        if isinstance(order, int):

            # Only fit valid values
            idx = [n for n, (i, j) in enumerate(zip(xval, yval)) if not hasattr(i, 'mask') and not np.isnan(i) and not hasattr(j, 'mask') and not np.isnan(j)]
            xd = np.array(xval, dtype=float)[idx]
            yd = np.array(yval, dtype=float)[idx]

            # Plot data
            label = 'Order {} fit'.format(order)
            xaxis = np.linspace(min(xd), max(xd), 100)
            coeffs = None

            # Fit the polynomial
            try:

                if yerr is not None:
                    ye = np.array(yerr, dtype=float)[idx]
                    coeffs, cov = np.polyfit(x=xd, y=yd, deg=order, w=1./ye, cov=True)
                else:
                    coeffs, cov = np.polyfit(x=xd, y=yd, deg=order, cov=True)

                # Plot the line
                if coeffs is None or any([np.isnan(i) for i in coeffs]):
                    self.message("Could not fit that data with an order {} polynomial".format(order))
                else:

                    # Calculate values and 1-sigma
                    TT = np.vstack([xaxis**(order-i) for i in range(order + 1)]).T
                    yaxis = np.dot(TT, coeffs)
                    C_yi = np.dot(TT, np.dot(cov, TT.T))
                    sig = np.sqrt(np.diag(C_yi))

                    # Plot the line and shaded error
                    fig.line(xaxis, yaxis, legend=label + ' {}'.format(coeffs[::-1]), color=color, line_alpha=0.3)
                    xpat = np.hstack((xaxis, xaxis[::-1]))
                    ypat = np.hstack((yaxis + sig, (yaxis - sig)[::-1]))
                    err_source = ColumnDataSource(dict(xaxis=xpat, yaxis=ypat))
                    glyph = Patch(x='xaxis', y='yaxis', fill_color=color, line_color=None, fill_alpha=0.1)
                    fig.add_glyph(err_source, glyph)

            except Exception as exc:
                print("Skipping the polynomial fit: {}".format(exc))

        # Set axis labels
        xunit = source.data[x].unit
        yunit = source.data[y].unit
        fig.xaxis.axis_label = '{}{}'.format(x, ' [{}]'.format(xunit) if xunit else '')
        fig.yaxis.axis_label = '{}{}'.format(y, ' [{}]'.format(yunit) if yunit else '')

        # Formatting
        fig.legend.location = "top_right"

        # Identify sources
        if isinstance(identify, list):
            id_cat = Catalog('Identified')
            for obj_id in identify:
                obj_result = self.get_SED(obj_id)
                if str(type(obj_result)) != "<class 'astropy.table.column.Column'>":
                    obj_result = [obj_result]
                for obj in obj_result:
                    id_cat.add_SED(obj)
            fig = id_cat.plot(x, y, fig=fig, size=size+5, marker='circle', line_color=id_color, fill_color=None, line_width=2, label_points=True)
            del id_cat

        if draw:
            show(fig)

        return fig

    def plot_SEDs(self, name_or_idx, scale=['log', 'log'], normalize=None, **kwargs):
        """Plot the SED for the given object or objects

        Parameters
        ----------
        idx_or_name: str, int, sequence
            The name or index of the SED to get
        scale: sequence
            The [x, y] scale to plot, ['linear', 'log']
        normalized: bool
            Normalize the SEDs to 1
        """
        COLORS = u.color_gen('Category10')

        # Plot all SEDS
        if name_or_idx in ['all', '*']:
            name_or_idx = list(range(len(self.results)))

        # Make it into a list
        if isinstance(name_or_idx, (str, int)):
            name_or_idx = [name_or_idx]

        # Make the plot
        TOOLS = ['pan', 'reset', 'box_zoom', 'wheel_zoom', 'save']
        title = self.name
        fig = figure(plot_width=800, plot_height=500, title=title,
                     y_axis_type=scale[1], x_axis_type=scale[0],
                     x_axis_label='Wavelength [{}]'.format(self.wave_units),
                     y_axis_label='Flux Density [{}]'.format(str(self.flux_units)),
                     tools=TOOLS)

        # Plot each SED
        for obj in name_or_idx:
            c = next(COLORS)
            targ = self.get_SED(obj)
            fig = targ.plot(fig=fig, color=c, output=True, normalize=normalize, legend=targ.name, **kwargs)

        return fig

    def remove_SED(self, name_or_idx):
        """Remove an SED from the catalog

        Parameters
        ----------
        name_or_idx: str, int
            The name or index of the SED to remove
        """
        # Add the index
        self.results.add_index('name')

        # Get the rows
        if isinstance(name_or_idx, str) and name_or_idx in self.results['name']:
            self.results = self.results[self.results['name'] != name_or_idx]

        elif isinstance(name_or_idx, int) and name_or_idx <= len(self.results):
            self.results.remove_row([name_or_idx])

        else:
            self.message('Could not remove SED {}'.format(name_or_idx))

            return

    def save(self, file):
        """Save the serialized data

        Parameters
        ----------
        file: str
            The filepath
        """
        path = os.path.dirname(file)

        if os.path.exists(path):

            # Make the file if necessary
            if not os.path.isfile(file):
                os.system('touch {}'.format(file))

            # Write the file
            f = open(file, 'wb')
            pickle.dump(self.results, f, pickle.HIGHEST_PROTOCOL)
            f.close()

            self.message('Catalog saved to {}'.format(file))

    @property
    def source(self):
        """Generates a ColumnDataSource from the results table"""
        # Remove SED column
        results_dict = {key: val for key, val in dict(self.results).items() if key != 'SED'}

        return ColumnDataSource(data=results_dict)
