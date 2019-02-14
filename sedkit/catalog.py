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

from astropy.io import ascii
import astropy.table as at
import astropy.units as q
import astropy.constants as ac
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource, LabelSet

from .sed import SED
from . import utilities as u


class Catalog:
    """An object to collect SED results for plotting and analysis"""
    def __init__(self, name='SED Catalog', marker='circle', color='blue', **kwargs):
        """Initialize the Catalog object"""
        # Metadata
        self.name = name
        self.marker = marker
        self.color = color
        self.wave_units = q.um
        self.flux_units = q.erg/q.s/q.cm**2/q.AA

        # List all the results columns
        self.cols = ['name', 'age', 'age_unc', 'distance', 'distance_unc',
                     'parallax', 'parallax_unc', 'radius', 'radius_unc',
                     'spectral_type', 'spectral_type_unc', 'SpT', 'SpT_fit',
                     'membership', 'reddening', 'fbol', 'fbol_unc', 'mbol',
                     'mbol_unc', 'Lbol', 'Lbol_unc', 'Lbol_sun',
                     'Lbol_sun_unc', 'Mbol', 'Mbol_unc', 'logg', 'logg_unc',
                     'mass', 'mass_unc', 'Teff', 'Teff_unc', 'Teff_evo',
                     'Teff_evo_unc', 'Teff_bb', 'SED']

        # A master table of all SED results
        self.results = at.QTable(names=self.cols, dtype=['O']*len(self.cols))
        self.results.add_index('name')

        # Set the units
        self.results['age'].unit = q.Gyr
        self.results['age_unc'].unit = q.Gyr
        self.results['distance'].unit = q.pc
        self.results['distance_unc'].unit = q.pc
        self.results['parallax'].unit = q.mas
        self.results['parallax_unc'].unit = q.mas
        self.results['radius'].unit = q.Rsun
        self.results['radius_unc'].unit = q.Rsun
        self.results['fbol'].unit = q.erg/q.s/q.cm**2
        self.results['fbol_unc'].unit = q.erg/q.s/q.cm**2
        self.results['Lbol'].unit = q.erg/q.s
        self.results['Lbol_unc'].unit = q.erg/q.s
        self.results['mass'].unit = q.Msun
        self.results['mass_unc'].unit = q.Msun
        self.results['Teff'].unit = q.K
        self.results['Teff_unc'].unit = q.K
        self.results['Teff_bb'].unit = q.K
        self.results['Teff_evo'].unit = q.K
        self.results['Teff_evo_unc'].unit = q.K

        # Try to set attributes from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __add__(self, other):
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
        if not type(other)==type(self):
            raise TypeError('Cannot add object of type', type(other))

        # Make a new catalog
        new_cat = Catalog()

        # Combine results
        new_cat.results = at.vstack([self.results, other.results])

        return new_cat

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
        results = []
        for col in self.cols[:-1]:

            if col+'_unc' in self.cols:
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

            val = val.to(self.results[col.replace('_unc','')].unit).value if hasattr(val, 'unit') else val

            results.append(val)

        # Add the SED
        results.append(sed)

        # Make the table
        cat = Catalog()
        table = cat.results
        table.add_row(results)
        table = at.Table(table)

        # Append apparent and absolute photometry
        for row in sed.photometry:

            # Add the apparent magnitude
            table.add_column(at.Column([row['app_magnitude']], name=row['band']))

            # Add the apparent uncertainty
            table.add_column(at.Column([row['app_magnitude_unc']], name=row['band']+'_unc'))

            # Add the absolute magnitude
            table.add_column(at.Column([row['abs_magnitude']], name='M_'+row['band']))

            # Add the absolute uncertainty
            table.add_column(at.Column([row['abs_magnitude_unc']], name='M_'+row['band']+'_unc'))

        # Stack with current table
        if len(self.results)==0:
            self.results = table
        else:
            self.results = at.vstack([self.results, table])

    def export(self, parentdir='.', dirname=None, format='ipac',
               sources=True, zipped=False):
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
        final = self.results.filled(np.nan)
        for col in final.colnames:
            final.rename_column(col, col.replace('.', '_'))

        # Write a directory of results and all SEDs...
        if sources:

            # Make a directory
            if not os.path.exists(dirpath):
                os.system('mkdir {}'.format(dirpath))
            else:
                raise IOError('Directory already exists:', dirpath)

            # Export the results table
            resultspath = os.path.join(dirpath,'{}_results.txt'.format(name))
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
            resultspath = dirpath+'_results.txt'
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

    def from_file(self, filepath, methods_list=['find_2MASS'], delimiter=','):
        """Generate a catalog from a file of source names and coordinates

        Parameters
        ----------
        filepath: str
            The path to an ASCII file
        methods_list: list
            A list of methods to run
        delimiter: str
            The column delimiter of the ASCII file
        """
        # Get the table of sources
        data = ascii.read(filepath, delimiter=delimiter)

        # Iterate over table
        for row in data:

            # Make the SED
            s = SED(row['name'], verbose=False)
            if 'ra' in row and 'dec' in row:
                s.sky_coords = row['ra']*q.deg, row['dec']*q.deg

            # Run the desired methods
            s.run_methods(methods_list)

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
            print('Could not retrieve SED', name_or_idx)

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

    def plot(self, x, y, marker=None, color=None, scale=['linear','linear'],
             xlabel=None, ylabel=None, fig=None, order=None, data=None,
             identify=None, id_color='red', label_points=False, **kwargs):
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
        data: dict
            Additional data to add to the plot
        identify: idx, str, sequence
            Names of sources to highlight in the plot
        id_color: str
            The color of the identified points
        """
        # Make the figure
        if fig is None:

            # Set up hover tool
            tips = [('Name', '@desc'), (x, '@x'), (y, '@y')]
            hover = HoverTool(tooltips=tips, names=['points'])

            # Make the plot
            TOOLS = ['pan', 'reset', 'box_zoom', 'save', hover]
            title = '{} v {}'.format(x,y)
            fig = figure(plot_width=800, plot_height=500, title=title, 
                         y_axis_type=scale[1], x_axis_type=scale[0], 
                         tools=TOOLS)

        # Make sure marker is legit
        size = kwargs.get('size', 8)
        kwargs['size'] = size
        marker = getattr(fig, marker or self.marker)
        color = color or self.color

        # Get manually input data
        if isinstance(data, dict):
            names = data['name']
            xdata = data[x]
            xerror = data.get('{}_unc'.format(x), np.zeros(len(xdata)))
            xunit = data.get('{}_unit'.format(x))
            ydata = data[y]
            yerror = data.get('{}_unc'.format(y), np.zeros(len(ydata)))
            yunit = data.get('{}_unit'.format(y))

        # Or catalog data
        else:
            # Get the data
            (xdata, xerror, xunit), (ydata, yerror, yunit) = self.get_data(x, y)

            # Get the source names
            names = self.results['name'] 

        # Set axis labels
        fig.xaxis.axis_label = '{}{}'.format(x, ' [{}]'.format(xunit) if xunit else '')
        fig.yaxis.axis_label = '{}{}'.format(y, ' [{}]'.format(yunit) if yunit else '')

        # Plot points with tips
        source = ColumnDataSource(data=dict(x=xdata, y=ydata, desc=names))
        marker('x', 'y', source=source, color=color, fill_alpha=0.7, name='points', **kwargs)

        # Add errorbars
        u.errorbars(fig, xdata, ydata, xerr=xerror, yerr=yerror, color=color)

        # Label points
        if label_points:
            labels = LabelSet(x='x', y='y', text='desc', level='glyph', x_offset=5, y_offset=5, source=source, render_mode='canvas')
            fig.add_layout(labels)

        # Fit polynomial
        if isinstance(order, int):
            # Make into arrays
            idx = [n for n, (i, j, k) in enumerate(zip(xdata.data, ydata.data, yerror.data)) if not hasattr(i, 'mask') and not hasattr(j, 'mask') and not hasattr(k, 'mask')]
            xdata = np.array(xdata.data, dtype=float)[idx]
            ydata = np.array(ydata.data, dtype=float)[idx]
            yerror = np.array(yerror.data, dtype=float)[idx]

            # Fit the polynomial
            coeffs = np.polyfit(x=xdata, y=ydata, deg=order, w=1/yerror)
            label = 'Order {} fit'.format(order)
            xaxis = np.linspace(min(xdata), max(xdata), 100)
            yaxis = np.polyval(coeffs, xaxis)
            fig.line(xaxis, yaxis, legend=label, color=color)

        # Formatting
        fig.legend.location = "top_right"
        fig.legend.click_policy = "hide"

        # Identify sources
        if isinstance(identify, list):
            id_cat = Catalog('Identified')
            for obj_id in identify:
                id_cat.add_SED(self.get_SED(obj_id))
            fig = id_cat.plot(x, y, fig=fig, size=size+5, marker='circle', line_color=id_color, fill_color=None, line_width=2, label_points=True)
            del id_cat

        return fig

    def plot_SEDs(self, name_or_idx, scale=['log', 'log'], normalized=False, **kwargs):
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
        TOOLS = ['pan', 'reset', 'box_zoom', 'save']
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
            fig = targ.plot(fig=fig, color=c, output=True, legend=targ.name, **kwargs)

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
            print('Could not remove SED', name_or_idx)

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

            print('Catalog saved to',file)
