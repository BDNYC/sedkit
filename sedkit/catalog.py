#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
A module to produce a catalog of spectral energy distributions
"""

import os
import dill
import pickle
from copy import copy
import importlib.resources
import shutil

from astropy.io import ascii
import astropy.table as at
import astropy.units as q
import numpy as np
from bokeh.models import HoverTool, ColumnDataSource, LabelSet, TapTool, CustomJS
from bokeh.plotting import figure, show
from bokeh.models.glyphs import Patch
from bokeh.layouts import Row

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
        self.palette = 'viridis'
        self.wave_units = q.um
        self.flux_units = q.erg/q.s/q.cm**2/q.AA
        self.array_cols = ['sky_coords', 'SED', 'app_spec_SED', 'abs_spec_SED', 'app_phot_SED', 'abs_phot_SED', 'app_specphot_SED', 'abs_specphot_SED', 'app_SED', 'abs_SED', 'spectra']
        self.phot_cols = []

        # List all the results columns
        self.cols = ['name', 'age', 'age_unc', 'distance', 'distance_unc',
                     'parallax', 'parallax_unc', 'radius', 'radius_unc',
                     'spectral_type', 'spectral_type_unc', 'SpT',
                     'membership', 'reddening', 'fbol', 'fbol_unc', 'mbol',
                     'mbol_unc', 'Lbol', 'Lbol_unc', 'Lbol_sun',
                     'Lbol_sun_unc', 'Mbol', 'Mbol_unc', 'logg', 'logg_unc',
                     'mass', 'mass_unc', 'Teff', 'Teff_unc']

        # A master table of all SED results
        self._results = self.make_results_table()

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
        new_results = at.vstack([at.Table(self._results), at.Table(other._results)])
        new_cat._results = new_results

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
        if name in self._results.colnames:
            raise ValueError("{}: Column already exists.".format(name))

        # Make sure data is the right length
        if len(data) != len(self._results):
            raise ValueError("{} != {}: Data is not the right size for this catalog.".format(len(data), len(self._results)))

        # Add the column
        self._results.add_column(data, name=name)

        # Add uncertainty column
        if unc is not None:

            # Uncertainty name
            name = name + '_unc'

            # Make sure column doesn't exist
            if name in self._results.colnames:
                raise ValueError("{}: Column already exists.".format(name))

            # Make sure data is the right length
            if len(unc) != len(self._results):
                raise ValueError(
                    "{} != {}: Data is not the right size for this catalog.".format(len(unc), len(self._results)))

            # Add the column
            self._results.add_column(unc, name=name)

    def add_SED(self, sed):
        """Add an SED to the catalog

        Parameters
        ----------
        sed: sedkit.sed.SED
            The SED object to add
        """
        # Overwrite duplicate names
        idx = None
        if sed.name in self.results['name']:
            self.message("{}: Target already in catalog. Overwriting with new SED...".format(sed.name))
            idx = np.where(self.results['name'] == sed.name)[0][0]

        # Turn off print statements
        sed.verbose = False

        # Check the units
        sed.wave_units = self.wave_units
        sed.flux_units = self.flux_units

        # Run the SED
        sed.make_sed()

        # Add the values and uncertainties if applicable
        new_row = {}
        for col in self.cols:

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

        # Store the spectra
        new_row['spectra'] = [spec['spectrum'] for spec in sed.spectra]

        # Store the SED arrays
        for pre in ['app', 'abs']:
            for dat in ['phot_', 'spec_', 'specphot_', '']:
                sed_name = '{}_{}SED'.format(pre, dat)
                new_row[sed_name] = getattr(sed, sed_name).spectrum if getattr(sed, sed_name) is not None else None

        # Add the SED
        new_row['SED'] = sed

        # Append apparent and absolute photometry
        for row in sed.photometry:

            # Add the column to the results table
            if row['band'] not in self._results.colnames:
                self._results.add_column(at.Column([None] * len(self._results), dtype='O', name=row['band']))
                self._results.add_column(at.Column([None] * len(self._results), dtype='O', name=row['band'] + '_unc'))
                self._results.add_column(at.Column([None] * len(self._results), dtype='O', name='M_' + row['band']))
                self._results.add_column(at.Column([None] * len(self._results), dtype='O', name='M_' + row['band'] + '_unc'))
                self.phot_cols += [row['band']]

            # Add the apparent magnitude
            if u.isnumber(row['app_magnitude']):
                new_row[row['band']] = row['app_magnitude']

                # Add the apparent uncertainty
                new_row['{}_unc'.format(row['band'])] = None if np.isnan(row['app_magnitude_unc']) else row['app_magnitude_unc']

                # Add the absolute magnitude
                new_row['M_{}'.format(row['band'])] = None if np.isnan(row['abs_magnitude']) else row['abs_magnitude']

                # Add the absolute uncertainty
                new_row['M_{}_unc'.format(row['band'])] = None if np.isnan(row['abs_magnitude_unc']) else row['abs_magnitude_unc']

        # Ensure missing photometry columns are None
        for band in self.phot_cols:
            if band not in sed.photometry['band']:
                new_row[band] = None
                new_row['{}_unc'.format(band)] = None
                new_row['M_{}'.format(band)] = None
                new_row['M_{}_unc'.format(band)] = None

        # Add the new row to the end of the list...
        if idx is None:
            self._results.add_row(new_row)

        # ...or replace the existing row
        else:
            self._results.remove_row(idx)
            self._results.insert_row(idx, new_row)

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
            for source in self._results['SED']:
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

        # If it's a list, just get the rows in the list
        if isinstance(value, (list, np.ndarray)):
            cat._results = self._results[[idx for idx, val in enumerate(self._results[param]) if val in value]]

        else:
            cat._results = u.filter_table(self._results, **{param: value})

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

            try:

                # Make the SED
                s = SED(row['name'], verbose=False)
                if 'ra' in row and 'dec' in row:
                    s.sky_coords = row['ra'] * q.deg, row['dec'] * q.deg

                # Run the desired methods
                s.run_methods(run_methods)

                # Add it to the catalog
                self.add_SED(s)

            except:
                self.message("Could not add SED '{}".format(row['name']))

    def get_data(self, *args):
        """Fetch the data for the given columns
        """
        data = []
        
        # Fill results table
        results = self.results.filled(np.nan)

        for x in args:

            # Get the data
            if '-' in x:
                x1, x2 = x.split('-')
                if results[x1].unit != results[x2].unit:
                    raise TypeError('Columns must be the same units.')

                xunit = results[x1].unit
                xdata = np.array(results[x1].tolist()) - np.array(results[x2].tolist())
                xerr1 = np.array(results['{}_unc'.format(x1)].tolist())
                xerr2 = np.array(results['{}_unc'.format(x2)].tolist())
                xerror = np.sqrt(xerr1**2 + xerr2**2)

            else:
                xunit = results[x].unit
                xdata = np.array(results[x].value.tolist()) if hasattr(results[x], 'unit') else np.array(results[x].tolist())
                xerror = np.array(results['{}_unc'.format(x)].value.tolist()) if hasattr(results['{}_unc'.format(x)], 'unit') else np.array(results['{}_unc'.format(x)].tolist())

            # Append to results
            data.append([xdata, xerror, xunit])

        return data

    def get_SED(self, name_or_idx):
        """Retrieve the SED for the given object

        Parameters
        ----------
        idx_or_name: str, int
            The name or index of the SED to get
        """
        # Add the index
        self._results.add_index('name')

        # Get the rows
        if isinstance(name_or_idx, str) and name_or_idx in self._results['name']:
            return copy(self._results[self._results['name'] == name_or_idx]['SED'][0])

        elif isinstance(name_or_idx, int) and name_or_idx <= len(self._results):
            return copy(self._results[name_or_idx]['SED'])

        else:
            self.message('Could not retrieve SED {}'.format(name_or_idx))

        return

    def generate_SEDs(self, table):
        """
        Generate SEDs from a Catalog results table

        Parameters
        ----------
        table: astropy.table.QTable
            The table of data to use

        Returns
        -------
        sequence
            The list of SEDs for each row in the input table
        """
        sed_list = []
        t = self.make_results_table()
        for row in table:
            s = SED(row['name'], verbose=False)

            for att in ['age', 'parallax', 'radius', 'spectral_type']:
                setattr(self, att, (row[att] * t[att].unit, row['{}_unc'.format(att)] * t[att].unit) if row[att] is not None else None)

            s.sky_coords = row['sky_coords']
            s.membership = row['membership']
            s.reddening = row['reddening']

            # Add spectra
            for spec in row['spectra']:
                s.add_spectrum(spec)

            # Add photometry
            for col in row.colnames:
                if '.' in col and not col.startswith('M_') and not col.endswith('_unc'):
                    if row[col] is not None and not np.isnan(row[col]):
                        s.add_photometry(col, float(row[col]), float(row['{}_unc'.format(col)]))

            # Make the SED
            s.make_sed()

            # Add SED object to the list
            sed_list.append(s)
            del s

        return sed_list

    def load(self, file, make_seds=False):
        """
        Load a saved Catalog

        Parameters
        ----------
        file: str
            The file to load
        """
        if os.path.isfile(file):

            # Open the file
            f = open(file, 'rb')
            results = pickle.load(f)
            f.close()

            # Make SEDs again
            if make_seds:
                seds = self.generate_SEDs(results)
                results.add_column(seds, name='SED')

            # Set results attribute
            self._results = results

            self.message("Catalog loaded from {}".format(file))

        else:

            self.message("Could not load Catalog from {}".format(file))

    def make_results_table(self):
        """Generate blank results table"""
        all_cols = self.cols + self.array_cols
        results = at.QTable(names=all_cols, masked=True, dtype=['O'] * len(all_cols))
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

        return results

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

    def iplot(self, x, y, marker=None, color=None, scale=['linear','linear'],
             xlabel=None, ylabel=None, draw=True, order=None, **kwargs):
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
            _ = source.add([None] * len(source.data['name']), '{}_unc'.format(x))
        if '{}_unc'.format(y) not in source.column_names:
            _ = source.add([None] * len(source.data['name']), '{}_unc'.format(y))

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

        # Tooltip names can't have '.' or '-'
        xname = source.add(source.data[x], x.replace('.', '_').replace('-', '_'))
        yname = source.add(source.data[y], y.replace('.', '_').replace('-', '_'))

        # Make photometry source
        phot_source = ColumnDataSource(data={'phot_wave': [], 'phot': []})
        phot_data = [row['app_phot_SED'][:2] if row['app_phot_SED'] is not None else [[], []] for row in self._results]
        phot_len = max([len(i[0]) for i in phot_data])
        for idx, row in enumerate(self._results):
            w, f = phot_data[idx]
            w = np.concatenate([w, np.zeros(phot_len - len(w)) * np.nan])
            f = np.concatenate([f, np.zeros(phot_len - len(f)) * np.nan])
            _ = phot_source.add(w, 'phot_wave{}'.format(idx))
            _ = phot_source.add(f, 'phot{}'.format(idx))

        # Make spectra source
        spec_source = ColumnDataSource(data={'spec_wave': [], 'spec': []})
        spec_data = [row['app_spec_SED'][:2] if row['app_spec_SED'] is not None else [[], []] for row in self._results]
        spec_len = max([len(i[0]) for i in spec_data])
        for idx, row in enumerate(self._results):
            w, f = spec_data[idx]
            w = np.concatenate([w, np.zeros(spec_len - len(w)) * np.nan])
            f = np.concatenate([f, np.zeros(spec_len - len(f)) * np.nan])
            _ = spec_source.add(w, 'spec_wave{}'.format(idx))
            _ = spec_source.add(f, 'spec{}'.format(idx))

        # Set up hover tool
        tips = [('Name', '@name'), (x, '@{}'.format(xname)), (y, '@{}'.format(yname))]
        hover = HoverTool(tooltips=tips, name='points')

        callback = CustomJS(args=dict(source=source, phot_source=phot_source, spec_source=spec_source), code="""
            var data = source.data;
            var phot_data = phot_source.data;
            var spec_data = spec_source.data;
            var selected = source.selected.indices;
            phot_source.data['phot_wave'] = phot_data['phot_wave' + selected[0]];
            phot_source.data['phot'] = phot_data['phot' + selected[0]];
            phot_source.change.emit();
            spec_source.data['spec_wave'] = spec_data['spec_wave' + selected[0]];
            spec_source.data['spec'] = spec_data['spec' + selected[0]];
            spec_source.change.emit();
            """)
        tap = TapTool(callback=callback)

        # Make the plot
        TOOLS = ['pan', 'reset', 'box_zoom', 'wheel_zoom', 'save', hover, tap]
        title = '{} v {}'.format(x, y)
        fig = figure(width=500, height=500, title=title, y_axis_type=scale[1], x_axis_type=scale[0], tools=TOOLS)

        # Get marker class
        size = kwargs.get('size', 8)
        kwargs['size'] = size
        marker = getattr(fig, marker or self.marker)
        color = color or self.color

        # Plot nominal values and errors
        marker(x, y, source=source, color=color, fill_alpha=0.7, name='points', **kwargs)
        fig = u.errorbars(fig, x, y, xerr='{}_unc'.format(x), yerr='{}_unc'.format(y), source=source, color=color)

        # Set axis labels
        xunit = source.data[x].unit if hasattr(source.data[x], 'unit') else None
        yunit = source.data[y].unit if hasattr(source.data[y], 'unit') else None
        fig.xaxis.axis_label = xlabel or '{}{}'.format(x, ' [{}]'.format(xunit) if xunit else '')
        fig.yaxis.axis_label = ylabel or '{}{}'.format(y, ' [{}]'.format(yunit) if yunit else '')

        # Formatting
        fig.legend.location = "top_right"

        # Draw sub figure
        sub = figure(width=500, height=500, title='Selected Source',
                     x_axis_label=str(self.wave_units), y_axis_label=str(self.flux_units),
                     x_axis_type='log', y_axis_type='log')
        sub.line('phot_wave', 'phot', source=phot_source, color='black', alpha=0.2)
        sub.circle('phot_wave', 'phot', source=phot_source, size=8, color='red', alpha=0.8)
        sub.line('spec_wave', 'spec', source=spec_source, color='red', alpha=0.5)

        # Make row layout
        layout = Row(children=[fig, sub])

        if draw:
            show(layout)

        return layout

    def plot(self, x, y, marker=None, color=None, scale=['linear','linear'],
             xlabel=None, ylabel=None, fig=None, order=None, identify=[],
             id_color='red', label_points=False, draw=True, **kwargs):
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
            _ = source.add([None] * len(source.data['name']), '{}_unc'.format(x))
        if '{}_unc'.format(y) not in source.column_names:
            _ = source.add([None] * len(source.data['name']), '{}_unc'.format(y))

        # Check if the x parameter is a color
        xname = x.replace('.', '_').replace('-', '_')
        if '-' in x and all([i in params for i in x.split('-')]):
            colordata = self.get_data(x)[0]
            if len(colordata) == 3:
                _ = source.add(at.Column(data=colordata[0], unit=colordata[2]), x)
                _ = source.add(at.Column(data=colordata[1], unit=colordata[2]), '{}_unc'.format(x))
                params.append(x)

        # Check if the y parameter is a color
        yname = y.replace('.', '_').replace('-', '_')
        if '-' in y and all([i in params for i in y.split('-')]):
            colordata = self.get_data(y)[0]
            if len(colordata) == 3:
                _ = source.add(at.Column(data=colordata[0], unit=colordata[2]), y)
                _ = source.add(at.Column(data=colordata[1], unit=colordata[2]), '{}_unc'.format(y))
                params.append(y)

        # Check the params are in the table
        if x not in params:
            raise ValueError("'{}' is not a valid x parameter. Please choose from {}".format(x, params))
        if y not in params:
            raise ValueError("'{}' is not a valid y parameter. Please choose from {}".format(y, params))

        # Make the figure
        if fig is None:

            # Tooltip names can't have '.' or '-'
            _ = source.add(at.Column(data=source.data[x]), xname)
            _ = source.add(at.Column(data=source.data[y]), yname)
            _ = source.add(at.Column(data=source.data['{}_unc'.format(x)]), '{}_unc'.format(xname))
            _ = source.add(at.Column(data=source.data['{}_unc'.format(y)]), '{}_unc'.format(yname))

            # Set up hover tool
            tips = [('Name', '@name'), ('Idx', '@idx'), (x, '@{0} (@{0}_unc)'.format(xname)), (y, '@{0} (@{0}_unc)'.format(yname))]
            hover = HoverTool(tooltips=tips, name='points')

            # Make the plot
            TOOLS = ['pan', 'reset', 'box_zoom', 'wheel_zoom', 'save', hover]
            title = '{} v {}'.format(x, y)
            fig = figure(width=800, height=500, title=title, y_axis_type=scale[1], x_axis_type=scale[0], tools=TOOLS)

        # Get marker class
        size = kwargs.get('size', 8)
        kwargs['size'] = size
        marker = getattr(fig, marker or self.marker)
        color = color or self.color

        # Prep data
        names = source.data['name']
        xval, xerr = source.data[x], source.data['{}_unc'.format(x)]
        yval, yerr = source.data[y], source.data['{}_unc'.format(y)]

        # Make error bars
        fig = u.errorbars(fig, xval, yval, xerr=xerr, yerr=yerr, color=color)

        # Plot nominal values
        marker(x, y, source=source, color=color, fill_alpha=0.7, name='points', **kwargs)

        # Identify sources
        idx = [ni for ni, name in enumerate(names) if name in identify]
        fig.circle(xval[idx], yval[idx], size=size + 5, color=id_color, fill_color=None, line_width=2)

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
                    coeffs, cov = np.polyfit(x=xd, y=yd, deg=order, w=1. / ye, cov=True)
                else:
                    coeffs, cov = np.polyfit(x=xd, y=yd, deg=order, cov=True)

                # Plot the line
                if coeffs is None or any([np.isnan(i) for i in coeffs]):
                    self.message("Could not fit that data with an order {} polynomial".format(order))
                else:

                    # Calculate values and 1-sigma
                    TT = np.vstack([xaxis**(order - i) for i in range(order + 1)]).T
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
        fig.xaxis.axis_label = xlabel or '{}{}'.format(x, ' [{}]'.format(xunit) if xunit else '')
        fig.yaxis.axis_label = ylabel or '{}{}'.format(y, ' [{}]'.format(yunit) if yunit else '')

        # Formatting
        fig.legend.location = "top_right"

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
        # Plot all SEDS
        if name_or_idx in ['all', '*']:
            name_or_idx = list(range(len(self.results)))

        # Make it into a list
        if isinstance(name_or_idx, (str, int)):
            name_or_idx = [name_or_idx]

        COLORS = u.color_gen(kwargs.get('palette', self.palette), n=len(name_or_idx))

        # Make the plot
        TOOLS = ['pan', 'reset', 'box_zoom', 'wheel_zoom', 'save']
        title = self.name
        fig = figure(width=800, height=500, title=title,
                     y_axis_type=scale[1], x_axis_type=scale[0],
                     x_axis_label='Wavelength [{}]'.format(self.wave_units),
                     y_axis_label='Flux Density [{}]'.format(str(self.flux_units)),
                     tools=TOOLS)

        # Plot each SED if it has been calculated
        for obj in name_or_idx:
            targ = self.get_SED(obj)
            if targ.calculated:
                c = next(COLORS)
                fig = targ.plot(fig=fig, color=c, one_color=True, output=True, normalize=normalize, label=targ.name, **kwargs)
            else:
                print("No SED to plot for source {}".format(obj))

        return fig

    def remove_SED(self, name_or_idx):
        """Remove an SED from the catalog

        Parameters
        ----------
        name_or_idx: str, int
            The name or index of the SED to remove
        """
        # Add the index
        self._results.add_index('name')

        # Get the rows
        if isinstance(name_or_idx, str) and name_or_idx in self._results['name']:
            self._results = self._results[self._results['name'] != name_or_idx]

        elif isinstance(name_or_idx, int) and name_or_idx <= len(self._results):
            self._results.remove_row([name_or_idx])

        else:
            self.message('Could not remove SED {}'.format(name_or_idx))

            return

    @property
    def results(self):
        """
        Return results table
        """
        # Get results table
        res_tab = self._results[[col for col in self._results.colnames if col not in self.array_cols]]

        # Mask empty elements
        for col in res_tab.columns.values():
            col.mask = [not bool(val) for val in col]

        return res_tab

    def save(self, file):
        """
        Save the serialized data

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

            # Get the pickle-safe data
            results = copy(self._results)
            results = results[[k for k in results.colnames if k != 'SED']]

            # Write the file
            f = open(file, 'wb')
            dill.dump(results, f)
            f.close()

            self.message('Catalog saved to {}'.format(file))

        else:

            self.message('{}: Path does not exist. Try again.'.format(path))

    @property
    def source(self):
        """Generates a ColumnDataSource from the results table"""
        results = copy(self.results)

        # Remove array columns
        results_dict = {key: val for key, val in dict(results).items()}

        # Add the index as a column in the table for tooltips
        results_dict['idx'] = np.arange(len(self.results))

        return ColumnDataSource(data=results_dict)


class MdwarfCatalog(Catalog):
    """A catalog of M dwarf stars"""
    def __init__(self, **kwargs):
        """Initialize the catalog object"""
        # Initialize object
        super().__init__(name='M Dwarf Catalog', **kwargs)

        # Read the names from the file
        file = str(importlib.resources.files('sedkit')/ 'data/sources.txt')
        self.from_file(file, run_methods=['find_SDSS', 'find_2MASS', 'find_WISE'])