#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
# !python3
"""
A module to generate a grid of model spectra
"""

from copy import copy
import os
import fileinput
import glob
import itertools
import pickle
from copy import copy
from functools import partial
from multiprocessing import Pool
from pkg_resources import resource_filename

import astropy.units as q
import astropy.io.votable as vo
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from svo_filters import svo

from . import utilities as u
from .spectrum import Spectrum


def interp_flux(flux, params, values):
    """
    Interpolate a cube of synthetic spectra

    Parameters
    ----------
    flux: np.ndarray
        The data array
    params: list
        A list of each free parameter range
    values: list
        A list of each free parameter values

    Returns
    -------
    tu
        The array of new flux values
    """
    # Iterate over each wavelength (-1 index of flux array)
    shp = flux.shape[-1]
    flx = np.zeros(shp)
    pn = len(params)

    for lam in range(shp):
        flx = flux[:, :, :, :, lam] if pn == 4 else flux[:, :, :, lam] if pn == 3 else flux[:, :, lam] if pn == 2 else flux[:, lam]
        interp_f = RegularGridInterpolator(params, flx)
        f, = interp_f(values)
        flx[lam] = f

    return flx


def load_model(file, parameters=None, wl_min=5000, wl_max=50000, max_points=10000):
    """Load a model from file

    Parameters
    ----------
    file: str
        The path to the file
    parameters: sequence
        The parameters to extract
    wl_min: float
        The minimum wavelength
    wl_max: float
        The maximum wavelength
    max_points: int
        If too high-res, rebin to this number of points

    Returns
    -------
    dict
        A dictionary of values
    """
    # Parse the XML file
    vot = vo.parse_single_table(file)

    # Parse the SVO filter metadata
    all_params = [str(p).split() for p in vot.params]

    meta = {}
    for p in all_params:

        # Extract the key/value pairs
        key = p[1].split('"')[1]
        val = p[-1].split('"')[1]

        if (parameters and key in parameters) or not parameters:

            # Do some formatting
            if p[2].split('"')[1] == 'float' or p[3].split('"')[1] == 'float':
                val = float(val)

            else:
                val = val.replace('b&apos;','').replace('&apos','').replace('&amp;','&').strip(';')

            # Add it to the dictionary
            meta[key] = val

    # Add the filename
    meta['filepath'] = file

    # Trim and add the data
    spectrum = np.array([list(i) for i in vot.array]).T
    spec_data = spectrum[:, (spectrum[0] >= wl_min) & (spectrum[0] <= wl_max)]

    # Rebin if too high resolution
    if len(spec_data[0]) > max_points:
        new_w = np.linspace(spec_data[0].min(), spec_data[0].max(), max_points)
        spec_data = u.spectres(new_w, *spec_data)

    # Store the data
    meta['spectrum'] = spec_data
    meta['label'] = '/'.join([str(v) for k, v in meta.items() if k not in
                              ['spectrum', 'filepath']])

    print(file, ': Done!')

    return meta


def load_ModelGrid(path):
    """Load a model grid from a file

    Parameters
    ----------
    path: str
        The path to the saved ModelGrid

    Returns
    -------
    sedkit.modelgrid.ModelGrid
        The loaded ModelGrid object
    """
    if not os.path.isfile(path):
        raise IOError("File not found:", path)

    data = pickle.load(open(path, 'rb'))

    mg = ModelGrid(data['name'], data['parameters'])
    for key, val in data.items():
        setattr(mg, key, val)

    return mg


class ModelGrid:
    """A class to store a model grid"""
    def __init__(self, name, parameters, wave_units=None, flux_units=None,
                 resolution=None, trim=None, verbose=True, ref=None, **kwargs):
        """Initialize the model grid from a directory of VO table files

        Parameters
        ----------
        name: str
            The name of the model grid
        parameters: sequence
            The list of parameters (column names) to include in the grid
        wave_units: astropy.units.quantity.Quantity
            The wavelength units
        flux_units: astropy.units.quantity.Quantity
            The flux units
        resolution: float
            The resolution of the models
        trim: sequence
            Trim the models to a particular wavelength range
        ref: str
            A reference for the modelgrid
        verbose: bool
            Print info
        """
        # Store the path and name
        self.path = None
        self.name = name
        self.parameters = parameters
        self.wave_units = wave_units
        self.native_wave_units = wave_units
        self.flux_units = flux_units
        self.native_flux_units = flux_units
        self.resolution = resolution
        self.trim = trim
        self.ref = ref
        self.verbose = verbose
        self.phot = False

        # Make all args into attributes
        for key, val in kwargs.items():
            setattr(self, key, val)

        # Make the empty table
        columns = self.parameters+['filepath', 'spectrum', 'label']
        self.index = pd.DataFrame(columns=columns)

    def add_model(self, spectrum, label=None, filepath=None, **kwargs):
        """Add the given model with the specified parameter values as kwargs

        Parameters
        ----------
        spectrum: sequence
            The model spectrum
        """
        # Check that all the necessary params are included
        if not all([i in kwargs for i in self.parameters]):
            raise ValueError("Must have kwargs for", self.parameters)

        # Make the dictionary of new data
        kwargs.update({'spectrum': spectrum, 'filepath': filepath, 'label': label})
        new_rec = pd.DataFrame({k: [v] for k, v in kwargs.items()})

        # Add it to the index
        self.index = self.index.append(new_rec)

    @staticmethod
    def closest_value(input_value, possible_values, n_vals=1):
        """
        This function calculates, given an input_value and an array of possible_values,
        the closest value to input_value in the array.

        Parameters
        ----------
        input_value: double
            Input value to compare against possible_values
        possible_values: np.ndarray
            Array of possible values to compare against input_value
        n_vals: int
            The number of closest values to return

        Returns
        -------
        double
            Closest value(s) on possible_values to input_value
        """
        # Calculate the difference
        difference = np.abs(possible_values - input_value)

        # Sort by difference
        idx = np.argsort(difference)
        sorted_diffs = possible_values[idx]

        # Get correct number of vals
        vals = sorted_diffs[:n_vals]

        return vals[0] if n_vals == 1 else vals

    def filter(self, **kwargs):
        """Retrieve all models with the specified parameters

        Returns
        -------
        list
            A list of the spectra as sedkit.spectrum.Spectrum objects
        """
        # Get the relevant table rows
        return u.filter_table(self.index, **kwargs)

    def get_spectrum(self, closest=False, snr=None, interp=True, spec_obj=True, trim=None, resolution=None, **kwargs):
        """Retrieve the first model with the specified parameters

        Parameters
        ----------
        closest: bool
            Rounds to closest effective temperature
        snr: int (optional)
            The SNR to generate for the spectrum
        interp: bool
            Interpolate the model grid if not present
        spec_obj: bool
            Return a sedkit.spectrum.Spectrum object

        Returns
        -------
        sedkit.spectrum.Spectrum or np.ndarray
            A numpy array of the spectrum
        """
        # Check if it's in range
        self._in_range(**kwargs)

        # Can't interp if strings
        if any([type(self.index[param][0]) == str for param in kwargs]):
            interp = False

        # Get the row index and filepath
        rows = copy(self.index)
        for arg, val in kwargs.items():

            if closest:
                old_val = copy(val)
                val = self.closest_value(old_val, rows[arg])
                print('{} = {} rounded to {}'.format(arg, old_val, val))

            rows = rows.loc[rows[arg] == val]

        if rows.empty:
            if interp:
                self.message("Interpolating model grid to point {}".format(kwargs))

                # Guess missing parameter values
                for param in self.parameters:
                    if param not in kwargs:
                        vals = getattr(self, '{}_vals'.format(param))
                        kwargs[param] = vals[len(vals) // 2]

                spec, name = self.interp(**kwargs)

            else:
                print("No models found satisfying", kwargs)
                return None

        else:
            spec = copy(rows.iloc[0].spectrum)
            spec[0] = (spec[0] * self.native_wave_units).to(self.wave_units).value
            spec[1] = (spec[1] * self.native_flux_units).to(self.flux_units).value
            name = rows.iloc[0].label

        # Scrub nans and infs
        spec = u.scrub(spec)

        # Convert to specobj
        spec = Spectrum(spec[0] * self.wave_units, spec[1] * self.flux_units, name=name, snr=snr, ref=self.ref, phot=self.phot, **kwargs)

        if trim is not None:
            spec = spec.trim(include=trim, concat=True)

        if resolution is not None:
            spec = spec.resamp(resolution=resolution)

        return spec if spec_obj else spec.data

    def index_models(self, parameters=None, wl_min=0.3*q.um, wl_max=25*q.um):
        """Generate model index file for faster reading

        Parameters
        ----------
        parameters: sequence
            The names of the parameters from the VOT files to index
        """
        # Get the files
        files = glob.glob(os.path.join(self.path, '*.xml'))
        self.n_models = len(files)
        print("Indexing {} models for {} grid...".format(self.n_models, self.name))

        # Grab the parameters and the filepath for each
        pool = Pool(8)
        func = partial(load_model, parameters=parameters, wl_min=wl_min.to(self.wave_units).value, wl_max=wl_max.to(self.wave_units).value)
        all_meta = pool.map(func, files)
        pool.close()
        pool.join()

        # Make the index table
        self.index = pd.DataFrame(all_meta)
        self.index.to_pickle(self.index_path)

        # Update attributes
        if parameters is None:
            parameters = [col for col in self.index.columns if col not in ['filepath', 'spectrum', 'label']]
        self.parameters = parameters

    def _in_range(self, **kwargs):
        """
        Check that the given parameter values are in range of the parameter space

        Returns
        -------
        bool
            If the values are in range
        """
        for param, val in kwargs.items():

            # Pass range test if column is strings
            if type(self.index[param][0]) != str:

                vals = getattr(self, '{}_vals'.format(param))
                mn, mx = vals.min(), vals.max()
                if val < mn or val > mx:
                    raise ValueError("{} outside of '{}' parameter space [{}, {}]".format(val, param, mn, mx))

    def interp(self, **kwargs):
        """
        Interpolate the grid to the desired parameters

        Returns
        -------
        dict
            A dictionary of arrays of the wavelength, flux, and
            mu values and the effective radius for the given model
        """
        # Make sure all parameters are included
        if not all([param in kwargs for param in self.parameters]):
            raise ValueError("{}: Please specify values for all parameters {}".format(kwargs, self.parameters))

        # Select subset of parameter space to speed calculation
        param_vals = []
        param_lims = []
        param_dims = []
        for param in self.parameters:
            possible_values = getattr(self, '{}_vals'.format(param))
            pval = kwargs[param]
            param_vals.append(pval)

            # On grid
            if pval in possible_values:
                pmin = pmax = pval
                dim = 1

            # Off grid
            else:

                try:
                    pmin, pmax = sorted(self.closest_value(pval, possible_values, n_vals=2))
                    dim = 2
                except:
                    raise ValueError("{} = {}: Please use parameter value in range {} - {}".format(param, pval, min(possible_values), max(possible_values)))

            param_lims.append((pmin, pmax))
            param_dims.append(dim)

        # Get length of wave array
        start_vals = {param: getattr(self, '{}_vals'.format(param))[0] for param in self.parameters}
        wavelength = self.get_spectrum(**start_vals, interp=False, spec_obj=False)[0]

        # Get the flux array by iterating through rows
        flux_array = np.empty(tuple(param_dims + [len(wavelength)]))
        for n0, d0 in enumerate(param_lims[0]):
            for n1, d1 in enumerate(param_lims[1]):
                for n2, d2 in enumerate(param_lims[2]):
                    for n3, d3 in enumerate(param_lims[3]):

                        model_vals = {self.parameters[0]: d0, self.parameters[1]: d1, self.parameters[2]: d2, self.parameters[3]: d3}

                        # Retrieve spectrum using the `get_spectrum()` method
                        spec = self.get_spectrum(**model_vals, interp=False, spec_obj=False)[1]
                        flux_array[n0 - 1, n1 - 1, n2 - 1, n3 - 1] = spec
                        del spec

        # Ignore dimensions that don't need interpolation
        flux_array = flux_array.squeeze()
        pidx = [pl > 1 for pl in param_dims]

        # Interpolate each wavelength point over the grid
        new_flux = np.empty_like(wavelength)
        pn = flux_array.ndim - 1
        for lam in range(len(wavelength)):
            flx = flux_array[:, :, :, :, lam] if pn == 4 else flux_array[:, :, :, lam] if pn == 3 else flux_array[:, :, lam] if pn == 2 else flux_array[:, lam]
            interp_f = RegularGridInterpolator(np.array(param_lims)[pidx], flx)
            new_flux[lam] = interp_f(np.array(param_vals)[pidx])[0]

        name = '/'.join([str(val) for key, val in kwargs.items()])

        return [wavelength, new_flux], name

    def load(self, dirname, **kwargs):
        """Load a model grid from a directory of VO table XML files

        Parameters
        ----------
        dirname: str
            The name of the directory
        """
        # Make the path
        if not os.path.exists(dirname):
            raise IOError(dirname, ": No such directory")

        # See if there is a table of parameters
        self.path = dirname
        self.index_path = os.path.join(dirname, 'index.p')
        if not os.path.isfile(self.index_path):
            os.system("touch {}".format(self.index_path))

            # Index the models
            self.index_models(parameters=self.parameters, **kwargs)

        # Load the index
        self.index = pd.read_pickle(self.index_path)

        # Store the parameter ranges
        for param in self.parameters:
            setattr(self, '{}_vals'.format(param), np.asarray(np.unique(self.index[param])))

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

    def photometry(self, bandpasses, weight=True):
        """
        Generate a new ModelGrid object of photometry in the given bands

        Parameters
        ----------
        bandpasses: list
            A list of the bandpasses to measure

        Returns
        -------
        ModelGrid
            The photometry ModelGrid
        """
        # Copy the ModelGrid and empty the index
        phot = ModelGrid(name=self.name, parameters=self.parameters)
        dic = copy(self.__dict__)
        dic['index'] = dic['index'][:0]
        phot.__dict__ = dic

        # Iterate over the rows
        for n, row in self.index.iterrows():

            # Get the spectrum
            params = {param: val for param, val in dict(row).items() if param in phot.parameters}
            spec = self.get_spectrum(**params)
            err = True if len(spec.spectrum) == 3 else False

            # Collect data
            data = []
            weights = []
            for band in bandpasses:

                # Get the bandpass
                bp = svo.Filter(band)

                # Weight the bands by the inverse of the band width
                weights.append((bp.wave_max - bp.wave_min).to(q.um).value if weight else 1)

                # Calculate the synthetic flux
                flx, flx_unc = spec.synthetic_flux(bp)
                if flx is not None and not np.isnan(flx):
                    if err:
                        data.append([bp.wave_eff.to(self.wave_units).astype(np.float16).value, flx.to(self.flux_units).value, flx_unc.to(self.flux_units).value])
                    else:
                        data.append([bp.wave_eff.to(self.wave_units).astype(np.float16).value, flx.to(self.flux_units).value])

            # Unpack and save as new spectrum
            dataT = np.array(data).T
            phot.add_model(dataT if err else dataT[:2], weights=weights, label=row['label'], **params)
            phot.phot = True

        return phot

    def plot(self, fig=None, scale='log', draw=True, **kwargs):
        """
        Plot the models using Spectrum.plot() with the given parameters

        Parameters
        ----------
        fig: bokeh.figure (optional)
            The figure to plot on
        scale: str
            The scale of the x and y axes, ['linear', 'log']
        draw: bool
            Draw the plot rather than just return it

        Returns
        -------
        bokeh.figure
            The figure
        """
        # Get the model
        model = self.get_spectrum(**kwargs)

        # Plot or return it
        return model.plot(fig=fig, scale=scale, draw=draw, **kwargs)

    def resample_grid(self, name=None, **kwargs):
        """
        Resample the model grid to new parameter values

        Parameters
        ----------
        kwargs
            A dictionary of the parameter name and list of values to resample the grid

        Returns
        -------
        sedkit.modelgrid.ModelGrid
        """
        # Make an empty index
        new_index = copy(self.index)[:0]

        # If a param is missing, use the original range
        for param in self.parameters:
            if param not in kwargs:
                vals = getattr(self, '{}_vals'.format(param))
                kwargs[param] = vals
                print("'{}' parameter missing. Assuming vals {}".format(param, vals))

        # Check that the new grid points are valid
        val_list = []
        par_list = []
        for param, vals in kwargs.items():

            # Check for valid param name
            if param not in self.parameters:
                raise ValueError("{}: not a model grid parameter. Valid parameters include {}".format(param, self.parameters))

            # Check for valid param range
            mn = getattr(self, '{}_vals'.format(param)).min()
            mx = getattr(self, '{}_vals'.format(param)).max()
            if any([np.logical_or(i < mn, i > mx) for i in vals]):
                raise ValueError("{} values not within existing parameter range {} - {}".format(param, mn, mx))

            # Add approved values to list
            val_list.append(vals)
            par_list.append(param)

        # Iterate over the new parameter space and add the models
        grid_params = list(itertools.product(*val_list))
        for vals in grid_params:

            try:

                # Populate the row
                pars = {par: val for par, val in zip(par_list, vals)}
                spec = self.get_spectrum(**pars, interp=True)
                pars['label'] = spec.name
                pars['spectrum'] = spec.data
                pars['filepath'] = 'interp'

                # Add line to the new dataframe
                new_index = new_index.append(pars, ignore_index=True)

            except ValueError:
                print("Could not interpolate grid point {}".format(pars))

        # Make the new ModelGrid
        new_grid = copy(self)
        new_grid.index = new_index
        new_grid.name = name or self.name
        new_grid.native_wave_units = self.wave_units
        new_grid.native_flux_units = self.flux_units

        # Store the new parameter values
        for param, vals in kwargs.items():
            setattr(new_grid, '{}_vals'.format(param), np.asarray(vals))

        return new_grid

    def save(self, file):
        """Save the model grid to file

        Parameters
        ----------
        file: str
            The path for the new file
        """
        path = os.path.dirname(file)

        if os.path.exists(path):

            # Make the file if necessary
            if not os.path.isfile(file):
                os.system('touch {}'.format(file))

            # Write the file
            f = open(file, 'wb')
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
            f.close()

            print("ModelGrid '{}' saved to {}".format(self.name, file))

    @property
    def wave_limits(self):
        """The wavelength limits of the models"""
        mn = np.min([np.min(spec[0]) for spec in self.index.spectrum])
        mx = np.max([np.max(spec[0]) for spec in self.index.spectrum])

        return (mn * self.native_wave_units).to(self.wave_units), (mx * self.native_wave_units).to(self.wave_units)


class BTSettl(ModelGrid):
    """Child class for the BT-Settl model grid"""
    def __init__(self, root=None, **kwargs):
        """Loat the model object"""
        # List the parameters
        params = ['alpha', 'logg', 'teff', 'meta']

        # Inherit from base class
        super().__init__('BT-Settl', params, q.AA, q.erg/q.s/q.cm**2/q.AA, ref='2014IAUS..299..271A', **kwargs)

        # Load the model grid
        modeldir = 'data/models/atmospheric/btsettl'
        root = root or resource_filename('sedkit', modeldir)
        self.load(root)


class Filippazzo2016(ModelGrid):
    """Child class for the Filippazzo et al. (2016) sample"""
    def __init__(self):
        """Load the model object"""
        model_path = 'data/models/atmospheric/Filippazzo2016.p'
        root = resource_filename('sedkit', model_path)

        data = pickle.load(open(root, 'rb'))

        # Inherit from base class
        super().__init__(data['name'], data['parameters'], ref='2017yCat..18100158F')

        # Copy to new __dict__
        for key, val in data.items():
            setattr(self, key, val)


class SpexPrismLibrary(ModelGrid):
    """Child class for the SpeX Prism Library model grid"""
    def __init__(self):
        """Loat the model object"""
        # List the parameters
        params = ['spty']

        # Inherit from base class
        super().__init__('SpeX Prism Library', params, q.AA, q.erg/q.s/q.cm**2/q.AA, ref='2014ASInC..11....7B')

        # Load the model grid
        model_path = 'data/models/atmospheric/spexprismlibrary'
        root = resource_filename('sedkit', model_path)
        self.load(root)

        # Add numeric spectral type
        self.index['SpT'] = [u.specType(i.split(',')[0].replace('Opt:','')\
                              .replace('NIR:',''))[0] for i in\
                              self.index['spty']]


def format_XML(modeldir):
    """Convert VO tables with '<RESOURCE type="datafile">' into 
    <RESOURCE type="results"> so astropy.io.votable can read it

    Parameters
    ----------
    modeldir: str
        The path to the mdel directory of files to convert
    """
    files = glob.glob(modeldir+'*.xml') 
    for line in fileinput.input(files, inplace=True):
        print(line.replace('datafile', 'results'), end='')
