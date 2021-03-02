#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
# !python3
"""
Make nice spectrum objects to pass around SED class
"""
import copy
from functools import wraps, partial
from itertools import groupby
from operator import itemgetter
from multiprocessing import Pool
import os
from pkg_resources import resource_filename

import astropy.constants as ac
import astropy.units as q
import astropy.io.votable as vo
from astropy.io import fits
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool
import numpy as np
from pandas import DataFrame
from svo_filters import Filter

from . import mcmc as mc
from . import utilities as u


def copy_raw(func):
    """A wrapper to copy the raw data to the new Spectrum object"""
    @wraps(func)
    def _copy_raw(*args, **kwargs):
        """Run the function then update the raw data attribute"""
        # Grab the original data
        raw = args[0].raw or args[0].spectrum

        # Run the function and update the raw attribute
        new_spec = func(*args, **kwargs)
        new_spec.raw = raw

        return new_spec

    return _copy_raw


class Spectrum:
    """
    A class to store, calibrate, fit, and plot a single spectrum
    """
    def __init__(self, wave, flux, unc=None, snr=None, name=None,
                 ref=None, header=None, verbose=False, **kwargs):
        """Initialize the Spectrum object

        Parameters
        ----------
        wave: astropy.units.quantity.Quantity
            The wavelength array
        flux: astropy.units.quantity.Quantity
            The flux density array
        unc: np.ndarray
            The flux density uncertainty array
        snr: float (optional)
            A value to override spectrum SNR
        snr_trim: float (optional)
            The SNR value to trim spectra edges up to
        name: str
            A name for the spectrum
        ref: str
            A reference for the data
        header: str
            The header for the spectrum file
        verbose: bool
            Print helpful stuff
        """
        # Meta
        self.verbose = verbose
        self.name = name or 'New Spectrum'
        self.ref = ref
        self.header = header
        self.phot = False

        # Make sure the arrays are the same shape
        if not wave.shape == flux.shape and ((unc is None) or not (unc.shape == flux.shape)):
            raise TypeError("Wavelength, flux and uncertainty arrays must be the same shape.")

        # Check wave units are length
        if not u.equivalent(wave, q.um):
            raise TypeError("Wavelength array must be in astropy.units.quantity.Quantity length units, e.g. 'um'")

        # Check flux units are flux density
        if not u.equivalent(flux, q.erg / q.s / q.cm**2 / q.AA):
            raise TypeError("Flux array must be in astropy.units.quantity.Quantity flux density units, e.g. 'erg/s/cm2/A'")

        # Generate uncertainty array
        if unc is None and isinstance(snr, (int, float)):
            unc = flux / snr

        # Make sure the uncertainty array is in the correct units
        if unc is not None:
            if not u.equivalent(unc, q.erg / q.s / q.cm**2 / q.AA):
                raise TypeError("Uncertainty array must be in astropy.units.quantity.Quantity flux density units, e.g. 'erg/s/cm2/A'")

        # Replace negatives, zeros, and infs with nans
        spectrum = [wave, flux]
        spectrum += [unc] if unc is not None else []
        spectrum = u.scrub(spectrum, fill_value=np.nan)

        # Store history info and raw data
        self.history = {}
        self.raw = None

        # Strip and store units
        self._wave_units = wave.unit
        self._flux_units = flux.unit
        self.units = [self._wave_units, self._flux_units]
        self.units += [self._flux_units] if unc is not None else []
        spectrum = [i.value for i in spectrum]

        # Add the data
        self.wave = spectrum[0]
        self.flux = spectrum[1]
        self.unc = None if unc is None else spectrum[2]

        # Store components if added
        self.components = None
        self.best_fit = {}

        # Store kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    @copy_raw
    def __add__(self, spec):
        """Add the spectra of this and another Spectrum object

        Parameters
        ----------
        spec: sedkit.spectrum.Spectrum
            The spectrum object to add

        Returns
        -------
        sedkit.spectrum.Spectrum
            A new spectrum object with the input spectra stitched together
        """
        # If None is added, just return a copy
        if spec is None:
            return Spectrum(*self.spectrum, name=self.name)

        if not isinstance(type(spec), type(Spectrum)):
            raise TypeError('spec must be sedkit.spectrum.Spectrum')

        # Make spec the same units
        spec.wave_units = self.wave_units
        spec.flux_units = self.flux_units

        # Get the two spectra to stitch
        s1 = self.data
        s2 = spec.data

        # Determine if overlapping
        overlap = True
        try:
            if s1[0][-1] > s1[0][0] > s2[0][-1] or s2[0][-1] > s2[0][0] > s1[0][-1]:
                overlap = False
        except IndexError:
            overlap = False

        # Concatenate and order two segments if no overlap
        if not overlap:

            # Drop uncertainties on both spectra if one is missing
            if self.unc is None or spec.unc is None:
                s1 = [self.wave, self.flux]
                s2 = [spec.wave, spec.flux]

            # Concatenate arrays and sort by wavelength
            new_spec = np.concatenate([s1, s2], axis=1).T
            new_spec = new_spec[np.argsort(new_spec[:, 0])].T

        # Otherwise there are three segments, (left, overlap, right)
        else:

            # Get the left segemnt
            left = s1[:, s1[0] <= s2[0][0]]
            if not np.any(left):
                left = s2[:, s2[0] <= s1[0][0]]

            # Get the right segment
            right = s1[:, s1[0] >= s2[0][-1]]
            if not np.any(right):
                right = s2[:, s2[0] >= s1[0][-1]]

            # Get the overlapping segements
            o1 = s1[:, np.where((s1[0] < right[0][0]) & (s1[0] > left[0][-1]))].squeeze()
            o2 = s2[:, np.where((s2[0] < right[0][0]) & (s2[0] > left[0][-1]))].squeeze()

            # Get the resolutions
            r1 = s1.shape[1] / (max(s1[0]) - min(s1[0]))
            r2 = s2.shape[1] / (max(s2[0]) - min(s2[0]))

            # Make higher resolution s1
            if r1 < r2:
                o1, o2 = o2, o1

            # Interpolate s2 to s1
            o2_flux = np.interp(o1[0], s2[0], s2[1])

            # Get the average
            o_flux = np.nanmean([o1[1], o2_flux], axis=0)

            # Calculate uncertainties if possible
            if len(s2) == len(o1) == 3:
                o2_unc = np.interp(o1[0], s2[0], s2[2])
                o_unc = np.sqrt(o1[2]**2 + o2_unc**2)
                overlap = np.array([o1[0], o_flux, o_unc])
            else:
                overlap = np.array([o1[0], o_flux])
                left = left[:2]
                right = right[:2]

            # Make sure it is 2D
            if overlap.shape == (3,):
                overlap.shape = 3, 1
            if overlap.shape == (2,):
                overlap.shape = 2, 1

            # Concatenate the segments
            new_spec = np.concatenate([left, overlap, right], axis=1)

        # Add units
        new_spec = [i * Q for i, Q in zip(new_spec, self.units)]

        # Make the new spectrum object
        new_spec = Spectrum(*new_spec)

        # Store the components
        new_spec.components = Spectrum(*self.spectrum), spec

        return new_spec

    def best_fit_model(self, modelgrid, report=None, name=None, **kwargs):
        """Perform simple fitting of the spectrum to all models in the given
        modelgrid and store the best fit

        Parameters
        ----------
        modelgrid: sedkit.modelgrid.ModelGrid
            The model grid to fit
        report: str
            The name of the parameter to plot versus the
            Goodness-of-fit statistic
        name: str
            A name for the fit
        """
        # Prepare data
        name = name or '{} fit'.format(modelgrid.name)
        spectrum = Spectrum(*self.spectrum)
        rows = [row for n, row in modelgrid.index.iterrows()]

        # Iterate over entire model grid
        pool = Pool(8)
        func = partial(fit_model, fitspec=spectrum, wave_units=modelgrid.wave_units)
        fit_rows = pool.map(func, rows)
        pool.close()
        pool.join()

        # Turn the results into a DataFrame and sort
        models = DataFrame(fit_rows)
        models = models.sort_values('gstat')

        # Get the best fit
        bf = copy.copy(models.iloc[0])

        # Make into a dictionary
        bdict = dict(bf)

        # Add full model
        bdict['full_model'] = modelgrid.get_spectrum(**{par: val for par, val in bdict.items() if par in modelgrid.parameters}, snr=5)
        bdict['wave_units'] = self.wave_units
        bdict['flux_units'] = self.flux_units

        self.message(bf[modelgrid.parameters])

        if report is not None:

            # Configure plot
            tools = "pan, wheel_zoom, box_zoom, reset"
            rep = figure(tools=tools, x_axis_label=report, y_axis_label='Goodness-of-fit', plot_width=600, plot_height=400)

            # Single out best fit
            best = ColumnDataSource(data=models.iloc[:1])
            others = ColumnDataSource(data=models.iloc[1:])

            # Add hover tool
            hover = HoverTool(tooltips=[('label', '@label'), ('gstat', '@gstat')])
            rep.add_tools(hover)

            # Plot the fits
            rep.circle(report, 'gstat', source=best, color='red', legend_label=bf['label'])
            rep.circle(report, 'gstat', source=others)

            # Show the plot
            show(rep)

        self.best_fit[name] = bdict

    def convolve_filter(self, filter, **kwargs):
        """
        Convolve the spectrum with a filter

        Parameters
        ----------
        filter: svo_filters.svo.Filter, str
            The filter object or name

        Returns
        -------
        sedkit.spectrum.Spectrum
            The convolved spectrum object
        """
        # Ensure filter object
        if isinstance(filter, str):
            filter = Filter(filter)

        # Get the wavelengths and throughput
        flx, unc = filter.apply(self.spectrum)

        # Make the new spectrum object
        new_spec = Spectrum(filter.wave[0], flx, unc=unc)

        return new_spec

    @property
    def data(self):
        """
        Store the spectrum without units
        """
        if self.unc is None:
            data = np.stack([self.wave, self.flux])
        else:
            data = np.stack([self.wave, self.flux, self.unc])

        return data

    def export(self, filepath, header=None):
        """
        Export the spectrum to file

        Parameters
        ----------
        filepath: str
            The path for the exported file
        """
        # Get target directory
        dirname = os.path.dirname(filepath) or '.'
        name = self.name.replace(' ', '_')

        # Check the parent directory
        if not os.path.exists(dirname):
            raise IOError('{}: No such directory'.format(dirname))

        # Write the file with a header
        head = '{}\nWavelength [{}], Flux Density [{}]'.format(name, self.wave_units, self.flux_units)
        if isinstance(header, str):
            head += '\n{}'.format(header)
        t_data = np.asarray(self.spectrum).T
        np.savetxt(filepath, t_data, header=head)

    def fit(self, spec, weights=None, wave_units=None, scale=True, plot=False):
        """Determine the goodness of fit between this and another spectrum

        Parameters
        ----------
        spec: sedkit.spectrum.Spectrum, np.ndarray
            The spectrum object or [W, F] array to fit
        wave_units: astropy.units.quantity.Quantity
            The wavelength units of the input spectrum if
            it is a numpy array
        scale: bool
            Scale spec when measuring the goodness of fit

        Returns
        -------
        tuple
            The fit statistic, and normalization for the fit
        """
        # In case the wavelength units are different
        xnorm = 1
        wav = self.wave

        if hasattr(spec, 'spectrum'):

            # Resample spec onto self wavelength
            spec2 = spec.resamp(self.spectrum[0])
            flx2 = spec2.flux
            err2 = np.ones_like(spec2.flux) if spec2.unc is None else spec2.unc

        elif isinstance(spec, (list, tuple, np.ndarray)):

            spec2 = copy.copy(spec)

            # Convert A to um
            if wave_units is not None:
                xnorm = q.Unit(wave_units).to(self.wave_units)
                spec2[0] = spec2[0] * xnorm

            # Resample spec onto self wavelength
            spec2 = u.spectres(self.wave, *spec2)
            wav = spec2[0]
            flx2 = spec2[1]
            err2 = np.ones_like(flx2) if len(spec2) == 2 else spec2[2]

        else:
            raise TypeError("Only an sedkit.spectrum.Spectrum or numpy.ndarray can be fit.")

        # Get the self data
        flx1 = self.flux
        err1 = np.ones_like(flx1) if self.unc is None else self.unc

        # Make default weights the bin widths, excluding gaps in spectra
        if weights is None:
            weights = np.gradient(wav)
            weights[weights > np.std(weights)] = 1

        # Run the fitting and get the normalization
        gstat, ynorm = u.goodness(flx1, flx2, err1, err2, weights)

        # Run it again with the scaling removed
        if scale:
            gstat, _ = u.goodness(flx1, flx2 * ynorm, err1, err2 * ynorm, weights)

        if plot:
            fig = self.plot(best_fit=False)
            fig.line(spec.wave, spec.flux * ynorm, legend_label='Fit')
            show(fig)

        return gstat, ynorm, xnorm

    @copy_raw
    def flux_calibrate(self, distance, target_distance=10 * q.pc, flux_units=None):
        """Flux calibrate the spectrum from the given distance to the target distance

        Parameters
        ----------
        distance: astropy.unit.quantity.Quantity, sequence
            The current distance or (distance, uncertainty) of the spectrum
        target_distance: astropy.unit.quantity.Quantity
            The distance to flux calibrate the spectrum to
        flux_units: astropy.unit.quantity.Quantity
            The desired flux units of the output

        Returns
        -------
        sedkit.spectrum.Spectrum
            The flux calibrated spectrum object
        """
        # Set target flux units
        if flux_units is None:
            flux_units = self.flux_units

        # Calculate the scaled flux
        flux = (self.spectrum[1] * (distance[0] / target_distance)**2).to(flux_units)

        # Calculate the scaled uncertainty
        if self.unc is None:
            unc = None
        else:
            term1 = (self.spectrum[2] * distance[0] / target_distance).to(flux_units)
            term2 = (2 * self.spectrum[1] * (distance[1] * distance[0] / target_distance**2)).to(flux_units)
            unc = np.sqrt(term1**2 + term2**2)

        return Spectrum(self.spectrum[0], flux, unc, name=self.name)

    @property
    def flux_units(self):
        """A property for flux_units"""
        return self._flux_units

    @flux_units.setter
    def flux_units(self, flux_units):
        """A setter for flux_units

        Parameters
        ----------
        flux_units: astropy.units.quantity.Quantity
            The astropy units of the SED wavelength
        """
        # Check the units
        if not u.equivalent(flux_units, q.erg / q.s / q.cm**2 / q.AA):
            raise TypeError("flux_units must be in flux density units, e.g. 'erg/s/cm2/A'")

        # Update the flux and unc arrays
        self.flux = self.flux * self.flux_units.to(flux_units)
        if self.unc is not None:
            self.unc = self.unc * self.flux_units.to(flux_units)

        # Set the flux_units
        self._flux_units = flux_units
        self._set_units()

    def integrate(self, units=q.erg / q.s / q.cm**2):
        """Calculate the area under the spectrum

        Parameters
        ----------
        units: astropy.units.quantity.Quantity
            The target units for the integral

        Returns
        -------
        sequence
            The integrated flux and uncertainty
        """
        # Make sure the target units are flux units
        if not u.equivalent(units, q.erg / q.s / q.cm**2):
            raise TypeError("units must be in flux units, e.g. 'erg/s/cm2'")

        # Calculate the factor for the given units
        m = self.flux_units * self.wave_units

        # Scrub the spectrum
        spec = u.scrub(self.data)
        val = (np.trapz(spec[1], x=spec[0]) * m).to(units)

        if self.unc is None:
            unc = None
        else:
            unc = np.sqrt(np.nansum((spec[2] * np.gradient(spec[0]) * m)**2)).to(units)

        return val, unc

    @copy_raw
    def interpolate(self, wave):
        """Interpolate the spectrum to another wavelength array

        Parameters
        ----------
        wave: astropy.units.quantity.Quantity, sedkit.spectrum.Spectrum
            The wavelength array to interpolate to

        Returns
        -------
        sedkit.spectrum.Spectrum
            The interpolated spectrum object
        """
        # Pull out wave if its a Spectrum object
        if hasattr(wave, 'spectrum'):
            wave = wave.spectrum[0]

        # Test units
        if not u.equivalent(wave, q.um):
            raise ValueError("New wavelength array must be in units of length.")

        # Get the data and make into same wavelength units
        w0 = self.wave * self.wave_units.to(wave.unit)
        f0 = self.spectrum[1]
        if len(self.spectrum) > 2:
            e0 = self.spectrum[2]
        else:
            e0 = np.zeros_like(f0)

        # Interpolate self to new wavelengths
        f1 = np.interp(wave.value, w0, f0.value, left=np.nan, right=np.nan) * self.flux_units
        e1 = np.interp(wave.value, w0, e0.value, left=np.nan, right=np.nan) * self.flux_units

        return Spectrum(wave, f1, e1, name=self.name)

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

    def mcmc_fit(self, model_grid, params=['teff'], walkers=5, steps=20, name=None, report=None):
        """
        Produces a marginalized distribution plot of best fit parameters from the specified model_grid

        Parameters
        ----------
        model_grid: sedkit.modelgrid.ModelGrid
            The model grid to use
        params: list
            The list of model grid parameters to fit
        walkers: int
            The number of walkers to deploy
        steps: int
            The number of steps for each walker to take
        name: str
            Name for the fit
        plot: bool
            Make plots
        """
        # Specify the parameter space to be walked
        for param in params:
            if param not in model_grid.parameters:
                raise ValueError("'{}' not a parameter in this model grid, {}".format(param, model_grid.parameters))

        # A name for the fit
        name = name or model_grid.name

        # Ensure modelgrid and spectruym are the same wave_units
        model_grid.wave_units = self.wave_units

        # Set up the sampler object
        self.sampler = mc.SpecSampler(self, model_grid, params)

        # Run the mcmc method
        self.sampler.mcmc_go(nwalk_mult=walkers, nstep_mult=steps)

        # Save the chi-sq best fit
        self.best_fit[name + ' (chi2)'] = self.sampler.spectrum.best_fit['best']

        # Make plots
        if report is not None:
            self.sampler.plot_chains()

        # Generate best fit spectrum the 50th quantile value
        best_fit_params = {k: v for k, v in zip(self.sampler.all_params, self.sampler.all_quantiles.T[1])}
        params_with_unc = self.sampler.get_error_and_unc()
        for param, quant in zip(self.sampler.all_params, params_with_unc):
            best_fit_params['{}_unc'.format(param)] = np.mean([quant[0], quant[2]])

        # Add missing parameters
        for param in model_grid.parameters:
            if param not in best_fit_params:
                best_fit_params[param] = getattr(model_grid, '{}_vals'.format(param))[0]

        # Get best fit model and scale to spectrum
        model = model_grid.get_spectrum(**{param: best_fit_params[param] for param in model_grid.parameters})
        model = model.norm_to_spec(self)

        # Make dict for best fit model
        best_fit_params['label'] = model.name
        best_fit_params['filepath'] = None
        best_fit_params['spectrum'] = np.array(model.spectrum)
        best_fit_params['full_model'] = model
        best_fit_params['const'] = 1.
        self.best_fit[name] = best_fit_params

    @copy_raw
    def norm_to_mags(self, photometry, force=False, exclude=[], include=[]):
        """
        Normalize the spectrum to the given bandpasses

        Parameters
        ----------
        photometry: astropy.table.QTable
            A table of the photometry
        exclude: sequence (optional)
            A list of bands to exclude from the normalization
        include: sequence (optional)
            A list of bands to include in the normalization

        Returns
        -------
        sedkit.spectrum.Spectrum
            The normalized spectrum object
        """
        # Default norm
        norm = 1

        # Compile list of photometry to include
        keep = []
        for band in photometry['band']:

            # Keep only explicitly included bands...
            if include:
                if band in include:
                    keep.append(band)

            # ...or just keep non excluded bands
            else:
                if band not in exclude:
                    keep.append(band)

        # Trim the table
        photometry = photometry[[idx for idx, bnd in enumerate(photometry['band']) if bnd in keep]]

        if len(photometry) == 0:
            self.message('No photometry to normalize this spectrum.')

        else:
            # Calculate all the synthetic magnitudes
            data = []
            for row in photometry:

                try:
                    bp = row['bandpass']
                    syn_flx, syn_unc = self.synthetic_flux(bp, force=force)
                    if syn_flx is not None:
                        flx = row['app_flux']
                        unc = row['app_flux_unc']
                        weight = bp.fwhm.value
                        unc = unc.value if hasattr(unc, 'unit') else None
                        syn_unc = syn_unc.value if hasattr(syn_unc, 'unit') else None
                        data.append([flx.value, unc, syn_flx.value, syn_unc, weight])
                except IOError:
                    pass

            # Check if there is overlap
            if len(data) == 0:
                self.message('No photometry in the range {} to normalize this spectrum.'.format([self.wave_min, self.wave_max]))

            # Calculate the weighted normalization
            else:
                f1, e1, f2, e2, weights = np.array(data).T
                if not any(e1):
                    e1 = None
                if not any(e2):
                    e2 = None
                gstat, norm = u.goodness(f1, f2, e1, e2, weights)

        # Make new spectrum
        spectrum = self.spectrum
        spectrum[1] *= norm
        if self.unc is not None:
            spectrum[2] *= norm

        return Spectrum(*spectrum, name=self.name)

    @copy_raw
    def norm_to_spec(self, spec, add=False, plot=False, **kwargs):
        """Normalize the spectrum to another spectrum

        Parameters
        ----------
        spec: sedkit.spectrum.Spectrum
            The spectrum to normalize to
        add: bool
            Add spec to self after normalization
        plot: bool
            Plot the spectra

        Returns
        -------
        sedkit.spectrum.Spectrum
          The normalized spectrum
        """
        # Resample self onto spec wavelengths
        w0 = self.wave * self.wave_units.to(spec.wave_units)
        slf = self.resamp(spec.spectrum[0])

        # Trim both to just overlapping wavelengths
        idx = u.idx_overlap(w0, spec.wave, inclusive=True)
        spec0 = slf.data[:, idx]
        spec1 = spec.data[:, idx]

        # Find the normalization factor
        norm = u.minimize_norm(spec1[1], spec0[1], **kwargs)

        # Make new spectrum
        spectrum = self.spectrum
        spectrum[1] *= norm
        if self.unc is not None:
            spectrum[2] *= norm

        # Make the new spectrum
        new_spec = Spectrum(*spectrum, name=self.name)

        # Add them together if necessary
        if add:
            new_spec = new_spec + spec

        if plot:
            # Rename and plot each
            new_spec.name = 'Normalized'
            self.name = 'Input'
            spec.name = 'Target'
            fig = new_spec.plot()
            fig = self.plot(fig=fig)
            fig = spec.plot(fig=fig)
            show(fig)

        return new_spec

    def plot(self, fig=None, components=False, best_fit=True, scale='log', draw=False, const=1., **kwargs):
        """Plot the spectrum

        Parameters
        ----------
        fig: bokeh.figure (optional)
            The figure to plot on
        components: bool
            Plot all components of the spectrum
        best_fit: bool
            Plot the best fit model if available
        scale: str
            The scale of the x and y axes, ['linear', 'log']
        draw: bool
            Draw the plot rather than just return it

        Returns
        -------
        bokeh.figure
            The figure
        """
        # Make the figure
        if fig is None:
            fig = figure(x_axis_type=scale, y_axis_type=scale)
            fig.xaxis.axis_label = "Wavelength [{}]".format(self.wave_units)
            fig.yaxis.axis_label = "Flux Density [{}]".format(self.flux_units)

        # Plot the spectrum
        c = kwargs.get('color', next(u.COLORS))

        # Scatter plot
        if self.phot:

            idx = np.arange(len(self.wave))

            # Uncertainties
            if self.unc is not None:

                # Plot points with errors
                idx, = np.where(self.unc > 0)
                if len(idx) > 0:
                    fig.circle(self.wave[idx], self.flux[idx] * const, color=c, fill_alpha=0.7, size=8, legend_label=self.name)
                    y_err_x = [(px, px) for px in self.wave[idx]]
                    y_err_y = [(py - err, py + err) for py, err in zip(self.flux[idx], self.unc[idx])]
                    fig.multi_line(y_err_x, y_err_y, color=c)

                # Plot points without errors
                idn, = np.where(self.unc <= 0)
                if len(idn) > 0:
                    fig.circle(self.wave[idn], self.flux[idn] * const, color=c, fill_alpha=0, size=8)

            else:
                fig.circle(self.wave, self.flux * const, color=c, fill_alpha=0.7, size=8, legend_label=self.name)

            # Plot the best fit
            if best_fit:
                for name, bf in self.best_fit.items():
                    fig.line(bf.spectrum[0], bf.spectrum[1] * const, alpha=0.3, color=next(u.COLORS),
                             legend_label=bf.label)

        # Line plot
        else:
            fig.line(self.wave, self.flux * const, color=c, alpha=0.8, legend_label=self.name)

            # Plot the uncertainties
            if self.unc is not None:
                band_x = np.append(self.wave, self.wave[::-1])
                band_y = np.append((self.flux - self.unc) * const, (self.flux + self.unc)[::-1] * const)
                fig.patch(band_x, band_y, color=c, fill_alpha=0.1, line_alpha=0)

            # Plot the components
            if components and self.components is not None:
                for spec in self.components:
                    fig.line(spec.wave, spec.flux * const, color=next(u.COLORS), legend_label=spec.name)

            # Plot the best fit
            if best_fit:
                for name, bf in self.best_fit.items():
                    fig.line(bf.spectrum[0], bf.spectrum[1] * const, alpha=0.3, color=next(u.COLORS), legend_label=bf.label)

        if draw:
            show(fig)
        else:
            return fig

    def renormalize(self, mag, bandpass, system='vegamag', force=False, no_spec=False, name=None):
        """Renormalize the spectrum to the given magnitude

        Parameters
        ----------
        mag: float
            The target magnitude
        bandpass: svo_filters.svo.Filter, str
            The bandpass to use
        system: str
            The magnitude system to use
        force: bool
            Force the synthetic photometry even if incomplete coverage
        no_spec: bool
            Return the normalization constant only

        Returns
        -------
        float, sedkit.spectrum.Spectrum
            The normalization constant or normalized spectrum object
        """
        # My solution
        norm = u.mag2flux(bandpass, mag)[0] / self.synthetic_flux(bandpass, force=force)[0]

        # Just return the normalization factor
        if no_spec:
            return float(norm.value)

        # Scale the spectrum
        spectrum = self.spectrum
        spectrum[1] *= norm
        if self.unc is not None:
            spectrum[2] *= norm

        return Spectrum(*spectrum, name=name or self.name)

    @copy_raw
    def resamp(self, wave=None, resolution=None, name=None):
        """Resample the spectrum onto a new wavelength array or to a new
        resolution

        Parameters
        ----------
        wave: astropy.units.quantity.Quantity (optional)
            The wavelength array to resample onto
        resolution: int (optional)
            The new resolution to resample to, keeping the same
            wavelength range

        Returns
        -------
        sedkit.spectrum.Spectrum
            The resampled spectrum
        """
        # Generate wavelength if resolution is set
        if resolution is not None:

            # Make the wavelength array
            mn = np.nanmin(self.wave)
            mx = np.nanmax(self.wave)
            d_lam = (mx - mn) / resolution
            wave = np.arange(mn, mx, d_lam) * self.wave_units

        if not u.equivalent(wave, q.um):
            raise TypeError("wave must be in length units")

        # Convert wave to target units
        self.wave_units = wave.unit
        wave = wave.value

        # Bin the spectrum
        binned = u.spectres(wave, self.wave, self.flux, self.unc)

        # Update the spectrum
        spectrum = [i * Q for i, Q in zip(binned, self.units)]

        return Spectrum(*spectrum, name=name or self.name)

    def restore(self):
        """
        Restore the spectrum to the original raw data
        """
        return Spectrum(*(self.raw or self.spectrum), name=self.name)

    def _set_units(self):
        """Set the units for convenience"""
        self.units = [self._wave_units, self._flux_units]
        self.units += [self._flux_units] if self.unc is not None else []

    @property
    def size(self):
        """The length of the data"""
        return len(self.wave)

    @copy_raw
    def smooth(self, beta, window=11):
        """
        Smooths the spectrum using a Kaiser-Bessel smoothing window of
        narrowness *beta* (~1 => very smooth, ~100 => not smooth)

        Parameters
        ----------
        beta: float, int
            The narrowness of the window
        window: int
            The length of the window

        Returns
        -------
        sedkit.spectrum.Spectrum
            The smoothed spectrum
        """
        s = np.r_[self.flux[window - 1:0:-1], self.flux, self.flux[-1:-window:-1]]
        w = np.kaiser(window, beta)
        y = np.convolve(w / w.sum(), s, mode='valid')
        smoothed = y[5:len(y) - 5]

        # Replace with smoothed spectrum
        spectrum = self.spectrum
        spectrum[1] = smoothed * self.flux_units

        return Spectrum(*spectrum, name=self.name)

    @property
    def spectrum(self):
        """Store the spectrum with units
        """
        return [i * Q for i, Q in zip(self.data, self.units)]

    def synthetic_flux(self, bandpass, force=False, plot=False):
        """
        Calculate the magnitude in a bandpass

        Parameters
        ----------
        bandpass: svo_filters.svo.Filter
            The bandpass to use
        force: bool
            Force the magnitude calculation even if
            overlap is only partial
        plot: bool
            Plot the spectrum and the flux point

        Returns
        -------
        float
            The magnitude
        """
        # Test overlap
        overlap = bandpass.overlap(self.spectrum)

        # Initialize
        flx = unc = None

        if overlap == 'full' or (overlap == 'partial' and force):

            # Convert self to bandpass units
            self.wave_units = bandpass.wave_units

            # Calculate the bits
            wav = bandpass.wave[0]
            rsr = bandpass.throughput
            grad = np.gradient(wav).value

            # Interpolate the spectrum to the filter wavelengths
            f = np.interp(wav.value, self.wave, self.flux, left=0, right=0) * self.flux_units

            # Filter out NaNs
            idx = np.where([not np.isnan(i) for i in f])[0]

            # Calculate the flux
            flx = (np.trapz(f[idx] * rsr[0][idx], x=wav[idx]) / np.trapz(rsr[0][idx], x=wav[idx])).to(self.flux_units)

            # Calculate uncertainty
            if self.unc is not None:
                sig_f = np.interp(wav.value, self.wave, self.unc, left=0, right=0) * self.flux_units
                unc = np.sqrt(np.sum(((sig_f * rsr * grad)**2).to(self.flux_units**2)))
            else:
                unc = None

            # Plot it
            if plot:
                fig = figure()
                fig.line(self.wave, self.flux, color='navy')
                fig.circle([bandpass.wave_eff], [flx], color='red')
                show(fig)

        return flx, unc

    def synthetic_magnitude(self, bandpass, force=False):
        """
        Calculate the synthetic magnitude in the given bandpass

        Parameters
        ----------
        bandpass: svo_filters.svo.Filter
            The bandpass to use
        force: bool
            Force the magnitude calculation even if
            overlap is only partial

        Returns
        -------
        tuple
            The flux and uncertainty
        """
        flx, flx_unc = self.synthetic_flux(bandpass, force=force)

        # Calculate the magnitude
        mag, mag_unc = None, None
        if flx is not None:
            mag, mag_unc = u.flux2mag((flx, flx_unc), bandpass)

        return mag, mag_unc

    def trim(self, include=[], exclude=[], concat=False):
        """Trim the spectrum in the given wavelength ranges

        Parameters
        ----------
        include: sequence
            The (min_wave, max_wave) ranges to include from the spectrum
        exclude: sequence
            The (min_wave, max_wave) ranges to exclude from the spectrum
        """
        if not isinstance(include, (list, tuple)):
            print("""Please provide a list of (lower,upper) bounds with units to include in trim, e.g. [(0*q.um,0.8*q.um)]""")

        if not isinstance(exclude, (list, tuple)):
            print("""Please provide a list of (lower,upper) bounds with units to exclude in trim, e.g. [(0*q.um,0.8*q.um)]""")

        # Default to include everything
        if len(include) == 0:
            include.append([self.wave_min, self.wave_max])

        # Get element indexes of included ranges
        idx_include = []
        for mn, mx in include:
            inc, = np.where(np.logical_and(self.spectrum[0] > mn, self.spectrum[0] < mx))
            idx_include.append(inc)

        # Get element indexes of excluded ranges
        idx_exclude = []
        for mn, mx in exclude:
            exc, = np.where(np.logical_and(self.spectrum[0] > mn, self.spectrum[0] < mx))
            idx_exclude.append(exc)

        # Get difference of each included set with each excluded set
        segments = []
        for inc in idx_include:
            set_inc = set(inc)
            for exc in idx_exclude:
                set_inc = set_inc.difference(set(exc))

            # Split the indexes by missing elements
            for k, g in groupby(enumerate(list(set_inc)), lambda i_x: i_x[0] - i_x[1]):

                # Make the spectra
                group = list(map(itemgetter(1), g))
                data = [i[group] for i in copy.copy(self.spectrum)]
                spectrum = Spectrum(*data, name=self.name)
                spectrum.wave_units = self.wave_units
                segments.append(spectrum)

        return np.sum(segments) if concat else segments

    @property
    def wave_max(self):
        """The minimum wavelength"""
        return max(self.wave) * self.wave_units

    @property
    def wave_min(self):
        """The minimum wavelength"""
        return min(self.wave) * self.wave_units

    @property
    def wave_units(self):
        """A property for wave_units"""
        return self._wave_units

    @wave_units.setter
    def wave_units(self, wave_units):
        """A setter for wave_units

        Parameters
        ----------
        wave_units: astropy.units.quantity.Quantity
            The astropy units of the SED wavelength
        """
        # Make sure the values are in length units
        if not u.equivalent(wave_units, q.um):
            raise TypeError("wave_units must be a unit of length, e.g. 'um'")

        # Update the wavelength array
        self.wave = self.wave * self.wave_units.to(wave_units)

        # Set the wave_units
        self._wave_units = wave_units
        self._set_units()


class Blackbody(Spectrum):
    """A spectrum object specifically for blackbodies"""
    def __init__(self, wavelength, teff, radius=None, distance=None, **kwargs):
        """
        Given a wavelength array and temperature, returns an array of Planck
        function values in [erg s-1 cm-2 A-1]

        Parameters
        ----------
        wavelength: array-like
            The array of wavelength values to evaluate the Planck function
        teff: astropy.unit.quantity.Quantity, sequence
            The effective temperature and (optional) uncertainty
        radius: astropy.unit.quantity.Quantity, sequence
            The radius and (optional) uncertainty
        distance: astropy.unit.quantity.Quantity, sequence
            The distance and (optional) uncertainty
        """
        # Check wavelength units
        if not u.equivalent(wavelength, q.um):
            raise TypeError("Wavelength must be in astropy units of length, e.g. 'um'")

        # Check teff
        if not u.issequence(teff, length=2):
            teff = teff, None
        if not u.equivalent(teff[0], q.K):
            raise TypeError("teff must be in astropy units of temperature, eg. 'K'")
        if not u.equivalent(teff[1], q.K) and teff[1] is not None:
            raise TypeError("teff_unc must be in astropy units of temperature, eg. 'K'")
        self.teff, self.teff_unc = teff

        # Check radius
        if not u.issequence(radius, length=2):
            radius = radius, None
        if not u.equivalent(radius[0], q.R_jup) and radius[0] is not None:
            raise TypeError("radius must be in astropy units of length, eg. 'R_jup'")
        if not u.equivalent(radius[1], q.R_jup) and radius[1] is not None:
            raise TypeError("radius_unc must be in astropy units of length, eg. 'R_jup'")
        self.radius, self.radius_unc = radius

        # Check distance
        if u.issequence(distance, length=3):
            distance = distance[:2]
        if not u.issequence(distance):
            distance = distance, None
        if not u.equivalent(distance[0], q.pc) and distance[0] is not None:
            raise TypeError("distance must be in astropy units of length, eg. 'pc'")
        if not u.equivalent(distance[1], q.pc) and distance[1] is not None:
            raise TypeError("distance_unc must be in astropy units of length, eg. 'pc'")
        self.distance, self.distance_unc = distance

        # Evaluate
        I, I_unc = self.eval(wavelength)

        # Inherit from Spectrum
        super().__init__(wavelength, I, I_unc, **kwargs)

        self.name = '{} Blackbody'.format(teff)

    def eval(self, wavelength, Flam=False):
        """Evaluate the blackbody at the given wavelengths

        Parameters
        ----------
        wavelength: sequence
            The wavelength values

        Returns
        -------
        list
            The blackbody flux and uncertainty arrays
        """
        if not u.equivalent(wavelength, q.um):
            raise TypeError("Wavelength must be in astropy units, e.g. 'um'")

        units = q.erg / q.s / q.cm**2 / (1 if Flam else q.AA)

        # Check for radius and distance
        if self.radius is not None and self.distance is not None:
            scale = (self.radius**2 / self.distance**2).decompose()
        else:
            scale = 1.

        # Get numerator and denominator
        const = ac.h * ac.c / (wavelength * ac.k_B)
        numer = 2 * np.pi * ac.h * ac.c**2 * scale / (wavelength**(4 if Flam else 5))
        denom = np.exp((const / self.teff)).decompose()

        # Calculate intensity
        I = (numer / (denom - 1.)).to(units)

        # Calculate dI/dr
        if self.radius is not None and self.radius_unc is not None:
            dIdr = (self.radius_unc * 2 * I / self.radius).to(units)
        else:
            dIdr = 0. * units

        # Calculate dI/dd
        if self.distance is not None and self.distance_unc is not None:
            dIdd = (self.distance_unc * 2 * I / self.distance).to(units)
        else:
            dIdd = 0. * units

        # Calculate dI/dT
        if self.teff is not None and self.teff_unc is not None:
            dIdT = (self.teff_unc * I * ac.h * ac.c / wavelength / ac.k_B / self.teff**2).to(units)
        else:
            dIdT = 0. * units

        # Calculate sigma_I from derivative terms
        I_unc = np.sqrt(dIdT**2 + dIdr**2 + dIdd**2)
        if isinstance(I_unc.value, float):
            I_unc = None

        return I, I_unc


class FileSpectrum(Spectrum):
    def __init__(self, file, wave_units=None, flux_units=None, ext=0,
                 survey=None, name=None, **kwargs):
        """Create a spectrum from an ASCII or FITS file

        Parameters
        ----------
        file: str
            The path to the ascii or FITS file
        wave_units: astropy.units.quantity.Quantity
            The wavelength units
        flux_units: astropy.units.quantity.Quantity
            The flux units
        ext: int, str
            The FITS extension name or index
        survey: str (optional)
            The name of the survey
        """
        # Read the fits data...
        if file.endswith('.fits'):

            if file.endswith('.fits'):
                data = u.spectrum_from_fits(file, ext=ext)

            elif survey == 'SDSS':
                head = fits.getheader(file)
                raw = fits.getdata(file)
                flux_units = 1E-17 * q.erg / q.s / q.cm**2 / q.AA
                wave_units = q.AA
                log_w = head['COEFF0'] + head['COEFF1'] * np.arange(len(raw.flux))
                data = [10**log_w, raw.flux, raw.ivar]

            # Check if it is a recarray
            elif isinstance(raw, fits.fitsrec.FITS_rec):

                # Check if it's an SDSS spectrum
                raw = fits.getdata(file, ext=ext)
                data = raw['WAVELENGTH'], raw['FLUX'], raw['ERROR']

            # Otherwise just an array
            else:
                print("Sorry, I cannot read the file at", file)

        # ...or the ascii data...
        elif file.endswith('.txt'):
            data = np.genfromtxt(file, unpack=True)

        # ...or the VO Table
        elif file.endswith('.xml'):
            vot = vo.parse_single_table(file)
            data = np.array([list(i) for i in vot.array]).T

        else:
            raise IOError('The file needs to be ASCII, XML, or FITS.')

        # Make sure units are astropy quantities
        if isinstance(wave_units, str):
            wave_units = q.Unit(wave_units)
        if isinstance(flux_units, str):
            flux_units = q.Unit(flux_units)

        # Sanity check for wave_units
        if data[0].min() > 100 and wave_units == q.um:
            print("WARNING: Your wavelength range ({} - {}) looks like Angstroms. Are you sure it's {}?".format(data[0].min(), data[0].max(), wave_units))

        # Apply units
        wave = data[0] * wave_units
        flux = data[1] * flux_units
        if len(data) > 2:
            unc = data[2] * flux_units
        else:
            unc = None

        if name is None:
            name = file

        super().__init__(wave, flux, unc, name=name, **kwargs)


class Vega(Spectrum):
    """A Spectrum object of Vega"""
    def __init__(self, wave_units=q.AA, flux_units=q.erg / q.s / q.cm**2 / q.AA, **kwargs):
        """Initialize the Spectrum object

        Parameters
        ----------
        wave_units: astropy.units.quantity.Quantity
            The desired wavelength units
        flux_units: astropy.units.quantity.Quantity
            The desired flux units
        """
        # Get the data and apply units
        vega_file = resource_filename('sedkit', 'data/STScI_Vega.txt')
        ref = '2007ASPC..364..315B, 2004AJ....127.3508B, 2005MSAIS...8..189K'
        wave, flux = np.genfromtxt(vega_file, unpack=True)
        wave *= q.AA
        flux *= q.erg / q.s / q.cm**2 / q.AA

        # Convert to target units
        wave = wave.to(wave_units)
        flux = flux.to(flux_units)

        # Make the Spectrum object
        super().__init__(wave, flux, ref=ref, **kwargs)

        self.name = 'Vega'


def fit_model(row, fitspec, wave_units=q.AA):
    """Fit the model grid row to the spectrum with the given parameters

    Parameters
    ----------
    row: pandas.Row
        The dataframe row to fit
    fitspec: sedkit.spectrum.Spectrum
        The spectrum to fit

    Returns
    -------
    pandas.Row
        The input row with the normalized spectrum and additional gstat
    """
    try:
        gstat, yn, xn = list(fitspec.fit(row['spectrum'], weights=row.get('weights'), wave_units=wave_units))
        spectrum = np.array([row['spectrum'][0] * xn, row['spectrum'][1] * yn])
        row['spectrum'] = spectrum
        row['gstat'] = gstat
        row['const'] = yn

    except ValueError:
        row['gstat'] = np.nan

    return row
