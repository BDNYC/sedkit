#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
# !python3
"""
Make nice spectrum objects to pass around SED class
"""
import copy
from contextlib import closing
from functools import wraps
import os
from pkg_resources import resource_filename
import shutil
import urllib.request as request

import astropy.constants as ac
import astropy.units as q
import astropy.io.votable as vo
from astropy.io import fits
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool
from functools import partial
from multiprocessing import Pool
import numpy as np
from pandas import DataFrame
from scipy import interpolate, ndimage
from svo_filters import Filter

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
    def __init__(self, wave, flux, unc=None, snr=None, trim=None, name=None,
                 ref=None, verbose=False, **kwargs):
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
        trim: sequence (optional)
            A sequence of (wave_min, wave_max) sequences to override spectrum
            trimming
        name: str
            A name for the spectrum
        ref: str
            A reference for the data
        verbose: bool
            Print helpful stuff
        """
        # Meta
        self.verbose = verbose
        self.name = name or 'New Spectrum'
        self.ref = ref

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

        # Trim manually
        if trim is not None:
            trimmed = self.trim(trim)
            self.__dict__ = trimmed.__dict__

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

    def best_fit_model(self, modelgrid, report=None, name=None):
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
        spectrum = Spectrum(*self.spectrum)
        rows = [row for n, row in modelgrid.index.iterrows()]

        # Iterate over entire model grid
        pool = Pool(8)
        func = partial(fit_model, fitspec=spectrum)
        fit_rows = pool.map(func, rows)
        pool.close()
        pool.join()

        # Turn the results into a DataFrame and sort
        models = DataFrame(fit_rows)
        models = models.sort_values('gstat')

        # Get the best fit
        bf = copy.copy(models.iloc[0])

        if self.verbose:
            print(bf[modelgrid.parameters])

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
            rep.circle(report, 'gstat', source=best, color='red', legend=bf['label'])
            rep.circle(report, 'gstat', source=others)

            # Show the plot
            show(rep)

        if bf['filepath'] in [i['filepath'] for n, i in self.best_fit.items()]:
            print('{}: model has already been fit'.format(bf['filepath']))
        else:
            self.best_fit[name] = bf

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
            fig.line(spec.wave, spec.flux * ynorm, legend='Fit')
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

    def _set_units(self):
        """Set the units for convenience"""
        self.units = [self._wave_units, self._flux_units]
        self.units += [self._flux_units] if self.unc is not None else []

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
            if self.verbose:
                print('No photometry to normalize this spectrum.')

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
                if self.verbose:
                    print('No photometry in the range {} to normalize this spectrum.'.format([self.wave_min, self.wave_max]))

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
        fig.line(self.wave, self.flux * const, color=c, alpha=0.8, legend=self.name)

        # Plot the uncertainties
        if self.unc is not None:
            band_x = np.append(self.wave, self.wave[::-1])
            band_y = np.append((self.flux - self.unc) * const, (self.flux + self.unc)[::-1] * const)
            fig.patch(band_x, band_y, color=c, fill_alpha=0.1, line_alpha=0)

        # Plot the components
        if components and self.components is not None:
            for spec in self.components:
                fig.line(spec.wave, spec.flux * const, color=next(u.COLORS), legend=spec.name)

        # Plot the best fit
        if best_fit:
            for name, bf in self.best_fit.items():
                fig.line(bf.spectrum[0], bf.spectrum[1] * const, alpha=0.3, color=next(u.COLORS), legend=bf.label)

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
        bandpass: sedkit.synphot.Bandpass
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
    def size(self):
        """The length of the data"""
        return len(self.wave)

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
                fig.circle([bandpass.eff], [flx], color='red')
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

    @copy_raw
    def trim(self, ranges):
        """Trim the spectrum in the given wavelength ranges

        Parameters
        ----------
        ranges: sequence
            The (min_wave, max_wave) ranges to trim from the spectrum
        """
        # Iterate over trim ranges
        if isinstance(ranges, (list, tuple)):
            for mn, mx in ranges:
                try:
                    idx, = np.where((self.spectrum[0] < mn) | (self.spectrum[0] > mx))

                    if len(idx) > 0:

                        # Trim the data
                        spectrum = [i[idx] for i in self.spectrum]

                        return Spectrum(*spectrum, name=self.name)

                except TypeError:
                    print("""Please provide a list of (lower,upper) bounds with units to trim, e.g. [(0*q.um,0.8*q.um)]""")

        else:
            raise TypeError("""Please provide a list of (lower,upper) bounds with units to trim, e.g. [(0*q.um,0.8*q.um)]""")

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
        wave, flux = np.genfromtxt(vega_file, unpack=True)
        wave *= q.AA
        flux *= q.erg / q.s / q.cm**2 / q.AA

        # Convert to target units
        wave = wave.to(wave_units)
        flux = flux.to(flux_units)

        # Make the Spectrum object
        super().__init__(wave, flux, **kwargs)

        self.name = 'Vega'


class ModelSpectrum(Spectrum):
    """Generate a test object with a theoretical ATLAS or PHOENIX stellar spectrum of choice"""
    def __init__(self, name=None, teff=5700.0, logg=4.0, feh=0.0, alpha=0.0, bandpass='2MASS.J', mag=9.0, stellar_model='PHOENIX', modeldir='.', **kwargs):
        """Get the test data and load the object

        Parmeters
        ---------
        teff: double
            The effective temperature [K] of the stellar source
        logg: double
            The log-gravity of the stellar source
        feh: double
            The [Fe/H] of the stellar source
        alpha: double
            The alpha enhancement of the stellar source
        bandpass: str
            The name of the bandpass used for scaling
        mag: double
            The J magnitude of the source. This will be used to scale the model stellar flux to Earth-values.
        stellar_model: str
            The stellar model grid to use. Can either be 'ATLAS' or 'PHOENIX'. Default is 'ATLAS'
        """
        if name is None:
            name = '/'.join([str(int(teff)), str(logg), str(feh), str(alpha)])

        # Set model directory
        self.modeldir = modeldir

        # Retrieve PHOENIX or ATLAS stellar models:
        if stellar_model.lower() == 'phoenix':
            w, f = self.get_phoenix_model(feh, alpha, teff, logg)
        elif stellar_model.lower() == 'atlas':
            w, f = self.get_atlas_model(feh, teff, logg)

        # Now scale model spectrum to user-input J-band:
        f = self.scale_spectrum(w, f, bandpass, mag)
        self.stellar_spectrum_wav = w
        self.stellar_spectrum_flux = f

        # Initialize base class
        super().__init__(name=name, wave=w, flux=f, **kwargs)

    def closest_value(self, input_value, possible_values):
        """
        This function calculates, given an input_value and an array of possible_values,
        the closest value to input_value in the array.

        Parameters
        ----------
        input_value: double
             Input value to compare against possible_values.
        possible_values: np.ndarray
             Array of possible values to compare against input_value.

        Returns
        -------
        double
            Closest value on possible_values to input_value.
        """
        distance = np.abs(possible_values - input_value)
        idx = np.where(distance == np.min(distance))[0]

        return possible_values[idx[0]]

    def get_atlas_folder(self, feh):
        """
        Given input metallicity, this function defines the first part of the URL that will define what
        file to download from the STScI website.

        Parameters
        ----------
        feh: np.double
             [Fe/H] of the desired spectrum.

        Returns
        -------
        string
            URL of ATLAS models closer to the input metallicity.
        """
        # Define closest possible metallicity from ATLAS models:
        model_metallicity = self.closest_value(feh, np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.2, 0.5]))
        met_sign = 'm'

        # Define the sign before the filename, obtain absolute value if needed:
        if model_metallicity >= 0.0:
            met_sign = 'p'
        else:
            model_metallicity = np.abs(model_metallicity)

        model_metallicity = ''.join(str(model_metallicity).split('.'))
        fname = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/ck{0:}{1:}/'.format(met_sign, model_metallicity)

        return fname

    def get_phoenix_folder(self, feh, alpha):
        """
        Given input metallicity and alpha-enhancement, this function defines the first part of the URL that will define what
        file to download from the PHOENIX site.

        Parameters
        ----------
        feh: np.double
             [Fe/H] of the desired spectrum.
        alpha: np.double
             Alpha-enhancement of the desired spectrum.

        Returns
        -------
        string
            FTP URL of PHOENIX file with the closest properties to the input properties.

        """
        # Define closest possible metallicity from PHOENIX models:
        model_metallicity = self.closest_value(feh, np.array([-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]))

        # Same for alpha-enhancement:
        model_alpha = self.closest_value(alpha, np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.20]))
        met_sign, alpha_sign = '-', '-'

        # Define the sign before the filename, obtain absolute value if needed:
        if model_metallicity > 0.0:
            met_sign = '+'
        else:
            model_metallicity = np.abs(model_metallicity)
        if model_alpha > 0.0:
            alpha_sign = '+'
        else:
            model_alpha = np.abs(model_alpha)

        # Create the folder name
        if alpha == 0.0:
            fname = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z{0:}{1:.1f}/'.format(met_sign, model_metallicity)
        else:
            fname = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z{0:}{1:.1f}.Alpha={2:}{3:.2f}/'.format(met_sign, model_metallicity, alpha_sign, model_alpha)

        return fname

    def download(self, url, fname):
        """
        Download files from ftp server at url in filename fname. Obtained/modified from jfs here: https://stackoverflow.com/questions/11768214/python-download-a-file-over-an-ftp-server

        Parameters
        ----------
        url: string
            URL pointing to the file to download.
        fname: string
            Output filename of the file to download.
        """
        with closing(request.urlopen(url)) as r:
            with open(fname, 'wb') as f:
                shutil.copyfileobj(r, f)

    def read_phoenix_list(self, phoenix_model_list):
        """
        This function extracts filenames, effective temperatures and log-gs given a filename that contains a list of PHOENIX model filenames.

        Parameters
        ----------
        phoenix_model_list: string
             Filename of file containing, on each row, a PHOENIX model filename.
        Returns
        -------
        np.ndarray
            Array of PHOENIX model filenames
        np.ndarray
            Array containing the effective temperatures (K) of each PHOENIX model filename.
        np.nadarray
            Array containing the log-g (cgs) of each PHOENIX model filename.
        """
        fin = open(phoenix_model_list, 'r')
        fnames = np.array([])
        teffs = np.array([])
        loggs = np.array([])

        while True:
            line = fin.readline()

            if line != '':
                fname = line.split()[-1]
                teff, logg = fname.split('-')[:2]
                fnames = np.append(fnames, fname)
                teffs = np.append(teffs, np.double(teff[3:]))
                loggs = np.append(loggs, np.double(logg))

            else:
                break

        return fnames, teffs, loggs

    def get_phoenix_model(self, feh, alpha, teff, logg):
        """
        This function gets you the closest PHOENIX high-resolution model to the input stellar parameters from the Goettingen website (ftp://phoenix.astro.physik.uni-goettingen.de).

        Parameters
        ----------
        feh: np.double
             [Fe/H] of the desired spectrum.
        alpha: np.double
             Alpha-enhancement of the desired spectrum.
        teff: np.double
             Effective temperature (K) of the desired spectrum.
        logg: np.double
             Log-gravity (cgs) of the desired spectrum.
        Returns
        -------
        np.ndarray
            Wavelength in um of the closest spectrum to input properties.
        np.ndarray
            Surface flux in f-lambda of the closest spectrum to input properties in units of erg/s/cm**2/angstroms.
        """
        # First get grid corresponding to input Fe/H and alpha:
        url_folder = self.get_phoenix_folder(feh, alpha)

        # Now define details for filenames and folders. First, extract metallicity and alpha-enhancement in
        # the PHOENIX filename format (and get rid of the "Z" in, e.g., "Z-1.0.Alpha=-0.20"):
        phoenix_met_and_alpha = url_folder.split('/')[-2][1:]

        # Define folders where we will save (1) all stellar model data and (2) all phoenix models:
        phoenix_folder_path = os.path.join(self.modeldir, 'phoenix')
        model_folder_path = os.path.join(phoenix_folder_path, 'phoenix_met_and_alpha')

        # Check if we even have stellarmodels folder created. Create it if not:
        if not os.path.exists(self.modeldir):
            os.mkdir(self.modeldir)

        # Same for phoenix folder:
        if not os.path.exists(phoenix_folder_path):
            os.mkdir(phoenix_folder_path)

        # Check if the current metallicity-alpha folder exists as well:
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)

        # Check if we have the PHOENIX wavelength solution. If not, download it:
        if not os.path.exists(phoenix_folder_path + 'wavsol.fits'):
            self.download('ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits', phoenix_folder_path + 'wavsol.fits')

        # Extract wavelength solution:
        wavelengths = fits.getdata(phoenix_folder_path + 'wavsol.fits')

        # Now, figure out the closest model to the input stellar parameters. For this, first figure out the range of teff and logg
        # for the current metallicity and alpha. For this, either retrieve from the system or download the full list of PHOENIX models
        # for the current metallicity and alpha. If not already here, save it on the system:
        phoenix_model_list = model_folder_path + 'model_list.txt'
        if not os.path.exists(phoenix_model_list):
            self.download(url_folder, phoenix_model_list)

        # Extract information from this list:
        model_names, possible_teffs, possible_loggs = self.read_phoenix_list(model_folder_path + 'model_list.txt')

        # Search the closest to the input teff:
        phoenix_teff = self.closest_value(teff, possible_teffs)

        # Raise a warning in case the found teff is outside the PHOENIX model range, give some
        # guidance on how to proceed:
        if np.abs(phoenix_teff - teff) > 200.:
            print('\t Warning: the input stellar effective temperature is outside the {0:}-{1:} K model range of PHOENIX models for {2:}.'.format(np.min(possible_teffs),
                  np.max(possible_teffs), phoenix_met_and_alpha))

            if 'Alpha' in phoenix_met_and_alpha:
                print('\t Modelling using a {0:} K model. Using models without alpha-enhancement (alpha = 0.0), which range from 2300 to 12000 K would perhaps help find more suitable temperature models.'.format(phoenix_teff))

            else:
                print('\t Modelling using a {0:} K model.'.format(phoenix_teff))

        # Same excercise for logg, given the teffs:
        idx_logg = np.where(np.abs(phoenix_teff - possible_teffs) == 0.)[0]
        phoenix_logg = self.closest_value(logg, possible_loggs[idx_logg])

        # Select final model:
        idx = np.where((np.abs(phoenix_teff - possible_teffs) == 0.) & (np.abs(possible_loggs == phoenix_logg)))[0]
        phoenix_model, phoenix_logg = model_names[idx][0], possible_loggs[idx][0]

        # Raise warning for logg as well:
        if np.abs(phoenix_logg - logg) > 0.5:
            print('\t Warning: the input stellar log-gravity is outside the {0:}-{1:} model range of PHOENIX models for {2:} and Teff {3:}.'.format(
                  np.min(possible_loggs[idx_logg]), np.max(possible_loggs[idx_logg]), phoenix_met_and_alpha, phoenix_teff))

        # Check if we already have the downloaded model. If not, download the corresponding file:
        if not os.path.exists(model_folder_path + phoenix_model):
            print('\t PHOENIX stellar models for {0:} not found in {1:}. Downloading...'.format(phoenix_met_and_alpha, model_folder_path))
            self.download(url_folder + phoenix_model, model_folder_path + phoenix_model)

        # Once we have the file, simply extract the data:
        print('\t Using the {0:} PHOENIX model (Teff {1:}, logg {2:}).'.format(phoenix_model, phoenix_teff, phoenix_logg))
        flux = fits.getdata(model_folder_path + phoenix_model, header=False)

        # Change units in order to match what is expected by the TSO modules:
        wav = (wavelengths * q.angstrom).to(q.um)
        flux = (flux * (q.erg / q.s / q.cm**2 / q.cm)).to(q.erg / q.s / q.cm**2 / q.AA)

        return wav, flux

    def get_atlas_model(self, feh, teff, logg):
        """
        This function gets you the closest ATLAS9 Castelli and Kurucz model to the input stellar parameters from the STScI website
        (http://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/castelli-and-kurucz-atlas).

        Parameters
        ----------
        feh: np.double
             [Fe/H] of the desired spectrum.
        teff: np.double
             Effective temperature (K) of the desired spectrum.
        logg: np.double
             Log-gravity (cgs) of the desired spectrum.
        Returns
        -------
        np.ndarray
            Wavelength in um of the closest spectrum to input properties.
        np.ndarray
            Surface flux in f-lambda of the closest spectrum to input properties in units of erg/s/cm**2/angstroms.
        """
        # First get grid corresponding to input Fe/H:
        url_folder = self.get_atlas_folder(feh)

        # Now define details for filenames and folders. Extract foldername with the metallicity info from the url_folder:
        atlas_met = url_folder.split('/')[-2]

        # Define folders where we will save (1) all stellar model data and (2) all atlas models:
        atlas_folder_path = os.path.join(self.modeldir, 'atlas')
        model_folder_path = os.path.join(atlas_folder_path, atlas_met)

        # Check if we even have stellarmodels folder created. Create it if not:
        if not os.path.exists(self.modeldir):
            os.mkdir(self.modeldir)

        # Same for phoenix folder:
        if not os.path.exists(atlas_folder_path):
            os.mkdir(atlas_folder_path)

        # Check if the current metallicity-alpha folder exists as well:
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)

        # Define possible teff and loggs (thankfully, this is easier for ATLAS models):
        possible_teffs, possible_loggs = np.append(np.arange(3500, 13250, 250), np.arange(14000, 51000, 1000)), np.arange(0.0, 5.5, 0.5)

        # Check the closest teff and logg to input ones:
        atlas_teff = self.closest_value(teff, possible_teffs)
        atlas_logg = self.closest_value(logg, possible_loggs)

        # Raise a warning in case the found teff is outside the ATLAS model range, give some
        # guidance on how to proceed:
        if np.abs(atlas_teff - teff) > 200.:
            print('\t Warning: the input stellar effective temperature is outside the {0:}-{1:} K model range of ATLAS models for {2:}.'.format(np.min(possible_teffs),
                  np.max(possible_teffs), atlas_met))
            print('\t Modelling using a {0:} K model.'.format(atlas_teff))

        # Now, if not already in the system, download the model corresponding to the chosen teff:
        atlas_fname = model_folder_path + atlas_met + '_{0:}.fits'.format(atlas_teff)
        if not os.path.exists(atlas_fname):
            self.download(url_folder + atlas_met + '_{0:}.fits'.format(atlas_teff), atlas_fname)

        # Read the file:
        d = fits.getdata(atlas_fname)

        # This variable will save non-zero logg at the given temperatures. Only useful to report back to the user and/or input logg
        # doesn't have data:
        real_possible_loggs = np.array([])

        # Check if the closest requested logg has any data. If not, check all possible loggs for non-zero data, and select the closest
        # to the input logg that has data:
        s_logg = 'g' + ''.join('{0:.1f}'.format(atlas_logg).split('.'))
        if np.count_nonzero(d[s_logg]) != 0:
            w, f = d['WAVELENGTH'], d[s_logg]
        else:
            real_possible_loggs = np.array([])
            for loggs in possible_loggs:
                s_logg = 'g' + ''.join('{0:.1f}'.format(loggs).split('.'))
                if np.count_nonzero(d[s_logg]) != 0:
                    real_possible_loggs = np.append(real_possible_loggs, loggs)
            atlas_logg = self.closest_value(logg, real_possible_loggs)
            s_logg = 'g' + ''.join('{0:.1f}'.format(atlas_logg).split('.'))
            w, f = d['WAVELENGTH'], d[s_logg]

        # Raise warning for logg as well:
        if np.abs(atlas_logg - logg) > 0.5:

            # If real_possible_loggs is empty, calculate it:
            if len(real_possible_loggs) == 0:
                for loggs in possible_loggs:
                    s_logg = 'g' + ''.join('{0:.1f}'.format(loggs).split('.'))
                    if np.count_nonzero(d[s_logg]) != 0:
                        real_possible_loggs = np.append(real_possible_loggs, loggs)

            print('\t Warning: the input stellar log-gravity is outside the {0:}-{1:} model range of ATLAS models for {2:} and Teff {3:}.'.format(
                  np.min(real_possible_loggs), np.max(real_possible_loggs), atlas_met, atlas_teff))

        # Change units in order to match what is expected by the TSO modules:
        wav = (w * q.angstrom).to(q.um)
        flux = f * q.erg / q.s / q.cm**2 / q.AA
        return wav, flux

    def get_resolution(self, w, f):
        """
        This function returns the (w) wavelength (median) resolution of input spectra (f)

        Parameters
        ----------
        w: np.ndarray
            Wavelengths of the spectrum
        f: np.ndarray
            Value at the given wavelength (can be flux, transmission, etc.)

        Returns
        -------
        np.double
            The median resolution of the spectrum.
        """
        eff_wav = np.sum(w * f) / np.sum(f)
        delta_wav = np.median(np.abs(np.diff(w)))

        return eff_wav / delta_wav

    def spec_integral(self, input_w, input_f, wT, TT):
        """
        This function computes the integral of lambda*f*T divided by the integral of lambda*T, where
        lambda is the wavelength, f the flux (in f-lambda) and T the transmission function. The input
        stellar spectrum is given by wavelength w and flux f. The input filter response wavelengths
        are given by wT and transmission curve by TT. It is assumed both w and wT are in the same
        wavelength units.

        Parameters
        ----------
        input_w: np.ndarray
            Wavelengths of the input spectrum
        input_f: np.ndarray
            Flux (in f-lambda) of the input spectrum
        wT: np.ndarray
            Wavelength of the input transmission function
        TT: np.ndarray
            Spectral response function of the transmission function

        Returns
        -------
        np.double
            Value of the integral (over dlambda) of lambda*f*T divided by the integral (over dlambda) of lambda*T.

        """
        # If resolution of input spectra in the wavelength range of the response function
        # is higher than it, degrade it to match the transmission function resolution. First,
        # check that resolution of input spectra is indeed higher than the one of the
        # transmisssion. Resolution of input transmission first:
        min_wav, max_wav = np.min(wT), np.max(wT)
        resT = self.get_resolution(wT, TT)

        # Resolution of input spectra in the same wavelength range:
        idx = np.where((input_w >= min_wav - 10) & (input_w <= max_wav + 10))[0]
        res = self.get_resolution(input_w[idx], input_f[idx])

        # If input spetrum resolution is larger, degrade:
        if res > resT:

            # This can be way quicker if we just take the gaussian weight *at* the evaluated
            # points in the interpolation. TODO: make faster.
            f = ndimage.gaussian_filter(input_f[idx], int(np.double(len(idx)) / np.double(len(wT))))
            w = input_w[idx]
        else:
            w, f = input_w, input_f

        interp_spectra = interpolate.interp1d(w, f)
        numerator = np.trapz(wT * interp_spectra(wT) * TT, x=wT)
        denominator = np.trapz(wT * TT, x=wT)

        return numerator / denominator

    def scale_spectrum(self, w, f, bandpass, mag):
        """
        This function scales an input spectrum to a given mag in a given bandpass. This follows eq. (8) in Casagrande et al. (2014, MNRAS, 444, 392).

        Parameters
        ----------
        w: np.ndarray
            Wavelengths of the spectrum in microns.
        f: np.ndarray
            Flux of the spectrum in erg/s/cm2/A.
        bandpass: str
            The name of the bandpass
        mag: np.double
            2MASS J-magnitude to which we wish to re-scale the spectrum.
        Returns
        -------
        np.ndarray
            Rescaled spectrum at wavelength w.
        """
        # Get filter response (note wT is in microns):
        bandpass = Filter(bandpass, wave_units=q.AA)
        wT = bandpass.wave
        TT = bandpass.throughput

        # Get spectrum of vega:
        vega = Vega(wave_units=q.AA)
        w_vega, f_vega = vega.spectrum[:2]

        # Use those two to get the absolute flux calibration for Vega (left-most term in equation (9) in Casagrande et al., 2014).
        vega_weighted_flux = self.spec_integral(w_vega.value, f_vega.value, wT.value, TT)

        # J-band zero-point is thus (maginutde of Vega, m_*, obtained from Table 1 in Casagrande et al, 2014):
        ZP = -0.001 + 2.5 * np.log10(vega_weighted_flux)

        # Now compute (inverse?) bolometric correction for target star. For this, compute same integral as for vega, but for target:
        target_weighted_flux = self.spec_integral(w.to(q.AA).value, f.value, wT.value, TT)

        # Get scaling factor for target spectrum (this ommits any extinction):
        scaling_factor = 10**(-((mag + 2.5 * np.log10(target_weighted_flux) - ZP) / 2.5))

        return f * scaling_factor


def fit_model(row, fitspec):
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
        gstat, yn, xn = list(fitspec.fit(row['spectrum'], wave_units='AA'))
        spectrum = np.array([row['spectrum'][0] * xn, row['spectrum'][1] * yn])
        row['spectrum'] = spectrum
        row['gstat'] = gstat

    except ValueError:
        row['gstat'] = np.nan

    return row
