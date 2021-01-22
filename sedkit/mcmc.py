"""
Module to perform MCMC fitting of a model grid to a spectrum

Code is largely borrowed from https://github.com/BDNYC/synth_fit
"""
from copy import copy

import astropy.units as q
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
import emcee
import numpy as np


def log_probability(model_params, model_grid, spectrum):
    """
    Calculates the probability that the model_params from the given model_grid
    reproduce the spectrum
    
    Parameters
    ----------
    model_params: sequence
        The free parameters for the model
    model_grid: sedkit.modelgrid.ModelGrid
        The model grid to fit
    spectrum: sedkit.spectrum.Spectrum
        The spectrum to fit

    Returns
    -------
    lnprob
        The log of the posterior probability for this model + data
    """
    # The first arguments correspond to the parameters of the model
    # the last, always, corresponds to the tolerance
    # the second to last corresponds to the normalization variance
    model_p = model_params[:model_grid.ndim]
    lns = model_params[-1]
    norm_values = model_params[model_grid.ndim:-1]

    normalization = 1 #self.calc_normalization(norm_values, self.wavelength_bins)

    if (lns > 1.0):
        return -np.inf

    # Check if any of the parameters are outside the limits of the model
    pdict = {}
    for i in range(model_grid.ndim):
        param = model_grid.params[i]
        pdict[param] = model_p[i]
        mx = getattr(model_grid, '{}_max'.format(param))
        mn = getattr(model_grid, '{}_min'.format(param))
        if model_p[i] > mx or model_p[i] < mn:
            return -np.inf

    # Get the model and interpolate to the spectrum
    model_grid.verbose = False
    try:
        model = model_grid.get_spectrum(**pdict, spec_obj=False)
        model_flux = np.interp(spectrum.wave, model[0], model[1])

        s = np.float64(np.exp(lns))
        unc_sq = (spectrum.unc ** 2 + s ** 2) * normalization ** 2
        flux_pts = (spectrum.flux - model_flux * normalization) ** 2 / unc_sq
        width_term = np.log(2 * np.pi * unc_sq)
        lnprob = -0.5 * (np.sum(flux_pts + width_term))

        return lnprob

    except ValueError:
        return -np.inf


class SpecSampler(object):
    """
    Class to contain and run emcee on a spectrum and model grid
    """
    def __init__(self, spectrum, model_grid, params=None, smooth=False, snap=False):
        """
        Parameters 
        ----------
        spectrum: sedkit.spectrum.Spectrum
            The spectrum object to fit
        model_grid: sedkit.modelgrid.ModelGrid
            The model grid to fit
        params: list (optional
            ModelGrid.parameters to vary in fit
        smooth: boolean (default=True)
            whether or not to smooth the model spectra before interpolation 
            onto the data wavelength grid
        """
        # Save attributes
        self.snap = snap
        self.spectrum = spectrum
        self.model_grid = model_grid
        self.model_grid.ndim = self.ndim = len(params)
        self.model_grid.params = params
        if params is None:
            params = self.model_grid.parameters
        self.params = params

        # Calculate starting parameters for the emcee walkers by minimizing
        # chi-squared for the grid of synthetic spectra
        self.spectrum.best_fit_model(self.model_grid, name='best')
        self.start_p = [self.spectrum.best_fit['best'][param] for param in params]
        self.min_chi = self.spectrum.best_fit['best']['gstat']

        # Avoid edges of parameter space
        for i in range(self.model_grid.ndim):
            vals = getattr(self.model_grid, '{}_vals'.format(self.params[i]))
            setattr(self.model_grid, '{}_max'.format(self.params[i]), vals.max())
            setattr(self.model_grid, '{}_min'.format(self.params[i]), vals.min())
            if self.start_p[i] >= vals.max():
                self.start_p[i] = self.start_p[i] * 0.95
            elif self.start_p[i] <= vals.min():
                self.start_p[i] = self.start_p[i] * 1.05

        # Add additional parameters beyond the atmospheric model parameters
        self.all_params = list(np.copy(self.params))

        wavelength_bins = np.array([0.9, 1.4, 1.9, 2.5]) * q.um
        if len(wavelength_bins) > 1:
            norm_number = len(wavelength_bins) - 1
        else:
            norm_number = 1
        for i in range(norm_number):
            self.all_params.append("N{}".format(i))

        # Add normalization parameter
        self.start_p = np.append(self.start_p, np.ones(norm_number))

        # Add (log of) tolerance parameter
        good_unc = [not np.isnan(i) for i in self.spectrum.unc]
        start_lns = np.log(2.0 * np.average(self.spectrum.unc[good_unc]))
        self.start_p = np.append(self.start_p, start_lns)
        self.all_params.append("ln(s)".format(i))

        # The total number of dimensions for the fit is the number of
        # parameters for the model plus any additional parameters added above
        self.ndim = len(self.all_params)

    def mcmc_go(self, nwalk_mult=20, nstep_mult=50):
        """
        Sets up and calls emcee to carry out the MCMC algorithm

        Parameters
        ----------
        nwalk_mult: integer
            Value multiplied by ndim to get the number of walkers
        nstep_mult: integer
            Value multiplied by ndim to get the number of steps
        """
        nwalkers, nsteps = self.ndim * nwalk_mult, self.ndim * nstep_mult

        # Initialize the walkers in a gaussian ball around start_p
        p0 = np.zeros((nwalkers, self.ndim))
        for i in range(nwalkers):
            p0[i] = self.start_p + (1e-2 * np.random.randn(self.ndim) * self.start_p)

        # Set up the sampler
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, log_probability, args=(self.model_grid, self.spectrum))

        # Burn in the walkers
        pos, prob, state = sampler.run_mcmc(p0, nsteps / 10, progress=True)

        # Reset the walkers, so the burn-in steps aren't included in analysis
        sampler.reset()

        # Run MCMC with the walkers starting at the end of the burn-in
        pos, prob, state = sampler.run_mcmc(pos, nsteps, progress=True)

        # Chains contains the positions for each parameter, for each walker
        self.chain = sampler.chain

        # Cut out the burn-in samples (first 10%, for now)
        burn_in = np.floor(nsteps * 0.1)
        self.cropchain = sampler.chain[:, int(burn_in):, :].reshape((-1, self.ndim))

        if self.snap:
            chain_shape = np.shape(self.chain[:, burn_in:, :])
            self.cropchain = self.model.snap_full_run(self.cropchain)
            self.chain = self.cropchain.reshape(chain_shape)

        # Reshape the chains to make one array with all the samples for each parameter
        self.cropchain = sampler.chain.reshape((-1, self.ndim))
        self.get_quantiles()

    # TODO: Convert triangle plot to bokeh
    # def plot_triangle(self, extents=None):
    #     """
    #     Calls triangle module to create a corner-plot of the results
    #     """
    #     self.corner_fig = triangle.corner(self.cropchain, labels=self.all_params, quantiles=[.16, .5, .84], verbose=False, extents=extents)  # , truths=np.ones(3))
    #     plt.suptitle(self.plot_title)

    def plot_chains(self):
        """
        Plot the chains with histograms
        """
        # Get data dimensions
        nwalkers, nsamples, ndim = self.chain.shape

        # For each parameter, I want to plot each walker on one panel, and a histogram of all links from all walkers
        plot_list = []
        for ii in range(ndim):
            walkers = self.chain[:, :, ii]
            flatchain = np.hstack(walkers)

            # Walker plot
            ax1 = figure()
            steps = np.arange(nsamples)
            for walker in walkers:
                ax1.step(steps, walker, color="#555555", alpha=0.5)

            # Create a histogram of all samples. Make 100 bins between the y-axis bounds defined by the 'walkers' plot.
            ax2 = figure()
            hist, edges = np.histogram(flatchain, density=True, bins=50)
            ax2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.5)

            # Add to the plot list
            plot_list.append([ax1, ax2])

        self.chain_fig = gridplot(plot_list)

        show(self.chain_fig)

    def quantile(self, x, quantiles):
        """
        Calculate the quantiles given by quantiles for the array x

        Parameters
        ----------
        x: sequence
            The data array
        quantiles: sequence
            The list of quantiles to compute

        Returns
        -------
        list
            The computed quantiles
        """
        xsorted = sorted(x)
        qvalues = [xsorted[int(q * len(xsorted))] for q in quantiles]
        return list(zip(quantiles, qvalues))

    def get_quantiles(self):
        """
        Calculates (16th, 50th, 84th) quantiles for all parameters
        """
        self.all_quantiles = np.ones((self.ndim, 3)) * -99.
        for i in range(self.ndim):
            quant_array = self.quantile(self.cropchain[:, i], [.16, .5, .84])
            self.all_quantiles[i] = [quant_array[j][1] for j in range(3)]

    def get_error_and_unc(self):
        """
        Calculates 1-sigma uncertainties for all parameters
        """
        self.get_quantiles()

        # The 50th quantile is the mean, the upper and lower "1-sigma"
        # uncertainties are calculated from the 16th- and 84th- quantiles
        # in imitation of Gaussian uncertainties
        self.means = self.all_quantiles[:, 1]
        self.lower_lims = self.all_quantiles[:, 2] - self.all_quantiles[:, 1]
        self.upper_lims = self.all_quantiles[:, 1] - self.all_quantiles[:, 0]

        self.error_and_unc = np.ones((self.ndim, 3)) * -99.
        self.error_and_unc[:, 1] = self.all_quantiles[:, 1]
        self.error_and_unc[:, 0] = (self.all_quantiles[:, 2] - self.all_quantiles[:, 1])
        self.error_and_unc[:, 2] = (self.all_quantiles[:, 1] - self.all_quantiles[:, 0])

        return self.error_and_unc
