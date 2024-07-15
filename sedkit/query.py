#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
Interface with astroquery to fetch data
"""
import os
import time
from urllib.request import urlopen
from bs4 import BeautifulSoup

from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
import astropy.units as q
from astroquery.vizier import Vizier
from astroquery.sdss import SDSS
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource
import numpy as np
from svo_filters import Filter

from . import utilities as u


# A list of photometry catalogs from Vizier
PHOT_CATALOGS = {'2MASS': {'catalog': 'II/246/out', 'cols': ['Jmag', 'Hmag', 'Kmag'], 'names': ['2MASS.J', '2MASS.H', '2MASS.Ks']},
            'WISE': {'catalog': 'II/328/allwise', 'cols': ['W1mag', 'W2mag', 'W3mag', 'W4mag'], 'names': ['WISE.W1', 'WISE.W2', 'WISE.W3', 'WISE.W4']},
            'PanSTARRS': {'catalog': 'II/349/ps1', 'cols': ['gmag', 'rmag', 'imag', 'zmag', 'ymag'], 'names': ['PS1.g', 'PS1.r', 'PS1.i', 'PS1.z', 'PS1.y']},
            'Gaia': {'catalog': 'I/345/gaia2', 'cols': ['Plx', 'Gmag', 'BPmag', 'RPmag', 'Teff', 'Lum'], 'names': ['parallax', 'Gaia.G', 'Gaia.bp', 'Gaia.rp', 'teff', 'Lbol']},
            'SDSS': {'catalog': 'V/147', 'cols': ['umag', 'gmag', 'rmag', 'imag', 'zmag'], 'names': ['SDSS.u', 'SDSS.g', 'SDSS.r', 'SDSS.i', 'SDSS.z']}}

Vizier.columns = ["**", "+_r"]


def query_spectra(catalog, target=None, sky_coords=None, subdir='sp', filecol='FileName', wave_units=q.AA, flux_units=q.erg/q.s/q.cm**2/q.AA, trim_blue=5, trim_red=5, **kwargs):
    """
    Search for spectra from a generic Vizier catalog

    Spectrum data storage may vary by catalog but this method works for catalogs
    that store the spectral data in a subdirectory, usually labeled "sp"

    Parameters
    ----------
    catalog: str
        The Vizier catalog, i.e. 'J/MNRAS/371/703/catalog'
    target: str
        The target name to query
    sky_coords: astropy.coordinates.SkyCoord
        The coordinates to query
    subdir: str
        The name of the subdirectory containing the spectra
    filecol: str
        The name of the column containing the spectra filenames
    trim_blue: int
        The number of data points to trim from the blue end of the spectrum
    trim_red: int
        The number of data points to trim from the red end of the spectrum

    Returns
    -------
    list
        The [W, F, E] spectrum of the target
    """

    # Query vizier for spectra
    results = query_vizier(catalog, target=target, sky_coords=sky_coords, col_names=[filecol], **kwargs)

    if len(results) == 0:

        return None

    else:

        # Construct data URL
        file = results[0][1]
        catdir = '%2f'.join(catalog.split('/')[:-1])
        url = 'http://cdsarc.u-strasbg.fr/viz-bin/nph-Plot/Vgraph:fits2a/txt?{}%2f.%2f{}%2f{}'.format(catdir, subdir, file)

        # Web scrape the data
        html = urlopen(url)
        soup = BeautifulSoup(html)
        rows = soup.get_text().split('\n')
        data = np.array([row.split() for row in rows[trim_blue:-trim_red]]).astype(float).T

        if len(data) == 3:
            wav, flx, err = data
        elif len(data) == 2:
            wav, flx = data
            err = None
        else:
            raise ValueError("Sorry but I don't understand the format of this data.")

        # Add units
        wav *= wave_units
        flx *= flux_units
        if err is not None:
            err *= flux_units

        return [wav, flx, err]


def query_SDSS_optical_spectra(target=None, sky_coords=None, idx=0, verbose=True, **kwargs):
    """
    Query for SDSS spectra

    Parameters
    ----------
    target: str
        The target name to query
    sky_coords: astropy.coordinates.SkyCoord
        The coordinates to query
    idx: int
        The index of the target to use from the results table
    verbose: bool
        Print statements

    Returns
    -------
    list
        The [W, F, E] spectrum of the target
    """

    # Fetch results
    try:
        results = SDSS.query_region(sky_coords, spectro=True, **kwargs)
    except:
        results = None
    n_rec = 0 if results is None else len(results)

    # Print info
    if verbose:
        print("[sedkit] {} record{} found in SDSS optical data.".format(n_rec, '' if n_rec == 1 else 's'))

    if n_rec == 0:

        return None, None, None

    else:

        # Download the spectrum file
        hdu = SDSS.get_spectra(matches=results)[idx]

        # Get the spectrum data
        data = hdu[1].data

        # Convert from log to linear units in Angstroms
        wav = 10**data['loglam'] * q.AA

        # Convert to FLAM units
        flx = data['flux'] * 1E-17 * q.erg / q.s / q.cm**2 / q.AA
        err = data['flux'] * 1E-18 * q.erg / q.s / q.cm**2 / q.AA  # TODO: Where's the error?

        # Metadata
        ref = 'SDSS'
        header = hdu[0].header

        return [wav, flx, err], ref, header


def query_SDSS_apogee_spectra(target=None, sky_coords=None, verbose=True, **kwargs):
    """
    Query the APOGEE survey data

    Parameters
    ----------
    target: str
        The target name to query
    sky_coords: astropy.coordinates.SkyCoord
        The coordinates to query
    verbose: bool
        Print statements

    Returns
    -------
    list
        The [W, F, E] spectrum of the target
    """

    # Query vizier for spectra
    catalog = 'III/284/allstars'
    results = query_vizier(catalog, target=target, sky_coords=sky_coords, col_names=['Ascap', 'File', 'Tel', 'Field'], wildcards=[], cat_name='APOGEE', verbose=verbose, **kwargs)
    n_rec = 0 if results is None else len(results)

    if n_rec == 0:

        return None, None, None

    else:

        ascap, file, tel, field = [row[1] for row in results]

        # Construct URL
        url = 'https://data.sdss.org/sas/dr16/apogee/spectro/redux/r12/stars/{}/{}/{}'.format(tel, field, file)

        # Download the file
        urlretrieve(url, file)

        # Get data
        hdu = fits.open(file)
        header = hdu[0].header

        # Generate wavelength
        wav = 10**(np.linspace(header['CRVAL1'], header['CRVAL1'] + (header['CDELT1'] * header['NWAVE']), header['NWAVE']))
        wav *= q.AA

        # Get flux and error
        flx = hdu[1].data[0] * 1E-17 * q.erg / q.s / q.cm**2 / q.AA
        err = hdu[2].data[0] * 1E-17 * q.erg / q.s / q.cm**2 / q.AA

        # Delete file
        hdu.close()
        os.system('rm {}'.format(file))

    return [wav, flx, err], catalog, header


# def query_IRAS_spectra(coords, verbose=True, **kwargs):
#     """
#     Query the IRAS survey data
#
#     Parameters
#     ----------
#     coords: astropy.coordinates.SkyCoord
#         The coordinates to query
#
#     Returns
#     -------
#     list
#         The [W, F, E] spectrum of the target
#     """
#
#     # Query vizier for spectra
#     catalog = 'III/197/lrs'
#     results = query_vizier(catalog, sky_coords=coords, wildcards=[], verbose=verbose)
#     # results = query_vizier(catalog, col_names=['Ascap', 'File', 'Tel', 'Field'], sky_coords=coords, wildcards=[], verbose=verbose)
#
#     if len(results) == 0:
#
#         return None, None, None
#
#     else:
#
#         file = [row[1] for row in results]
#
#         # Construct URL
#         url = 'https://cdsarc.unistra.fr/ftp/III/197/{}'.format(file)
#
#         # Download the file
#         urlretrieve(url, file)
#
#         # Get data
#         hdu = fits.open(file)
#         header = hdu[0].header
#
#         # Generate wavelength
#         wav = 10**(np.linspace(header['CRVAL1'], header['CRVAL1'] + (header['CDELT1'] * header['NWAVE']), header['NWAVE']))
#         wav *= q.AA
#
#         # Get flux and error
#         flx = hdu[1].data[0] * 1E-17 * q.erg / q.s / q.cm**2 / q.AA
#         err = hdu[2].data[0] * 1E-17 * q.erg / q.s / q.cm**2 / q.AA
#
#         # Delete file
#         hdu.close()
#         os.system('rm {}'.format(file))
#
#     return [wav, flx, err], catalog, header


def query_vizier(catalog, target=None, sky_coords=None, col_names=None, wildcards=['e_*'], target_names=None, search_radius=20 * q.arcsec, idx=0, cat_name=None, verbose=True, preview=False, **kwargs):
    """
    Search Vizier for photometry in the given catalog

    Parameters
    ----------
    catalog: str
        The Vizier catalog name or address, e.g. '2MASS' or 'II/246/out'
    target: str (optional)
        A target name to search for, e.g. 'Trappist-1'
    sky_coords: astropy.coordinates.SkyCoord (optional)
        The sky coordinates to search
    col_names: sequence
        The list of column names to fetch
    wildcards: sequence
        A list of wildcards for each column name, e.g. 'e_*' includes errors
    target_names: sequence (optional)
        The list of renamed columns, must be the same length as col_names
    search_radius: astropy.units.quantity.Quantity
        The search radius for the Vizier query
    idx: int
        The index of the record to use if multiple Vizier results
    preview: bool
        Make a plot of all photometry results
    """
    # Get the catalog
    if catalog in PHOT_CATALOGS:
        cat_name = cat_name or catalog
        meta = PHOT_CATALOGS[catalog]
        catalog = meta['catalog']
        cols = col_names or meta['cols']
        names = target_names or meta['names']
    else:
        cat_name = cat_name or catalog
        cols = col_names
        names = target_names

    # Name for the catalog
    n_rec = 0

    try:

        # Get photometry using designation...
        if isinstance(target, str):

            try:
                viz_cat = Vizier.query_object(target, catalog=[catalog], **kwargs)
            except Exception as exc:
                print("[sedkit] {}".format(exc))
                print("[sedkit] Trying query again...")
                time.sleep(10)
                viz_cat = Vizier.query_object(target, catalog=[catalog], **kwargs)

            if len(viz_cat) > 0:
                viz_cat = viz_cat[0] if len(viz_cat) > 0 else []

            # Print info
            n_rec = len(viz_cat)
            if verbose:
                print("[sedkit] {} record{} found in '{}' using target name '{}'".format(n_rec, '' if n_rec == 1 else 's', cat_name, target))

        # ...or use coordinates...
        if n_rec == 0 and search_radius is not None and isinstance(sky_coords, SkyCoord):

            try:
                viz_cat = Vizier.query_region(sky_coords, radius=search_radius, catalog=[catalog], **kwargs)
            except Exception as exc:
                print("[sedkit] {}".format(exc))
                print("[sedkit] Trying again...")
                time.sleep(10)
                viz_cat = Vizier.query_region(sky_coords, radius=search_radius, catalog=[catalog], **kwargs)

            if len(viz_cat) > 0:
                viz_cat = viz_cat[0] if len(viz_cat) > 0 else []

            # Print info
            n_rec = len(viz_cat)
            if verbose:
                print("[sedkit] {} record{} found in '{}' using {} radius around {}".format(n_rec, '' if n_rec == 1 else 's', cat_name, search_radius, sky_coords))

        # ...or abort
        if n_rec == 0:
            viz_cat = []

    except IOError:

        viz_cat = []

    # Check there are columns to fetch
    if cols is None:
        cols = viz_cat.colnames

    # Check for wildcards
    wildcards = wildcards or []

    # Check for target names or just use native column names
    names = names or cols

    # Parse the record
    results = []
    if n_rec > 0:

        if preview:

            # Make the figure
            prev = figure(width=900, height=400, y_axis_type="log", x_axis_type="log")
            filters = [Filter(name) for name in names]
            colors = u.color_gen(kwargs.get('palette', 'viridis'), n=n_rec)
            phot_tips = [('Band', '@desc'), ('Wave', '@wav'), ('Flux', '@flx'), ('Unc', '@unc'), ('Idx', '@idx')]
            hover = HoverTool(names=['phot'], tooltips=phot_tips)
            prev.add_tools(hover)

            def valid(flx, err):
                return err > 0 and not np.isnan(flx) and not np.isnan(err)

            # Get all mags from the queried results table
            for n, row in enumerate(viz_cat):
                try:
                    color = next(colors)
                    mags = [row[col] for col in cols if valid(row[col], row['e_{}'.format(col)])]
                    uncs = [row['e_{}'.format(col)] for col in cols if valid(row[col], row['e_{}'.format(col)])]
                    flxs, uncs = np.array([u.mag2flux(filt, m, e) for filt, m, e in zip(filters, mags, uncs)]).T
                    source = ColumnDataSource(data=dict(wav=[filt.wave_eff.value for filt in filters], flx=flxs, unc=uncs, idx=[n] * len(flxs), desc=names))
                    prev.line('wav', 'flx', source=source, color=color, name='phot', alpha=0.1, hover_color="firebrick", hover_alpha=1)
                    prev.circle('wav', 'flx', source=source, color=color, size=8, alpha=0.3, hover_color="firebrick", hover_alpha=1)
                    prev = u.errorbars(prev, 'wav', 'flx', yerr='unc', source=source, color=color, alpha=0.3, hover_color="firebrick", hover_alpha=1)
                except ValueError:
                    pass

            return prev

        else:

            # Grab the record
            rec = dict(viz_cat[idx])
            ref = viz_cat.meta['name']

            # Pull out the photometry
            for name, col in zip(names, cols):

                # Data for this column
                data = []

                # Add name
                data.append(name)

                # Check for nominal value
                nom = rec.get(col)
                data.append(nom)
                if nom is None:
                    print("[sedkit] Could not find '{}' column in '{}' catalog.".format(col, cat_name))

                # Check for wildcards
                for wc in wildcards:
                    wc_col = wc.replace('*', col)
                    val = rec.get(wc_col)
                    data.append(val)

                # Add reference
                data.append(ref)
                results.append(data)

    return results
