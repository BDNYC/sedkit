#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
Interface with astroquery to fetch data
"""
import os
import time
from urllib.request import urlretrieve

from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
import astropy.units as q
from astroquery.vizier import Vizier
from astroquery.sdss import SDSS
import numpy as np


# A list of photometry catalogs from Vizier
PHOT_CATALOGS = {'2MASS': {'catalog': 'II/246/out', 'cols': ['Jmag', 'Hmag', 'Kmag'], 'names': ['2MASS.J', '2MASS.H', '2MASS.Ks']},
            'WISE': {'catalog': 'II/328/allwise', 'cols': ['W1mag', 'W2mag', 'W3mag', 'W4mag'], 'names': ['WISE.W1', 'WISE.W2', 'WISE.W3', 'WISE.W4']},
            'PanSTARRS': {'catalog': 'II/349/ps1', 'cols': ['gmag', 'rmag', 'imag', 'zmag', 'ymag'], 'names': ['PS1.g', 'PS1.r', 'PS1.i', 'PS1.z', 'PS1.y']},
            'Gaia': {'catalog': 'I/345/gaia2', 'cols': ['Plx', 'Gmag', 'Teff', 'Lum'], 'names': ['parallax', 'Gaia.G', 'teff', 'Lbol']},
            'SDSS': {'catalog': 'V/147', 'cols': ['umag', 'gmag', 'rmag', 'imag', 'zmag'], 'names': ['SDSS.u', 'SDSS.g', 'SDSS.r', 'SDSS.i', 'SDSS.z']}}

Vizier.columns = ["**", "+_r"]


def query_SDSS_optical_spectra(coords, idx=0, verbose=True, **kwargs):
    """
    Query for SDSS spectra

    Parameters
    ----------
    coords: astropy.coordinates.SkyCoord
        The coordinates to query
    idx: int
        The index of the target to use from the results table
    verbose: bool
        Print messages

    Returns
    -------
    list
        The [W, F, E] spectrum of the target
    """

    # Fetch results
    results = SDSS.query_region(coords, spectro=True, **kwargs)
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


def query_SDSS_apogee_spectra(coords, verbose=True, **kwargs):
    """
    Query the APOGEE survey data

    Parameters
    ----------
    coords: astropy.coordinates.SkyCoord
        The coordinates to query

    Returns
    -------
    list
        The [W, F, E] spectrum of the target
    """

    # Query vizier for spectra
    catalog = 'III/284/allstars'
    results = query_vizier(catalog, col_names=['Ascap', 'File', 'Tel', 'Field'], sky_coords=coords, wildcards=[], cat_name='APOGEE', verbose=verbose, **kwargs)

    if len(results) == 0:

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


def query_vizier(catalog, target=None, sky_coords=None, col_names=None, wildcards=['e_*'], target_names=None, search_radius=20 * q.arcsec, idx=0, cat_name=None, verbose=True, **kwargs):
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
    """
    # Get the catalog
    if catalog in PHOT_CATALOGS:
        meta = PHOT_CATALOGS[catalog]
        catalog = meta['catalog']
        cols = col_names or meta['cols']
        names = target_names or meta['names']
    else:
        cols = col_names
        names = target_names

    # Name for the catalog
    cat_name = cat_name or catalog

    try:

        # Get photometry using designation...
        if isinstance(target, str):

            try:
                viz_cat = Vizier.query_object(target, catalog=[catalog])
            except Exception as exc:
                print("[sedkit] {}".format(exc))
                print("[sedkit] Trying query again...")
                time.sleep(10)
                viz_cat = Vizier.query_object(target, catalog=[catalog])

        # ...or use coordinates...
        elif search_radius is not None and isinstance(sky_coords, SkyCoord):

            try:
                viz_cat = Vizier.query_region(sky_coords, radius=search_radius, catalog=[catalog])
            except Exception as exc:
                print("[sedkit] {}".format(exc))
                print("[sedkit] Trying again...")
                time.sleep(10)
                viz_cat = Vizier.query_region(sky_coords, radius=search_radius, catalog=[catalog])

        # ...or abort
        else:
            viz_cat = []

    except:

        viz_cat = []

    # Check there are columns to fetch
    if cols is None:
        cols = viz_cat[0].colnames

    # Check for wildcards
    wildcards = wildcards or []

    # Check for target names or just use native column names
    names = names or cols

    # Print info
    if verbose:
        n_rec = len(viz_cat)
        print("[sedkit] {} record{} found in {}.".format(n_rec, '' if n_rec == 1 else 's', cat_name))

    # Parse the record
    results = []
    if len(viz_cat) > 0:
        if len(viz_cat) > 1:
            print('[sedkit] {} {} records found.'.format(len(viz_cat), name))

        # Grab the record
        rec = dict(viz_cat[0][idx])
        ref = viz_cat[0].meta['name']

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
                print("[sedkit] Could not find '{}' column in '{}' catalog.".format(col, catalog))

            # Check for wildcards
            for wc in wildcards:
                wc_col = wc.replace('*', col)
                val = rec.get(wc_col)
                data.append(val)

            # Add reference
            data.append(ref)
            results.append(data)

    return results
