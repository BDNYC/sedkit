#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
Interface with astroquery to fetch data
"""
from astropy.coordinates import Angle, SkyCoord
import astropy.units as q
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from . import utilities as u


# A list of photometry catalogs from Vizier
PHOT_CATALOGS = {'2MASS': {'catalog': 'II/246/out', 'cols': ['Jmag', 'Hmag', 'Kmag'], 'names': ['2MASS.J', '2MASS.H', '2MASS.Ks']},
            'WISE': {'catalog': 'II/328/allwise', 'cols': ['W1mag', 'W2mag', 'W3mag', 'W4mag'], 'names': ['WISE.W1', 'WISE.W2', 'WISE.W3', 'WISE.W4']},
            'PanSTARRS': {'catalog': 'II/349/ps1', 'cols': ['gmag', 'rmag', 'imag', 'zmag', 'ymag'], 'names': ['PS1.g', 'PS1.r', 'PS1.i', 'PS1.z', 'PS1.y']},
            'Gaia': {'catalog': 'I/345/gaia2', 'cols': ['Gmag'], 'names': ['Gaia.G']},
            'SDSS': {'catalog': 'V/147', 'cols': ['umag', 'gmag', 'rmag', 'imag', 'zmag'], 'names': ['SDSS.u', 'SDSS.g', 'SDSS.r', 'SDSS.i', 'SDSS.z']}}

Vizier.columns = ["**", "+_r"]


def query_vizier(catalog, target=None, sky_coords=None, cols=None, wildcards=['e_*'], names=None, search_radius=20*q.arcsec, idx=0, places=3, cat_name=None, verbose=True, **kwargs):
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
    cols: sequence
        The list of column names to fetch
    wildcards: sequence
        A list of wildcards for each column name, e.g. 'e_*' includes errors
    target_names: sequence (optional)
        The list of renamed columns, must be the same length as band_names
    search_radius: astropy.units.quantity.Quantity
        The search radius for the Vizier query
    idx: int
        The index of the record to use if multiple Vizier results
    """
    # Get the catalog
    if catalog in PHOT_CATALOGS:
        meta = PHOT_CATALOGS[catalog]
        catalog = meta['catalog']
        cols = cols or meta['cols']
        names = names or meta['names']

    # Name for the catalog
    if cat_name is None:
        cat_name = catalog

    # If search_radius is explicitly set, use that
    if search_radius is not None and isinstance(sky_coords, SkyCoord):
        viz_cat = Vizier.query_region(sky_coords, radius=search_radius, catalog=[catalog])

    # ...or get photometry using designation...
    elif isinstance(target, str):
        viz_cat = Vizier.query_object(target, catalog=[catalog])

    # ...or abort
    else:
        viz_cat = None

    # Check there are columns to fetch
    if cols is None:
        raise ValueError("No column names to fetch!")

    # Check for wildcards
    if wildcards is None:
        wildcards = []

    # Check for target names or just use native column names
    if names is None:
        names = cols

    # Print info
    if verbose:
        n_rec = len(viz_cat)
        print("{} record{} found in {}.".format(n_rec, '' if n_rec == 1 else 's', cat_name))

    results = []

    # Parse the record
    if viz_cat is not None and len(viz_cat) > 0:
        if len(viz_cat) > 1:
            print('{} {} records found.'.format(len(viz_cat), name))

        # Grab the record
        rec = viz_cat[0][idx]
        ref = viz_cat[0].meta['name']

        # Pull out the photometry
        for name, viz in zip(names, cols):
            fetch = [viz]+[wc.replace('*', viz) for wc in wildcards]
            if all([i in rec.columns for i in fetch]):
                data = [round(val, places) if u.isnumber(val) else val for val in rec[fetch]]
                results.append([name]+data+[ref])
            else:
                print("{}: Could not find all those columns".format(fetch))

    return results
