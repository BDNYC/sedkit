"""A suite of tests for the query.py module"""

import astropy.units as q
from astropy.coordinates import SkyCoord

from .. import query


def test_query_vizier():
    """Test for the query_vizier function"""
    # 2MASS catalog
    catalog = '2MASS'
    sky_coords = SkyCoord(ra=1.23, dec=2.34, unit=(q.degree, q.degree), frame='icrs')

    # Query target
    results = query.query_vizier(catalog, target='Vega', search_radius=20 * q.arcsec, verbose=True)
    assert len(results) > 0

    # Query coords
    results = query.query_vizier(catalog, sky_coords=sky_coords, search_radius=20 * q.arcmin, verbose=True)
    assert len(results) > 0

    # No results
    results = query.query_vizier(catalog, sky_coords=sky_coords, search_radius=0.1 * q.arcsec, verbose=True)
    assert len(results) == 0


def test_query_SDSS_optical_spectra():
    """Test for the query_SDSS_optical_spectra function"""
    # Some results
    sky_coords = SkyCoord('0h8m05.63s +14d50m23.3s', frame='icrs')
    results = query.query_SDSS_optical_spectra(sky_coords, radius=20 * q.arcsec)
    assert len(results) > 0

    # No results
    sky_coords = SkyCoord(ra=1.23, dec=2.34, unit=(q.degree, q.degree), frame='icrs')
    results = query.query_SDSS_optical_spectra(sky_coords, radius=0.1 * q.arcsec)
    assert len(results) > 0


def test_query_SDSS_apogee_spectra():
    """Test for the query_SDSS_apogee_spectra function"""
    sky_coords = SkyCoord(ra=1.23, dec=2.34, unit=(q.degree, q.degree), frame='icrs')

    # Some results
    results = query.query_SDSS_apogee_spectra(sky_coords, search_radius=10 * q.degree)
    assert len(results) > 0

    # No results
    results = query.query_SDSS_apogee_spectra(sky_coords, search_radius=0.1 * q.arcsec)
    assert len(results) > 0
