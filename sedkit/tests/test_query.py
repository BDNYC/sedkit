"""A suite of tests for the query.py module"""

import astropy.units as q
from astropy.coordinates import SkyCoord

from .. import query


def test_query_vizier():
    """Test for equivalent function"""
    # 2MASS catalog
    catalog = 'II/246/out'
    cols = ['Jmag', 'Hmag', 'Kmag']
    names = ['2MASS.J', '2MASS.H', '2MASS.Ks']
    cat = '2MASS'

    # Query target
    results = query.query_vizier(catalog, target='Vega', cols=cols, wildcards=['e_*'], names=names, search_radius=20 * q.arcsec, idx=0, places=3, cat_name=cat, verbose=True)
    assert len(results) > 0

    # Query coords
    sky_coords = SkyCoord(ra=1.23, dec=2.34, unit=(q.degree, q.degree), frame='icrs')
    results = query.query_vizier(catalog, sky_coords=sky_coords, cols=cols, wildcards=['e_*'], names=None, search_radius=20 * q.arcmin, idx=0, places=3, cat_name=cat, verbose=True)
    assert len(results) > 0

    # No results
    results = query.query_vizier(catalog, sky_coords=sky_coords, cols=cols, wildcards=['e_*'], names=None, search_radius=0.1 * q.arcsec, idx=0, places=3, cat_name=cat, verbose=True)
    assert len(results) == 0

