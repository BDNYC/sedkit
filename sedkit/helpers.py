#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
# !python3
"""
This module is a collection of functions used to homogenize models for
sedkit ingestion
"""

from glob import glob
import os
from pkg_resources import resource_filename

from astropy.io import ascii
import astropy.table as at


def process_dmestar(dir=None, filename='dmestar_solar.txt'):
    """Combine all DMESTAR isochrones into one text file"""
    # Get the filenames
    if dir is None:
        dir = resource_filename('sedkit', 'data/models/evolutionary/DMESTAR/')

    files = glob(os.path.join(dir, '*'))

    # Make a list of the tables
    tables = []
    cols = 'N', 'mass', 'logg', 'log(Teff)', 'Lbol', 'log(R/Ro)'
    for f in files:
        t = ascii.read(f, names=cols)
        t.remove_column('N')
        t['age'] = float(f.split('_')[1][:-3])
        t['teff'] = 10**t['log(Teff)']
        t['radius'] = 10**t['log(R/Ro)']
        t.remove_column('log(Teff)')
        t.remove_column('log(R/Ro)')
        
        tables.append(t)

    if len(tables) > 0:
        table = at.vstack(tables)
        table.meta = None
        path = resource_filename('sedkit', 'data/models/evolutionary/')
        table.write(os.path.join(path, filename), format='csv')

    else:
        print('No DMEStar files to process at {}'.format(dir))