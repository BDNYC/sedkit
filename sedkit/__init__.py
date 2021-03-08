#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
# !python3
"""
Initialize modules

Author: Joe Filippazzo, jfilippazzo@stsci.edu
"""
import re

from .catalog import Catalog
from .sed import SED, VegaSED
from .spectrum import Spectrum, FileSpectrum, Vega, Blackbody
from .modelgrid import ModelGrid, BTSettl, SpexPrismLibrary

__version_commit__ = ''
_regex_git_hash = re.compile(r'.*\+g(\w+)')

__version__ = '1.1.2'

# from pkg_resources import get_distribution, DistributionNotFound
# try:
#     __version__ = get_distribution(__name__).version
# except DistributionNotFound:
#     __version__ = 'dev'

if '+' in __version__:
    commit = _regex_git_hash.match(__version__).groups()
    if commit:
        __version_commit__ = commit[0]
