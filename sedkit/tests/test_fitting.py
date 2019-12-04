# import unittest
# import copy
# from pkg_resources import resource_filename
#
# import numpy as np
# import astropy.units as q
# from astropy.modeling.blackbody import blackbody_lambda
#
# from .. import fitting
# from .. import spectrum
# from .. import modelgrid
#
#
# def test_lmfit_modelgrid():
#     """Test for lmfit_modelgrid function"""
#     # Make a spectrum
#     spec = spectrum.Vega()
#
#     # Make a model grid
#     models = modelgrid.SpexPrismLibrary()
#
#     # Find the best fit
#     fit = fitting.lmfit_modelgrid(spec, models)
#
#     # Positive tests
#     assert fit.name == 'Best Fit'
