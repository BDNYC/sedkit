"""A suite of tests for the helpers.py module"""
import os
from pkg_resources import resource_filename

from sedkit import helpers as help


def test_process_dmestar():
    """Test the process_dmestar function"""
    # Process DMEStar files
    dir = resource_filename('sedkit', 'data/models/evolutionary/DMESTAR/')
    help.process_dmestar(dir=dir, filename='dmestar_test.txt')

    # Delete temporary file
    path = resource_filename('sedkit', 'data/models/evolutionary/dmestar_test.txt')
    os.system('rm {}'.format(path))