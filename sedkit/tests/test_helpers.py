"""A suite of tests for the helpers.py module"""
import os
import importlib.resources

from sedkit import helpers as help


def test_process_dmestar():
    """Test the process_dmestar function"""
    # Process DMEStar files
    dir = str(importlib.resources.files('sedkit')/ 'data/models/evolutionary/DMESTAR/')
    help.process_dmestar(dir=dir, filename='dmestar_test.txt')

    # Delete temporary file
    temp_file = str(importlib.resources.files('sedkit')/ 'data/models/evolutionary/dmestar_test.txt')
    os.remove(temp_file)