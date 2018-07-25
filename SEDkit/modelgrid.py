"""
A module to generate a grid of model spectra

Author: Joe Filippazzo, jfilippazzo@stsci.edu
"""
import os
import glob
import numpy as np
import astropy.io.ascii as ii
import astropy.table as at
import astropy.io.votable as vo
from pkg_resources import resource_filename

# A list of all supported evolutionary models
EVO_MODELS = [os.path.basename(m).replace('.txt', '') for m in glob.glob(resource_filename('SEDkit', 'data/models/evolutionary/*'))]


class ModelGrid:
    """A class to store a model grid"""
    def __init__(self, name):
        """Initialize the model grid from a directory of VOT files
        
        Parameters
        ----------
        name: str
            The name of the model grid"""
        # Store the path and name
        self.name = name

        # Make the path
        model_path = 'data/models/atmospheric/{}'.format(name)
        root = resource_filename('SEDkit', model_path)
        if not os.path.exists(root):
            raise IOError(root, ": No such directory")

        # See if there is a table of parameters
        self.path = root
        self.index_path = os.path.join(root, 'index.txt')
        if not os.path.isfile(self.index_path):
            os.system("touch {}".format(self.index_path))

            # Index the models
            self.index_models()

        # Load the index
        self.index = ii.read(self.index_path)

        # Store the parameters
        self.parameters = [col for col in self.index.colnames if col != 'filepath']

    def index_models(self, parameters=None):
        """Generate model index file for faster reading
        
        Parameters
        ----------
        parameters: sequence
            The names of the parameters from the VOT files to index
        """
        # Get the files
        files = glob.glob(os.path.join(self.path, '*.xml'))
        self.n_models = len(files)
        print("Indexing {} models for {} grid...".format(self.n_models, self.name))

        # Grab the parameters and the filepath for each
        all_meta = []
        for file in files:

            try:
                # Parse the XML file
                vot = vo.parse_single_table(file)

                # Parse the SVO filter metadata
                all_params = [str(p).split() for p in vot.params]
                params = parameters or all_params
                                    
                meta = {}
                for p in params:
                    
                    # Extract the key/value pairs
                    key = p[1].split('"')[1]
                    val = p[-1].split('"')[1]

                    # Do some formatting
                    if p[2].split('"')[1] == 'float' or p[3].split('"')[1] == 'float':
                        val = float(val)

                    else:
                        val = val.replace('b&apos;','').replace('&apos','').replace('&amp;','&').strip(';')

                    # Add it to the dictionary
                    meta[key] = val

                # Add the filename
                meta['filepath'] = file

                all_meta.append(meta)
                
            except IOError:
                print(file, ": Could not parse file")

        # Make the index table
        index = at.Table(all_meta)
        index.write(self.index_path, format='ascii.tab', overwrite=True)

    # def get_model(self, **kwargs):
    #     """Retrieve the model with the specified parameters"""
        

class BTSettl(ModelGrid):
    """Child class for the BT-Settl model grid"""
    def __init__(self):
        """Loat the model object"""
        # Inherit from base class
        super().__init__('btsettl')
        
        # Specifiy which parameters to use
        self.index_models(['teff', 'logg',' meta', 'alpha'])
        
        