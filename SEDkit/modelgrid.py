"""
A module to generate a grid of model spectra

Author: Joe Filippazzo, jfilippazzo@stsci.edu
"""
import os
import glob
import numpy as np
import astropy.io.ascii as ii
import astropy.table as at
import astropy.units as q
import astropy.io.votable as vo
from pkg_resources import resource_filename
from . import utilities as u
from .spectrum import FileSpectrum

# A list of all supported evolutionary models
EVO_MODELS = [os.path.basename(m).replace('.txt', '') for m in glob.glob(resource_filename('SEDkit', 'data/models/evolutionary/*'))]


class ModelGrid:
    """A class to store a model grid"""
    def __init__(self, name, wave_units=None, flux_units=None):
        """Initialize the model grid from a directory of VO table files
        
        Parameters
        ----------
        name: str
            The name of the model grid
        wave_units: astropy.units.quantity.Quantity
            The wavelength units
        flux_units: astropy.units.quantity.Quantity
            The flux units
        """
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
        self.wave_units = wave_units
        self.flux_units = flux_units
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
                                    
                meta = {}
                for p in all_params:

                    # Extract the key/value pairs
                    key = p[1].split('"')[1]
                    val = p[-1].split('"')[1]
                    
                    if (parameters and key in parameters) or not parameters:

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
        
        # Update attributes
        if parameters is None:
            parameters = [col for col in index.colnames if col != 'filepath']
        self.parameters = parameters
        self.index = index

    def get_models(self, **kwargs):
        """Retrieve all models with the specified parameters
        
        Returns
        -------
        list
            A list of the spectra as SEDkit.spectrum.Spectrum objects
        """
        # Get the relevant table rows
        table = u.filter_table(self.index, **kwargs) 
        
        # Collect the spectra
        spectra = []
        for row in table:
            spec = FileSpectrum(row['filepath'], wave_units=self.wave_units,
                                flux_units=self.flux_units)
            for col in table.colnames:
                setattr(spec, col, row[col])
                
            spectra.append(spec)
        
        return spectra
        

class BTSettl(ModelGrid):
    """Child class for the BT-Settl model grid"""
    def __init__(self):
        """Loat the model object"""
        # Inherit from base class
        super().__init__('btsettl', q.AA, q.erg/q.s/q.cm**2/q.AA)
        
        