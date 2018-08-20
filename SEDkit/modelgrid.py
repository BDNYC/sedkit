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
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

# A list of all supported evolutionary models
EVO_MODELS = [os.path.basename(m).replace('.txt', '') for m in glob.glob(resource_filename('SEDkit', 'data/models/evolutionary/*'))]


class ModelGrid:
    """A class to store a model grid"""
    def __init__(self, name, wave_units=None, flux_units=None, resolution=200):
        """Initialize the model grid from a directory of VO table files
        
        Parameters
        ----------
        name: str
            The name of the model grid
        wave_units: astropy.units.quantity.Quantity
            The wavelength units
        flux_units: astropy.units.quantity.Quantity
            The flux units
        resolution: float
            The resolution of the models
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
        
        # Store the parameter ranges
        for param in self.parameters:
            setattr(self, '{}_vals'.format(param), np.asarray(np.unique(self.index[param])))

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
        pool = ThreadPool(8) 
        func = partial(FileSpectrum, wave_units=self.wave_units,
                       flux_units=self.flux_units)
        spectra = pool.map(func, table['filepath'])
        pool.close()
        pool.join()
        
        # Add the metadata
        for n, (row, spec) in enumerate(zip(table, spectra)):
            
            for col in table.colnames:
                setattr(spectra[n], col, row[col])
        
        if len(spectra) == 1:
            spectra = spectra.pop()
        
        return spectra
        
    def get(self, resolution=None, interp=True, **kwargs):
        """
        Retrieve the wavelength, flux, and effective radius
        for the spectrum of the given parameters

        Parameters
        ----------
        resolution: int (optional)
            The desired wavelength resolution (lambda/d_lambda)
        interp: bool
            Interpolate the model if possible

        Returns
        -------
        SEDkit.spectrum.Spectrum, list
            A Spectrum object or list of Spectrum objects
        """
        # See if the model with the desired parameters is witin the grid
        in_grid = []
        for param, value in kwargs.items():
            
            # Get the value range
            vals = getattr(self, param+'_vals')
            if min(vals) <= value <= max(vals):
                in_grid.append(True)
                
            else:
                in_grid.append(False)

        if all(in_grid):

            # See if the model with the desired parameters is a true grid point
            on_grid = []
            for param, value in kwargs.items():
            
                # Get the value range
                vals = getattr(self, param+'_vals')
                if value in vals:
                    on_grid.append(True)
                
                else:
                    on_grid.append(False)

            # Grab the data if the point is on the grid
            if all(on_grid):
                return self.get_models(**kwargs)

            # If not on the grid, interpolate to it
            else:
                # Call grid_interp method
                if interp:
                    spec_dict = self.grid_interp(**kwargs)
                else:
                    return

        else:
            param_str = ['{}={}'.format(k, v) for k, v in kwargs.items()]
            print(', '.join(param_str)+' model not in grid.')
            return

    def grid_interp(self, **kwargs):
        """
        Interpolate the grid to the desired parameters

        Returns
        -------
        SEDkit.spectrum.Spectrum
            The interpolated Spectrum object
        """
        return True
        # # Load the fluxes
        # if isinstance(self.flux, str):
        #     self.load_flux()
        #
        # # Get the flux array
        # flux = self.flux.copy()
        #
        # # Get the interpolable parameters
        # params, values = [], []
        # for p, v in zip([self.Teff_vals, self.logg_vals, self.FeH_vals],
        #                 [Teff, logg, FeH]):
        #     if len(p) > 1:
        #         params.append(p)
        #         values.append(v)
        # values = np.asarray(values)
        # label = '{}/{}/{}'.format(Teff, logg, FeH)
        #
        # try:
        #     # Interpolate flux values at each wavelength
        #     # using a pool for multiple processes
        #     print('Interpolating grid point [{}]...'.format(label))
        #     processes = 4
        #     mu_index = range(flux.shape[-2])
        #     start = time.time()
        #     pool = multiprocessing.Pool(processes)
        #     func = partial(utils.interp_flux, flux=flux, params=params,
        #                    values=values)
        #     new_flux, generators = zip(*pool.map(func, mu_index))
        #     pool.close()
        #     pool.join()
        #
        #     # Clean up and time of execution
        #     new_flux = np.asarray(new_flux)
        #     generators = np.asarray(generators)
        #     print('Run time in seconds: ', time.time()-start)
        #
        #     # Interpolate mu value
        #     interp_mu = RegularGridInterpolator(params, self.mu)
        #     mu = interp_mu(np.array(values)).squeeze()
        #
        #     # Interpolate r_eff value
        #     interp_r = RegularGridInterpolator(params, self.r_eff)
        #     r_eff = interp_r(np.array(values)).squeeze()
        #
        #     # Make a dictionary to return
        #     grid_point = {'Teff': Teff, 'logg': logg, 'FeH': FeH,
        #                   'mu': mu, 'r_eff': r_eff,
        #                   'flux': new_flux, 'wave': self.wavelength,
        #                   'generators': generators}
        #
        #     return grid_point
        #
        # except IOError:
        #     print('Grid too sparse. Could not interpolate.')
        #     return
        
    def load_spectra(self, reload=False):
        """
        Retrieve the spectra for all models and load into the
        ModelGrid.array attribute
        
        Parameters
        ----------
        reload: bool
            Relod the data even if it already exists
        """
        if reload:

            # Delete the old file and clear the flux attribute
            self.flux = None
            self.wave = None

        if self.flux is None:

            print('Loading flux into table...')

            if os.path.isfile(self.flux_file):

                # Load the flux from the HDF5 file
                f = h5py.File(self.flux_file, "r")
                self.flux = f['flux'][:]
                f.close()

            else:

                # Get array dimensions
                T, G, M = self.Teff_vals, self.logg_vals, self.FeH_vals
                shp = [len(T), len(G), len(M)]
                n, N = 1, np.prod(shp)

                # Iterate through rows
                for nt, teff in enumerate(T):
                    for ng, logg in enumerate(G):
                        for nm, feh in enumerate(M):

                            try:

                                # Retrieve flux using the `get()` method
                                d = self.get(teff, logg, feh, interp=False)

                                if d:

                                    # Make sure arrays exist
                                    if isinstance(self.flux, str):
                                        new_shp = shp+list(d['flux'].shape)
                                        self.flux = np.zeros(new_shp)
                                    if isinstance(self.r_eff, str):
                                        self.r_eff = np.zeros(shp)
                                    if isinstance(self.mu, str):
                                        new_shp = shp+list(d['mu'].shape)
                                        self.mu = np.zeros(new_shp)

                                    # Add data to respective arrays
                                    self.flux[nt, ng, nm] = d['flux']
                                    if d['r_eff'] is None:
                                        self.r_eff[nt, ng, nm] = np.nan
                                    else:
                                        self.r_eff[nt, ng, nm] = d['r_eff']
                                    self.mu[nt, ng, nm] = d['mu'].squeeze()

                                    # Get the wavelength array
                                    if isinstance(self.wavelength, str):
                                        self.wavelength = d['wave']

                                    # Garbage collection
                                    del d

                                    # Print update
                                    n += 1
                                    msg = "{: .2f}% complete.".format(n*100./N)
                                    print(msg, end='\r')

                            except IOError:
                                # No model computed so reduce total
                                N -= 1

                # Load the flux into an HDF5 file
                f = h5py.File(self.flux_file, "w")
                # dset = f.create_dataset('flux', data=self.flux)
                f.create_dataset('flux', data=self.flux)
                f.close()
                # del dset
                print("100.00 percent complete!", end='\n')

        else:
            print('Data already loaded.')
    
        

class BTSettl(ModelGrid):
    """Child class for the BT-Settl model grid"""
    def __init__(self):
        """Loat the model object"""
        # Inherit from base class
        super().__init__('btsettl', q.AA, q.erg/q.s/q.cm**2/q.AA)
        
        