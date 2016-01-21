import syn_phot, logging, cPickle, synth_fit, itertools, utilities as u, astropy.units as q, numpy as np, matplotlib.pyplot as plt, pandas as pd
from astrodbkit.astrodbkit import astrodb

def pd_interp_models(params, coordinates, model_grid, smoothing=1):
  """
  Interpolation code that accepts a model grid and a list of parameters/values to return an interpolated spectrum.

  Parameters
  ----------
  params: list
      A list of the model parameters, e.g. ['teff', 'logg', 'f_sed']
  coordinates: list
      A list of the coordinates in parameter space to evaluate, e.g. [1643, 5.1, 2.3]
  model_grid: Pandas DataFrame
      A Pandas dataframe of the database
  
  Returns
  -------
  spectrum: list of arrays
       The wavelength and flux at the specified **values** in parameter space

  Notes
  -----
  You might have to update scipy and some other things to run this. Do:
  >>> conda install scipy

  """
  from scipy.interpolate import LinearNDInterpolator
  
  # Make sure the model grid is a DataFrame
  if not isinstance(model_grid, pd.DataFrame):
    model_grid['flux'] = model_grid['flux'].value
    model_grid['wavelength'] = [model_grid['wavelength'].value]*len(model_grid['flux'])
    model_grid = pd.DataFrame(model_grid)
  
  # Transpose the Series of flux arrays into a Series of element-wise arrays of the flux at each wavelength point
  flux_columns = pd.Series(np.asarray(model_grid['flux'].tolist()).T.tolist())
  
  # Take out nusiance parameters and build parameter space from arrays
  grid = np.asarray(model_grid.loc[:,params])
  
  # Define the wavelength array and an empty flux array
  W, F = model_grid['wavelength'][0], np.zeros(len(model_grid['wavelength'][0]))
  
  # Interpolate to specified coordinates for each wavelength point in parameter space
  for n in range(len(F)):
    
    # Create grid interpolation function to pass coordinates to
    interpND = LinearNDInterpolator(grid, flux_columns[n], rescale=True)
  
    # Find flux value at desired coordinates in parameter space and insert into interpolated flux array
    F[n] = interpND(coordinates)
  
  return [W,u.smooth(F,smoothing) if smoothing else F]  
  

def make_model_db(model_grid_name, model_atmosphere_db, grid_data='spec', param_lims=[('teff',400,700,50),('logg',3.5,5.5,0.5)], fill_holes=True, bands=[], rebin_models=True, use_pandas=False):
  '''
  Given a **model_grid_name**, returns the grid from the model_atmospheres.db as a Pandas DataFrame
  
  Parameters
  ----------
  model_grid_name: str
    The name of the model grid table in the model_atmospheres.db SQL file, e.g. 'bt_settl_2013'
  model_atmosphere_db: str
    The path to model_atmospheres.db
  grid_data: 'spec' or 'phot'
    Returns a grid of spectra or synthetic photometry
  param_lims: list of tuples (optional)
    A list of tuples with the parameter name, lower limit, upper limit, and increment for each parameter to be constrained, e.g. [('teff',400,800,100),('logg',4,5,0.5)]
  rebin_models: array or bool
    The wavelength array to which all model spectra should be rebinned OR True if random rebinning is desired
  
  Returns
  -------
  models: Pandas DataFrame
    The resulting model grid as a Pandas DataFrame
  
  '''
  # Load the model_atmospheres database and pull all the data from the specified table
  db = astrodb.get_db(model_atmosphere_db)
  if param_lims:
    limit_text = ' AND ' .join([l[0]+' IN ('+','.join(map(str,np.arange(l[1],l[2]+l[3],l[3])))+')' for l in param_lims])
    model_grid = db.dict("SELECT * FROM {} WHERE {}".format(model_grid_name,limit_text)).fetchall()  
  else: model_grid = db.dict("SELECT * FROM {}".format(model_grid_name)).fetchall()
    
  # Load the model atmospheres into a data frame and define the parameters
  models = pd.DataFrame(model_grid)
  params = [p for p in models.columns.values.tolist() if p in ['teff','logg','f_sed','k_zz']]
  
  # Get the uppler bound, lower bound, and increment of the parameters
  plims = {p[0]:p[1:] for p in param_lims} if param_lims else {}
  for p in params:
    if p not in plims: 
      plims[p] = (min(models.loc[:,p]),max(models.loc[:,p]),max(np.diff(np.unique(np.asarray(models.loc[:,p]))), key=list(np.diff(np.unique(np.asarray(models.loc[:,p])))).count))
  
  # Choose template wavelength array to rebin all other spectra
  W = rebin_models if isinstance(rebin_models,(list,np.ndarray)) else models['wavelength'][0]

  
  # Rebin model spectra or calculate synthetic magnitudes
  if grid_data=='phot':
    model_phot = []
    for w,f in zip(list(models['wavelength']),list(models['flux'])):
      model_phot.append(syn_phot.all_mags([w*q.um, f*q.erg/q.s/q.cm**2/q.AA], bands=bands, Flam=False, to_flux=True, photon=False, to_list=True))

    pF, pW = map(list,zip(*[map(list,zip(*[[i.value if hasattr(i,'unit') else i for i in j][::2] for j in k])) for k in model_phot]))
    models['wavelength'], models['flux'] = pd.Series([np.array(i) for i in pW]), pd.Series([np.array(i) for i in pF])

  else:
    models['flux'] = pd.Series([u.rebin_spec([w*q.um, f*q.erg/q.s/q.cm**2/q.AA], W*q.um)[1].value for w,f in zip(list(models['wavelength']),list(models['flux']))])
    models['wavelength'] = pd.Series([W]*len(models['flux']))
  
  if fill_holes:
    # Get the coordinates in parameter space of each existing grid point
    coords = models.loc[:,params].values
  
    # Get the coordinates in parameter space of each desired grid point
    template = np.asarray(list(itertools.product(*[np.arange(l[0],l[1]+l[2],l[2]) for p,l in plims.items()])))

    # Find the holes in the grid based on the defined grid resolution without expanding the grid borders
    def find_holes(coords, template=''):
      # Make a grid of all the parameters
      coords = np.asanyarray(coords)
      uniq, labels = zip(*[np.unique(c, return_inverse=True) for c in coords.T])
      grid = np.zeros(map(len, uniq), bool)
      # if template!='':
      #   temp = np.asanyarray(template)
      #   uniqT, labelsT = zip(*[np.unique(c, return_inverse=True) for c in temp.T])
      #   gridT = np.zeros(map(len, uniqT), bool)
      grid[labels] = True
      candidates = np.zeros_like(grid)

      # Test if there are neighboring models for interpolation
      for dim in range(grid.ndim):
        grid0 = np.rollaxis(grid, dim)
        inside = np.logical_or.accumulate(grid0, axis=0) & np.logical_or.accumulate(grid0[::-1], axis=0)[::-1]
        candidates |= np.rollaxis(inside, 0, dim+1)
      holes = candidates & ~grid
      hole_labels = np.where(holes)
      return np.column_stack([u[h] for u, h in zip(uniq, hole_labels)])
  
    grid_holes = find_holes(coords, template=template)
  
    # Interpolate the grid to fill in the holes
    for h in grid_holes:
      print 'Filling grid hole at {}'.format(h)
      new_spectrum = pd_interp_models(params, h, models, smoothing=False)
      new_row = {k:v for k,v in zip(params,h)}
      new_row.update({'wavelength':new_spectrum[0], 'flux':new_spectrum[1], 'comments':'interpolated'})
      new_row.update({'wavelength':new_spectrum[0], 'flux':new_spectrum[1], 'comments':'interpolated', 'metallicity':0, 'id':None})
      models = models.append(new_row, ignore_index=True)
    
  # Sort the DataFrame by teff and logg?
  models.sort(list(reversed(params)), inplace=True)

  # Turn Pandas DataFrame into a dictionary of arrays if not using Pandas
  if not use_pandas:
    M = {k:models[k].values for k in models.columns.values}    
    M['flux'] = q.erg/q.AA/q.cm**2/q.s*np.asarray(M['flux'])
    M['wavelength'] = q.um*M['wavelength'][0]
    return M

  else: return models

# ===========================================================================================================================================
# ===================================== Non-Pandas ==========================================================================================
# ===========================================================================================================================================


def fit_spectrum(raw_spectrum, model_grid, walkers, steps, mask=[], db='', object_name='Test', log=False, plot=True, outfile=None):
  '''
  Given **raw_spectrum** as an integer id from the SPECTRUM table or a [W,F,E] list with astropy units, 
  returns a marginalized distribution plot of best fit parameters from the specified **model_grid** name.
  
  Parameters
  ----------
  raw_spectrum: sequence, dict
    A dictionary or [w,f,e] sequence of astropy quantity arrays to be fit
  model_grid: str
    The name of the model grid to be used in the fit, e.g. 'bt_settl_2013'
  walkers: int
    The number of walkers to deploy
  steps: int
    The number of steps for each walker to take
  mask: sequence (optional)
    Tuples of wavelength ranges to exclude in the model fits, e.g. mask=[(1.12,1.16),(1.35,1.42)] for J-H-K water absorption bands
  db: instance
    The pre-loaded astrodbkit.astrodb.get_db() database instance to pull the spectrum from
  
  Returns
  --------
  bdsamp: object
    The MCMC result instance
  '''
  if log: logging.basicConfig(level=logging.DEBUG)
  
  # Input [W,F,E] or spectrum dictionary
  spectrum = raw_spectrum if isinstance(raw_spectrum,dict) else {'wavelength':raw_spectrum[0], 'flux':raw_spectrum[1], 'unc':raw_spectrum[2]}
  
  # Apply mask to flux and unc arrays to exclude those regions from the MCMC fit
  for m in mask: spectrum['flux'], spectrum['unc'] = [np.ma.masked_where(np.logical_and(spectrum['wavelength']>m[0]*q.um,spectrum['wavelength']<m[-1]*q.um), spectrum[arr])*spectrum[arr].unit for arr in ['flux','unc']]
  
  # Specify the parameter space to be walked
  params = [i for i in model_grid.keys() if i in ['logg', 'teff', 'f_sed', 'k_zz']]
  
  # Set up the sampler object (it's a wrapper around emcee)
  bdsamp = synth_fit.bdfit.BDSampler(object_name, spectrum, model_grid, params, smooth=False, plot_title="{}, {}".format(object_name,"BT-Settl 2013"), snap=False) # smooth=False if model already matches data, snap=True if no interpolation is needed on grid
  
  # Run the mcmc method
  bdsamp.mcmc_go(nwalk_mult=walkers, nstep_mult=steps, outfile=outfile)
  
  # Plotting
  if plot: bdsamp.plot_triangle(), bdsamp.plot_chains()
  
  # Printing
  if log: 
    logging.info("ran MCMC")
    logging.info("all done!")
    
  # Generate best fit spectrum the 50th quantile value
  PD = {k:v for k,v in zip(bdsamp.all_params,bdsamp.all_quantiles.T[1])}
  bdsamp.best_fit_spectrum = pd_interp_models(params, [PD[p] for p in params], model_grid)
  
  return bdsamp

def interp_models(params, coordinates, model_grid, smoothing=1):
  """
  Interpolation code that accepts a model grid and a list of parameters/values to return an interpolated spectrum.

  Parameters
  ----------
  params: list
      A list of the model parameters, e.g. ['teff', 'logg', 'f_sed']
  coordinates: list
      A list of the coordinates in parameter space to evaluate, e.g. [1643, 5.1, 2.3]
  model_grid: object
      The output of make_model_db()
  
  Returns
  -------
  spectrum: list of arrays
       The wavelength and flux at the specified **values** in parameter space

  Notes
  -----
  You might have to update scipy and some other things to run this. Do:
  >>> conda update scipy 
  
  or... 
  
  >>> ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
  >>> brew install gcc
  >>> pip install scipy --upgrade

  """
  from scipy.interpolate import LinearNDInterpolator
  
  # Transpose the flux arrays to interpolate the grid at each wavelength position
  flux_columns = [np.array([i[n] for i in model_grid['flux'].value]) for n in range(len(model_grid['flux'].value[0]))]
  
  # Take out nusiance parameters and build parameter space from arrays
  params, coordinates = zip(*[i for i in zip(params,coordinates) if i[0] in ['teff','logg','k_zz','f_sed']])
  grid = [model_grid.get(p) for p in params]
  
  # Define the wavelength array and an empty flux array
  W, F = model_grid['wavelength'].value, np.zeros(len(flux_columns))
  
  # Interpolate to specified coordinates for each wavelength point in parameter space
  for n in range(len(flux_columns)):
    
    # Create grid interpolation function to pass coordinates to
    interpND = LinearNDInterpolator(zip(*grid), flux_columns[n], rescale=True)
  
    # Find flux value at desired coordinates in parameter space and insert into interpolated flux array
    F[n] = interpND(coordinates)
  
  return [W,u.smooth(F,smoothing) if smoothing else F]

# ============================================================================================================================================
# ====================================================== TESTS ===============================================================================
# ============================================================================================================================================

def model_grid_smoothness_test(models, param_lims={'teff':(0,800), 'logg':(4.0,6.0)}, rebin_models=True):
  '''
  Given a **model_grid_name**, returns the grid from the model_atmospheres.db in the proper format to work with fit_spectrum()
  '''
  for g in list(set(models['logg'])):
    plt.figure()
    upper, lower = max(models[models['logg']==g]['teff']), min(models[models['logg']==g]['teff'])
    cont = plt.contourf([[0,0],[0,0]], range(lower,upper+50,50), cmap=plt.cm.jet_r)
    plt.clf()
    for n,(t,f) in enumerate(zip(models[models['logg']==g]['teff'],models[models['logg']==g]['flux'])):
      color = plt.cm.jet_r((1.*t-lower)/(upper-lower),1.)
      plt.loglog(W, f, color=color)
      plt.loglog(*interp_models(['teff','logg'], [t+25,g], models, smoothing=1), color=color, alpha=0.5)
      plt.xlim(0.8,2.5), plt.ylim(1E-3,1E4)
    plt.title('log(g) = {}'.format(g)), plt.colorbar(cont)

def model_grid_interp_test(model_grid, teff, logg):
  '''
  Perform a tet of the model grid interpolation by specifying the teff, logg and grid resolution of the given model grid.
  
  Parameters
  ----------
  model_grid: dict
    The model grid object that results from running make_model_db()
  teff: tuple, list
    A sequence of the teff value and increment over which to interpolate, e.g. (1200,100) tests the model at 1200K by interpolating between the 1100K and 1300K models
  logg: tuple, list
    A sequence of the logg value and increment over which to interpolate, e.g. (4.5,0.5) tests the model at 4.5dex by interpolating between the 4.0dex and 5.0dex models

  Returns
  -------
  None
  '''
  # Get the upper, lower and target teff and logg values
  (t1,g1), (t2,g2), (t3,g3) = [(teff[0]+(teff[1]*i),logg[0]+(logg[1]*i)) for i in [-1.,0.,1.]]
  
  # Find the indexes of the models with the appropriate parameters
  idx1, idx2, idx3 = [zip(model_grid['teff'],model_grid['logg']).index((t,g)) for t,g in [(t1,g1), (t2,g2), (t3,g3)]]
  
  # Pull out the wavelength and flux values of the appropriate models
  (w1,f1), (w2,f2), (w3,f3) = [(model_grid['wavelength'][i],model_grid['flux'][i]) for i in [idx1,idx2,idx3]]
  
  # Pop the true model and interpolate the model grid to see how they compare
  mg = model_grid.copy()
  mg['teff'], mg['logg'], mg['wavelength'], mg['flux'] = [np.delete(mg[p], idx2, 0) for p in ['teff','logg','wavelength','flux']]
  W, F = interp_models(['teff','logg'], [t2,g2], mg)
  
  # Plot the interpolated model and the upper, lower, anf true models
  plt.loglog(W, F, color='k', lw=2, label='Interpolated')
  for w,f,c,t,g in [[w1,f1,'r',t1,g1],[w2,f2,'g',t2,g2],[w3,f3,'b',t3,g3]]:
    f = u.smooth(f, 1)
    plt.loglog(w.value, f.value, color=c, label='{} {}'.format(t,g), alpha=0.5)
  plt.legend(loc=0, fontsize=14, frameon=False)
