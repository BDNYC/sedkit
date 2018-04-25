import pysynphot as ps
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
import glob
import astropy.table as at
import astropy.io.ascii as asc
from pkg_resources import resource_filename

# Area of the telescope has to be in centimeters2
ps.setref(area=250000.)

FILT_PATH = resource_filename('SEDkit', 'data/filters/')

# class bandpass(ps.ArrayBandpass):
#     """
#     A class that inherits from pysynphot and adds uncertainty handling
#     """
#     def __init__(self, band):
#         pass
        

def bandpass(filt, inst, filt_dir=FILT_PATH):
    """
    Creates a pysynphot bandpass with the given filter
    
    Parameters
    ----------
    filt: str
        The filter name
    inst: str
        The instrument name
    filt_dir: str
        The directory of filters to look in
    
    Returns
    -------
    pysynphot.spectrum.ArraySpectralElement
        The bandpass object
    """
    # Get the filename
    filename = 'JWST_{}.{}.dat.txt'.format(inst.upper(),filt.upper())
    
    # Read the file
    wave, thru = np.genfromtxt(os.path.join(filt_dir,filename), unpack=True)
    
    # Create the bandpass
    bandpass = ps.ArrayBandpass(wave=wave, throughput=thru, waveunits='angstrom', name=filt)
    
    return bandpass

def synthetic_magnitude(spectrum, bandpass, plot=False):
    """
    Calculate the magnitude in a bandpass
    
    Parameters
    ----------
    spectrum: pysynphot.spectrum.ArraySpectralElement
        The spectrum to process
    bandpass: pysynphot.spectrum.ArraySpectralElement
        The bandpass to use
    plot: bool
        Plot the original and processed spectrum
    
    Returns
    -------
    float
        The magnitude
    """
    # Calculate flux in band
    star = ps.Observation(spectrum, bandpass, binset=bandpass.wave)
    
    # Calculate zeropoint flux in band
    vega = ps.Observation(ps.Vega, bandpass, binset=bandpass.wave)
    
    # Calculate the magnitude
    mag = -2.5*np.log10(star.integrate()/vega.integrate())
    
    if plot:
        plt.plot(spectrum.wave, spectrum.flux)
        plt.plot(star.wave, star.flux)
    
    return round(mag, 3)

def mag_table(spectra=None, bandpasses=FILT_PATH, models='phoenix', jmag=10, save=None):
    """
    Calculate the magnitude of all given spectra in all given bandpasses
    
    Parameters
    ----------
    spectra: sequence
        A sequence of [Teff, FeH, logg] values 
    bandpasses: sequence
        A list of bandpass objects
    models: str
        The model grid to use
    jmag: float
        The J magnitude to renormalize to
    save: str
        The file to save the results to
    """
    # Get the J bandpass
    jband = ps.ObsBandpass('j')
    
    # Make the list of spectra
    if spectra==None:
        teff_range = np.arange(2000, 2550, 50)
        feh_range = np.arange(-0.5, 1.0, 0.5)
        logg_range = np.arange(4.5, 5.5, 0.5)
        ranges = [teff_range, feh_range, logg_range]
        spectra = list(itertools.product(*ranges))
        
    # Make the list of bandpasses if given a directory
    if isinstance(bandpasses, str) and os.path.exists(bandpasses):
        
        # Get the files
        files = glob.glob(os.path.join(bandpasses,'*'))
        bandpasses = [(i.split('.')[-3],i.split('.')[-4].split('_')[-1]) for i in files]
        
    # Make the list of bandpasse
    if isinstance(bandpasses, (list,tuple)):
        bandpasses = [bandpass(filt, inst) for filt, inst in bandpasses]
        
    else:
        print("Please provide a list of (filter,instrument) tuples or a directory of filters.")
        return
    
    # An empty list of tables
    tables = []

    print("Calculating synthetic mags for...")
    
    # For each set of params...
    for n, (teff, feh, logg) in enumerate(spectra):
        
        # ...get the spectrum...
        spectrum = ps.Icat(models, teff, feh, logg)
        
        # Renormalize the spectrum
        try:
            spectrum = spectrum.renorm(jmag, 'vegamag', jband)
            print((teff, feh, logg))
            
        except:
            print('Error:',(teff, feh, logg))
            continue
        
        # Make the table for this spectrum
        table = at.Table([[teff], [feh], [logg]], names=('teff','feh','logg'))
        
        # ...and calculate the magnitude...
        for bp in bandpasses:
            
            # ...in each bandpass...
            mag = synthetic_magnitude(spectrum, bp)
            
            # ...and add the mag to the list
            table[bp.name] = [mag]
            
        tables.append(table)
        
    # Stack all the tables
    mag_table = at.vstack(tables)
    
    # Save to file
    if os.path.exists(os.path.dirname(save)) and '.' in save:
        
        if not save.endswith('.csv'):
            save = save.split('.')[0]+'.csv'
            
        mag_table.write(save, format='ascii.csv', overwrite=True)
        
    # Or return
    else:
        return mag_table
    
def color_color_plot(colorx, colory, table, **kwargs):
    """
    Make a color-color plot for the two bands
    
    Parameters
    ----------
    colorx: str
        Two bandpass names delimited with a '-' sign for the x axis, e.g. 'F115W-F356W'
    colory: str
        Two bandpass names delimited with a '-' sign for the y axis, e.g. 'F430M-F480M'
    table: str, astropy.table.Table
        An astropy table or path to a CSV file of magnitudes
    """
    # Get teh table of data
    if os.path.isfile(table):
        table = asc.read(table)
    
    # Get the bands to retrieve
    bandx1, bandx2 = colorx.split('-')
    bandy1, bandy2 = colory.split('-')
    
    # Make a new table with the calculated colors
    table[colorx] = table[bandx1]-table[bandx2]
    table[colory] = table[bandy1]-table[bandy2]
    
    # Filter by parameter
    for param in ['teff', 'logg', 'feh']:
        if isinstance(kwargs.get(param), (int, float)):
            table = table[table[param]==kwargs[param]]
    
    # Plot it
    markers = ['o','s','d','x','v']
    for i,g in enumerate(np.unique(table['logg'])):
        tab = table[table['logg']==g]
        plt.scatter(tab[colorx], tab[colory], c=tab['teff'], marker=markers[i%len(markers)], label='logg = {}'.format(g))
        
    plt.colorbar()
    plt.legend(loc=0)
