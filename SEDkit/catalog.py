#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
A module to produce a catalog of spectral energy distributions
"""
import os
import numpy as np
import pickle
import astropy.table as at
import astropy.units as q
import astropy.constants as ac
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from .sed import SED

class SEDCatalog:
    """An object to collect SED results for plotting and analysis"""
    def __init__(self):
        """Initialize the SEDCatalog object"""
        # List all the results columns
        self.cols = ['name', 'age', 'age_unc', 'distance', 'distance_unc',
                     'parallax', 'parallax_unc', 'radius', 'radius_unc',
                     'spectral_type', 'spectral_type_unc', 'membership',
                     'fbol', 'fbol_unc', 'mbol', 'mbol_unc', 'Lbol', 'Lbol_unc',
                     'Lbol_sun', 'Lbol_sun_unc', 'Mbol', 'Mbol_unc',
                     'logg', 'logg_unc', 'mass', 'mass_unc', 'Teff', 'Teff_unc',
                     'SED']
                
        # A master table of all SED results
        self.results = at.QTable(names=self.cols, dtype=['O']*len(self.cols))
        self.results.add_index('name')
        
        # Set the units
        self.results['age'].unit = q.Myr
        self.results['age_unc'].unit = q.Myr
        self.results['distance'].unit = q.pc
        self.results['distance_unc'].unit = q.pc
        self.results['parallax'].unit = q.mas
        self.results['parallax_unc'].unit = q.mas
        self.results['radius'].unit = ac.R_sun
        self.results['radius'].unit = ac.R_sun
        self.results['fbol'].unit = q.erg/q.s/q.cm**2
        self.results['fbol_unc'].unit = q.erg/q.s/q.cm**2
        self.results['Lbol'].unit = q.erg/q.s
        self.results['Lbol_unc'].unit = q.erg/q.s
        self.results['mass'].unit = q.M_sun
        self.results['mass_unc'].unit = q.M_sun
        self.results['Teff'].unit = q.K
        self.results['Teff_unc'].unit = q.K
        
    
    def add_SED(self, sed):
        """Add an SED to the catalog
        
        Parameters
        ----------
        sed: SEDkit.sed.SED
            The SED object to add
        """
        # Turn off print statements
        sed.verbose = False
        
        # Add the values and uncertainties if applicable
        results = []
        for col in self.cols[:-1]:
            
            if col+'_unc' in self.cols:
                if isinstance(getattr(sed, col), tuple):
                    val = getattr(sed, col)[0]
                else:
                    val = None
            elif col.endswith('_unc'):
                if isinstance(getattr(sed, col.replace('_unc', '')), tuple):
                    val = getattr(sed, col.replace('_unc', ''))[1]
                else:
                    val = None
            else:
                val = getattr(sed, col)
        
            val = val.to(self.results[col.replace('_unc','')].unit).value if hasattr(val, 'unit') else val
            
            results.append(val)
            
        # Add the SED
        results.append(sed)
        
        # Make the table
        cat = SEDCatalog()
        table = cat.results
        table.add_row(results)
        table = at.Table(table)
        
        # Append apparent and absolute photometry
        for row in sed.photometry:

            # Add the apparent magnitude
            table.add_column(at.Column([row['app_magnitude']], name=row['band']))

            # Add the apparent uncertainty
            table.add_column(at.Column([row['app_magnitude_unc']], name=row['band']+'_unc'))

            # Add the absolute magnitude
            table.add_column(at.Column([row['abs_magnitude']], name='M_'+row['band']))

            # Add the absolute uncertainty
            table.add_column(at.Column([row['abs_magnitude_unc']], name='M_'+row['band']+'_unc'))
        
        # Stack with current table
        if len(self.results)==0:
            self.results = table
        else:
            self.results = at.vstack([self.results,table])
        
        
    def get_SED(self, name):
        """Retrieve the SED for the given object"""
        # Add the index
        self.results.add_index('name')
        
        # Get the rows
        try:
            return self.results.loc[name]['SED']
            
        except IOError:
            print('No SEDs named',name)
            
            return
        
        
    def filter(self, param, value):
        """Retrieve the filtered rows
        
        Parameters
        ----------
        param: str
            The parameter to filter by, e.g. 'Teff'
        value: str, float, int, sequence
            The criteria to filter by, 
            which can be single valued like 1400
            or a range with operators [<,<=,>,>=],
            e.g. (>1200,<1400), ()
        
        Returns
        -------
        SEDkit.sed.SEDCatalog
            The filtered catalog
        """
        # Make a new catalog
        cat = SEDCatalog()
        
        # Wildcard case
        if isinstance(value, str) and '*' in value:
            
            # Strip souble quotes
            value = value.replace("'",'').replace('"','')
            
            # Split the wildcard
            start, end = value.split('*')
            
            # Get indexes
            data = np.array(self.results[param])
            idx = np.where([np.logical_and(i.startswith(start),i.endswith(end)) for i in data])
            
            # Filter results
            cat.results = self.results[idx]
            
        else:
            
            # Make single value string into conditions
            if isinstance(value, str):
                
                # Check for operator
                if any([value.startswith(o) for o in ['<','>','=']]):
                    value = [value]
                    
                # Assume eqality if no operator
                else:
                    value = ['=='+value]
                
            # Turn numbers into strings
            if isinstance(value, (int,float)):
                value = ["=={}".format(value)]
            
            # Iterate through multiple conditions
            for cond in value:
                
                # Equality
                if cond.startswith('='):
                    v = cond.replace('=','')
                    cat.results = self.results[self.results[param]==eval(v)]
                
                # Less than or equal
                elif cond.startswith('<='):
                    v = cond.replace('<=','')
                    cat.results = self.results[self.results[param]<=eval(v)]
                
                # Less than
                elif cond.startswith('<'):
                    v = cond.replace('<','')
                    cat.results = self.results[self.results[param]<eval(v)]
            
                # Greater than or equal
                elif cond.startswith('>='):
                    v = cond.replace('>=','')
                    cat.results = self.results[self.results[param]>=eval(v)]
                
                # Greater than
                elif cond.startswith('>'):
                    v = cond.replace('>','')
                    cat.results = self.results[self.results[param]>eval(v)]
                
                else:
                    raise ValueError("'{}' operator not understood.".format(cond))
        
        return cat
        
        
    def from_file(self, coords):
        """Generate a catalog from a list of coordinates
        
        Parameters
        ----------
        coords: str
            The path to the two-column file of ra, dec values
        """
        data = np.genfromtxt(coords)
        
        for ra, dec, *k in data:

            # Make the SED
            s = SED()
            s.verbose = False
            s.sky_coords = ra*q.deg, dec*q.deg
            s.find_Gaia()
            s.find_2MASS()
            s.find_WISE()
            s.make_sed()

            # Add it to the catalog
            self.add_SED(s)
        
        
    def load(self, file):
        """Load a saved SEDCatalog"""
        if os.path.isfile(file):
            
            f = open(file)
            cat = pickle.load(f)
            f.close()
            
            f = open(file, 'rb')
            cat = pickle.load(f)
            f.close()

            self.results = cat
        
    
    def plot(self, x, y, scale=['linear','linear'], fig=None,
             xlabel=None, ylabel=None):
        """Plot parameter x versus parameter y
        
        Parameters
        ----------
        x: str
            The name of the x axis parameter, e.g. 'SpT'
        y: str
            The name of the y axis parameter, e.g. 'Teff'
        """
        # Make the figure
        if fig is None:
            # Make the plot
            TOOLS = ['pan', 'resize', 'reset', 'box_zoom', 'save']
            title = '{} v {}'.format(x,y)
            fig = figure(plot_width=800, plot_height=500, title=title, 
                         y_axis_type=scale[1], x_axis_type=scale[0], 
                         tools=TOOLS)
                         
        # Get the source names
        names = self.results['name'] 
                        
        # Get the x data
        if '-' in x:
            x1, x2 = x.split('-')
            if self.results[x1].unit!=self.results[x2].unit:
                raise TypeError('x-axis columns must be the same units.')
            
            xunit = self.results[x1].unit
            xdata = self.results[x1]-self.results[x2]
        else:
            xunit = self.results[x].unit
            xdata = self.results[x]
            
        # Get the y data
        if '-' in y:
            y1, y2 = y.split('-')
            if self.results[y1].unit!=self.results[y2].unit:
                raise TypeError('y-axis columns must be the same units.')
                
            yunit = self.results[y1].unit
            ydata = self.results[y1]-self.results[y2]
        else:
            yunit = self.results[y].unit
            ydata = self.results[y]
            
        # Set axis labels
        fig.xaxis.axis_label = '{}{}'.format(x, ' [{}]'.format(xunit) if xunit else '')
        fig.yaxis.axis_label = '{}{}'.format(y, ' [{}]'.format(yunit) if yunit else '')
        
        # Set up hover tool
        tips = [( 'Name', '@desc'), (x, '@x'), (y, '@y')]
        hover = HoverTool(tooltips=tips)
        fig.add_tools(hover)

        # Plot points with tips
        source = ColumnDataSource(data=dict(x=xdata, y=ydata, desc=names))
        fig.circle('x', 'y', source=source, legend='Photometry', name='photometry', fill_alpha=0.7, size=8)
        
        # Formatting
        fig.legend.location = "top_right"
        fig.legend.click_policy = "hide"
        
        return fig
        
        
    def remove_SED(self, idx_or_name):
        """Remove an SED from the catalog
        
        Parameters
        ----------
        idx_or_name: str, int
            The name or index of the SED to remove
        """
        pass
        
        
    def save(self, file):
        """Save the serialized data
        
        Parameters
        ----------
        file: str
            The filepath
        """
        path = os.path.dirname(file)
        
        if os.path.exists(path):
            
            # Make the file if necessary
            if not os.path.isfile(file):
                os.system('touch {}'.format(file))
                
            # Write the file
            f = open(file, 'wb')
            pickle.dump(self.results, f, pickle.HIGHEST_PROTOCOL)
            f.close()
            
            print('SEDCatalog saved to',file)