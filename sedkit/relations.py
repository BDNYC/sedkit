"""
This is the code used to generate the polynomial relations
used in sedkit's calculations
"""
from pkg_resources import resource_filename

import astropy.io.ascii as ii
import astropy.units as q
import astropy.table as at
from astroquery.vizier import Vizier
from bokeh.plotting import figure, show
import numpy as np

from . import utilities as u


V = Vizier(columns=["**"])


class SpectralTypeRadius:
    def __init__(self, order=8, name='Spectral Type vs. Radius'):
        """Initialize the object

        Parameters
        ----------
        order: int
            The order polynomial to fit to the spt-radius data
        """
        self.name = name
        self.generate(order)

    def get_radius(self, spt, plot=False):
        """Get the radius for the given spectral type

        Parameters
        ----------
        spt: str, int
            The alphanumeric (e.g. 'A0') or integer (0-99 => O0-Y9) spectral
            type
        plot: bool
            Generate a plots

        Returns
        -------
        tuple
            The radius and uncertainty in solar radii
        """
        # Test valid range
        if not 0 <= spt <= 99:
            raise TypeError("Please provide a spectral type within [0, 99]")

        # Evaluate the polynomials
        radius = np.polyval(self.coeffs, spt)*q.Rsun
        radius_unc = np.interp(spt, self.spt, self.sig_yi)*q.Rsun

        if plot:
            fig = self.plot()
            fig.circle([spt], [radius.value], color='red', size=15)
            show(fig)

        return radius, radius_unc

    def generate(self, order):
        """Generate a polynomial that describes the radius as a function of
        spectral type for empirically measured AFGKM main sequence stars
        (Boyajian+ 2012b, 2013) and MLTY model isochrone interpolated stars
        (Filippazzoet al. 2015, 2016)

        Parameters
        ----------
        order: int
            The polynomial order to fit to the data
        generate: bool
            Generate the polynomial
        """
        # ====================================================================
        # Boyajian AFGKM data
        # ====================================================================

        afgk = resource_filename('sedkit', 'data/AFGK_radii.txt')
        self.AFGK = ii.read(afgk, format='csv', comment='#')
        self.AFGK['name'] = 'AFGK'

        # ====================================================================
        # Filippazzo MLTY data
        # ====================================================================

        # Get the data
        cat1 = V.query_constraints('J/ApJ/810/158/table1')[0]
        cat2 = V.query_constraints('J/ApJ/810/158/table9')[0]

        # Join the tables to getthe spectral types and radii in one table
        MLTY = at.join(cat1, cat2, keys='ID', join_type='outer')

        # Rename columns
        MLTY.rename_column('SpT', 'spectral_type')
        MLTY.rename_column('Rad', 'radius')
        MLTY.rename_column('e_Rad', 'radius_unc')

        # Make solar radii units
        MLTY['radius'] = MLTY['radius'].to(q.Rsun)
        MLTY['radius_unc'] = MLTY['radius_unc'].to(q.Rsun)
        MLTY['name'] = 'MLTY'
        self.MLTY = MLTY

        # ====================================================================
        # Combine and fit the data
        # ====================================================================

        # Join the two tables
        data = at.vstack([self.AFGK, self.MLTY], join_type='inner')

        # Translate string SPT to numbers
        spts = []
        keep = []
        for n,i in enumerate(data['spectral_type']):
            try:
                spt = u.specType(i)
                spts.append(spt)
                keep.append(n)
            except:
                pass

        # Filter bad spectral types
        data = data[keep]

        # Add the number to the table
        num, *_, lum = np.array(spts).T
        data['spt'] = num.astype(float)
        data['lum'] = lum

        # Filter out sub-giants
        data = data[data['lum'] == 'V']

        # Filter out nans
        data = data[data['radius'] < 4]
        data = data[data['radius'] > 0]
        data = data[data['radius_unc'] > 0]
        self.data = data[data['spt'] > 0]

        # Fit polynomial
        self.coeffs, self.C_p = np.polyfit(data['spt'], data['radius'], order,
                                           w=1./data['radius_unc'], cov=True)

        # Do the interpolation for plotting
        self.spt = np.arange(np.nanmin(data['spt']), np.nanmax(data['spt'])+1)

        # Matrix with rows 1, spt, spt**2, ...
        self.sptT = np.vstack([self.spt**(order-i) for i in range(order+1)]).T

        # Matrix multiplication calculates the polynomial values
        self.yi = np.dot(self.sptT, self.coeffs)

        # C_y = TT*C_z*TT.T
        self.C_yi = np.dot(self.sptT, np.dot(self.C_p, self.sptT.T))

        # Standard deviations are sqrt of diagonal
        self.sig_yi = np.sqrt(np.diag(self.C_yi))

        # Store the new order
        self.order = order

    def plot(self, draw=False):
        """Plot the relation

        Parameters
        ----------
        draw: bool
            Draw the figure, else return it

        Returns
        -------
        bokeh.plotting.figure
            The plotted figure
        """
        AFGK_color = '#1f77b4'
        MLTY_color = '#2ca02c'
        data_color = '#7f7f7f'

        # Configure plot
        TOOLS = ['pan', 'reset', 'box_zoom', 'wheel_zoom', 'save']
        xlab = 'Spectral Type'
        ylab = r'$Radius [R_\odot]$'
        fig = figure(plot_width=800, plot_height=500, title=self.name,
                          x_axis_label=xlab, y_axis_label=ylab,
                          tools=TOOLS)

        # Plot the fit
        fig.line(self.spt, self.yi, color=data_color,
                 legend='Order {} Fit'.format(self.order))
        x = np.append(self.spt, self.spt[::-1])
        y = np.append(self.yi-self.sig_yi, (self.yi+self.sig_yi)[::-1])
        fig.patch(x, y, fill_alpha=0.1, line_alpha=0, color=data_color)

        # Add the AFGK data
        AFGK = self.data[self.data['name'] == 'AFGK']
        fig.circle(AFGK['spt'], AFGK['radius'], size=8, color=AFGK_color,
                   legend='Boyajian+ 2012b, 2013')

        # Add the MLTY data
        MLTY = self.data[self.data['name'] == 'MLTY']
        fig.circle(MLTY['spt'], MLTY['radius'], size=8, color=MLTY_color,
                   legend='Filippazzo+ 2015')

        if draw:
            show(fig)
        else:
            return fig
