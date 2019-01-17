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
    def __init__(self, orders=(5, 3), name='Spectral Type vs. Radius'):
        """Initialize the object

        Parameters
        ----------
        order: int
            The order polynomial to fit to the spt-radius data
        """
        self.name = name
        self.generate(orders)

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
        # Convert to integer
        if isinstance(spt, (str, bytes)):
            spt = u.specType(spt)[0]

        # Test valid ranges
        if not isinstance(spt, (int, float)) or not 30 <= spt <= 99:
            raise ValueError("Please provide a spectral type within [30, 99]")

        # Evaluate the polynomials
        if spt > 64:
            data = self.MLTY
        else:
            data = self.AFGK
        radius = np.polyval(data['coeffs'], spt)*q.Rsun
        radius_unc = np.interp(spt, data['spt'], data['sig_yi'])*q.Rsun

        if plot:
            fig = self.plot()
            fig.triangle([spt], [radius.value], color='red', size=15, legend=u.specType(spt))
            show(fig)

        return radius, radius_unc

    def generate(self, orders):
        """Generate a polynomial that describes the radius as a function of
        spectral type for empirically measured AFGKM main sequence stars
        (Boyajian+ 2012b, 2013) and MLTY model isochrone interpolated stars
        (Filippazzoet al. 2015, 2016)

        Parameters
        ----------
        orders: sequence
            The order polynomials to fit to the MLTY and AFGK data
        generate: bool
            Generate the polynomials
        """
        # ====================================================================
        # Boyajian AFGKM data
        # ====================================================================

        afgk = resource_filename('sedkit', 'data/AFGK_radii.txt')
        afgk_data = ii.read(afgk, format='csv', comment='#')

        # ====================================================================
        # Filippazzo MLTY data
        # ====================================================================

        # Get the data
        cat1 = V.query_constraints('J/ApJ/810/158/table1')[0]
        cat2 = V.query_constraints('J/ApJ/810/158/table9')[0]

        # Join the tables to getthe spectral types and radii in one table
        mlty_data = at.join(cat1, cat2, keys='ID', join_type='outer')

        # Only keep field age
        mlty_data = mlty_data[mlty_data['b_Age'] >= 0.5]

        # Rename columns
        mlty_data.rename_column('SpT', 'spectral_type')
        mlty_data.rename_column('Rad', 'radius')
        mlty_data.rename_column('e_Rad', 'radius_unc')

        # Make solar radii units
        mlty_data['radius'] = mlty_data['radius'].to(q.Rsun)
        mlty_data['radius_unc'] = mlty_data['radius_unc'].to(q.Rsun)

        # ====================================================================
        # Fit and save the data
        # ====================================================================

        for data, name, order, ref, rng in zip([afgk_data, mlty_data],
                                               ['AFGK', 'MLTY'], orders,
                                               ['Boyajian+ 2012b, 2013', 'Filippazzo+ 2015'],
                                               [(30, 65), (65, 99)]):

            # Container for data
            container = {}

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
            data = data[(data['spt'] > rng[0]) & (data['spt'] < rng[1])]
            data = data[data['lum'] == 'V']
            data = data[((data['radius'] < 1.8) & (data['spt'] > 37)) | (data['spt'] <= 37)]

            # Filter out nans
            data = data[data['radius'] < 4]
            data = data[data['radius'] > 0]
            data = data[data['radius_unc'] > 0]
            container['data'] = data[data['spt'] > 0]
            container['rng'] = rng

            # Fit polynomial
            container['coeffs'], container['C_p'] = np.polyfit(data['spt'],
                                                               data['radius'],
                                                               order,
                                                               w=1./data['radius_unc'],
                                                               cov=True)

            # Do the interpolation for plotting
            container['spt'] = np.arange(np.nanmin(data['spt'])-3, np.nanmax(data['spt'])+1)

            # Matrix with rows 1, spt, spt**2, ...
            container['sptT'] = np.vstack([container['spt']**(order-i) for i in range(order+1)]).T

            # Matrix multiplication calculates the polynomial values
            container['yi'] = np.dot(container['sptT'], container['coeffs'])

            # C_y = TT*C_z*TT.T
            container['C_yi'] = np.dot(container['sptT'], np.dot(container['C_p'], container['sptT'].T))

            # Standard deviations are sqrt of diagonal
            container['sig_yi'] = np.sqrt(np.diag(container['C_yi']))

            # Store the new order
            container['order'] = order

            # Set the reference
            container['ref'] = ref

            # Add the container as an attribute
            setattr(self, name, container)

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

        # Configure plot
        TOOLS = ['pan', 'reset', 'box_zoom', 'wheel_zoom', 'save']
        xlab = 'Spectral Type'
        ylab = 'Solar Radii'
        fig = figure(plot_width=800, plot_height=500, title=self.name,
                          x_axis_label=xlab, y_axis_label=ylab,
                          tools=TOOLS)

        # Plot the fit
        for n, (data, color) in enumerate(zip([self.AFGK, self.MLTY], [AFGK_color, MLTY_color])):

            # Add the data
            if n == 0:
                fig.circle(data['data']['spt'], data['data']['radius'], size=8,
                           color=color, legend=data['ref'])
            else:
                fig.square(data['data']['spt'], data['data']['radius'], size=8,
                           color=color, legend=data['ref'])

            # Add the fit line and uncertainty
            fig.line(data['spt'], data['yi'], color=color,
                     legend='Order {} Fit'.format(data['order']))
            x = np.append(data['spt'], data['spt'][::-1])
            y = np.append(data['yi']-data['sig_yi'], (data['yi']+data['sig_yi'])[::-1])
            fig.patch(x, y, fill_alpha=0.1, line_alpha=0, color=color)

        if draw:
            show(fig)
        else:
            return fig
