"""
This is the code used to generate the polynomial relations
used in SEDkit's calculations
"""
from pkg_resources import resource_filename

import astropy.io.ascii as ii
import astropy.units as q
import astropy.table as at
from astroquery.vizier import Vizier
import numpy as np

from . import utilities as u


V = Vizier(columns=["**"])


def spt_radius_relation(xp=None, order=8, sig_order=7, generate=False):
    """Generate a polynomial that describes the radius as
    a function of spectral type for empirically measured
    main sequence stars (Boyajian+ 2012b, 2013)
    
    Parameters
    ----------
    xp: float
        The spectral type float, where 0-99 correspond to types O0-Y9
    order: int
        The polynomial order to fit to the data
    generate: bool
        Generate the polynomial
    """
    if generate:
        
        #======================================================================
        # Boyajian sample (A0-M6)
        #======================================================================
        
        # Get the data
        f = resource_filename('SEDkit', 'data/radii.txt')
        AFGK = ii.read(f, format='csv', comment='#')
        
        #======================================================================
        # Filippazo sample (M6-T8)
        #======================================================================
        
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
        
        #======================================================================
        
        # Join the two tables
        data = at.vstack([AFGK, MLTY], join_type='inner')
    
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
        data = data[data['lum']=='V']
        
        # Filter out nans
        data = data[data['radius']<4]
        data = data[data['radius']>0]
        data = data[data['radius_unc']>0]
        data = data[data['spt']>0]
    
        # Fit polynomial
        p, C_p = np.polyfit(data['spt'], data['radius'], order, w=1./data['radius_unc'], cov=True)

        # Do the interpolation for plotting
        t = np.arange(np.nanmin(data['spt']), np.nanmax(data['spt'])+1)

        # Matrix with rows 1, t, t**2, ...
        TT = np.vstack([t**(order-i) for i in range(order+1)]).T
        yi = np.dot(TT, p)  # matrix multiplication calculates the polynomial values
        C_yi = np.dot(TT, np.dot(C_p, TT.T)) # C_y = TT*C_z*TT.T
        sig_yi = np.sqrt(np.diag(C_yi))  # Standard deviations are sqrt of diagonal

        # Caluclate the uncertainty as a function of spectral type
        sig_p = np.polyfit(t, sig_yi, sig_order)

        # # Plot it
        # fg, (ax1, ax2) = plt.subplots(2, 1)
        # ax1.set_title("Fit for Polynomial (degree {}) with $\pm1\sigma$-interval".format(order))
        # ax1.fill_between(t, yi+sig_yi, yi-sig_yi, alpha=.25)
        # ax1.plot(t, yi,'-')
        # ax1.errorbar(data['spt'], data['radius'], yerr=data['radius_unc'], ls='none', marker='o', c='r', label='Boyajian+ 2012b, 2013')
        # ax1.set_ylabel('Radius [$R_\odot$]')
        # ax2.plot(t, sig_yi)
        # ax2.plot(t, np.polyval(sig_p, t))
        # ax2.set_title("Fit for $1\sigma$ Polynomial (degree {})".format(sig_order))
        # ax2.set_xlabel('Spectral Type')
        # ax2.set_ylabel('Uncertainty [$R_\odot$]')
        
        return p, sig_p
        
    elif xp is not None and 20<=xp<=90:
        
        # Use precomputed polynomial
        default = np.array([7.58092691e-12,-3.38093782e-09,6.43208620e-07,-6.80497975e-05,4.37080146e-03,-1.74200537e-01,4.19992200e+00,-5.59630176e+01,3.17382890e+02])
        sig_default = np.array([-2.86548806e-12,1.17656611e-09,-2.02883746e-07,1.90972008e-05,-1.06440025e-03,3.53953274e-02,-6.58260385e-01,5.38782735e+00])
        
        # Evaluate the polynomials
        radius = np.polyval(default, xp)*q.Rsun
        radius_unc = np.polyval(sig_default, xp)*q.Rsun
        
        return radius, radius_unc
        
    else:
        
        raise TypeError("Please provide 20<=xp<=66 to evaluate or set 'generate=True'")
        
        
        
    