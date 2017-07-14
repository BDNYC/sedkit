#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Joe Filippazzo, jfilippazzo@stsci.edu
#!python3
"""
SEDkit rewritten with astropy, pysynphot, and astrodbkit
"""

class GetData(object):
    def __init__(self, pickle_path):
        """
        Loads the data pickle constructed from SED calculations

        Parameters
        ----------
        pickle_path: str
          The path to the data pickle

        """
        try:
            self.pickle = open(pickle_path, 'rb')
            self.data = pickle.load(self.pickle)
            self.path = pickle_path
            self.close = self.pickle.close()
            print('Data from {} loaded!'.format(pickle_path))

        except IOError:
            print("Data from {} not loaded! Try again.".format(pickle_path))

    def add_source(self, data_dict, name, update=False):
        """
        Adds data to the pickle

        Parameters
        ----------
        data_dict: dict
          A nested dictionary of new data to add to self.data
        name: str
          The dictionary key to use for the new data
        update: bool
          Performs an update of the nested dictionary instead of replacing it

        """
        if isinstance(data_dict, dict):
            # Add the data to the active dictionary so we don't have to reload
            if update:
                self.data[name].update(data_dict)
            else:
                self.data[name] = data_dict

            # ...and add it to the pickle permanently
            try:
                pickle.dump(self.data, open(self.path, 'wb'))
                print('{} data added to {} pickle!'.format(name, self.path))
            except:
                print('Ut oh! {} data NOT added to {} pickle!'.format(name, self.path))

        else:
            print('The data input must be in the form of a Python dictionary.')

    def delete_source(self, name):
        """
        Removes the nested dictionary associated with the given source

        Parameters
        ----------
        name: str
          The dictionary key to delete from the data pickle

        """
        if name in self.data:
            # Remove this source from the active dictionary so we don't have to reload
            self.data.pop(name)

            # ...and remove it from the pickle permanently
            try:
                pickle.dump(self.data, open(self.path, 'wb'))
                print('{} data removed from {} pickle!'.format(name, self.path))
            except TypeError:
                print('Ut oh! {} data NOT removed from {} pickle!'.format(name, self.path))

        else:
            print('Source {} not found in {} pickle.'.format(name, self.data))

    def export_SED(self, name, filepath, key='SED_abs', header=True, 
                   wavelength_units='', flux_units=''):
        """
        Export an SEDs to an ascii file.

        Parameters
        ----------
        name: str
            The name of the object in the dictionary
        filepath: str
            The path to the directory for the file
        key: str
            The keyword to export
        header: bool
             Include the header
        flux_units: astropy.units.core.CompositeUnit
            The astropy units to convert to
        wavelength_units: astropy.units.core.CompositeUnit
            The astropy units to convert to
        """
        # Pull the object from the dictionary
        data = self.data[name]
        
        # Get the desired SED
        sed = data.get(key)
        
        # Convert to desired units
        sed[0] = sed[0].to(q.um)
        
        if flux_units==q.Jy:
            sed[1] = (sed[1]*sed[0]**2/ac.c).to(q.Jy)
            if len(sed)==3:
                sed[2] = (sed[2]*sed[0]**2/ac.c).to(q.Jy)
        else:
            sed[1] = sed[1].to(flux_units)
            if len(sed)==3:
                sed[2] = sed[2].to(flux_units)
        
        if sed:
            # Pull out all other values which are not arrays and place in the header
            head = sorted([['# {}'.format(k), str(v)] for k, v in data.items() 
                              if 'SED' not in k and 'RJ' not in k], key=lambda x: x[0])

            # Open the file
            fn = filepath + name.replace(' ', '_').replace('+', '%2B') + '.txt'

            # Write the header
            if head and header:
                try:
                    ascii.write([np.asarray(i) for i in np.asarray(head).T], fn, delimiter='=', format='no_header')
                except IOError:
                    pass

            # Write the data
            names = ['# wavelength [{}]'.format(str(wavelength_units)), 'flux [{}]'.format(str(flux_units))]
            if len(sed) == 3: 
                names += ['unc [{}]'.format(str(flux_units))]
            names[-1] = names[-1] + ' / {}'.format(key)

            with open(fn, mode='a') as f:
                ascii.write([np.asarray(i, dtype=np.float64) for i in sed], f, names=names, delimiter='\t')

        else:
            print('No {} for source {}'.format(key, name))

    def generate_mag_mag_relations(self, mag_mag_pickle=package + '/Data/Pickles/polynomial_relations.p', \
                                   pop=['TWA 27B', 'CD-35 2722b', 'HR8799b', '2MASS J11271382-3735076',
                                        'WISEA J182831.08+265037.6', '51 Eridani b']):
        """
        Generate estimated optical and MIR magnitudes for objects with NIR photometry based on magnitude-magnitude relations of the flux calibrated sample

        Parameters
        ----------
        mag_mag_pickle: str
          The path to the pickle which will store the magnitude-magnitude relations
        pop: sequence (ooptional)
          A list of sources to exclude when calculating mag-mag relations

        """
        bands, est_mags, rms_vals = ['M_' + i for i in RSR.keys()], [], []
        # pickle.dump({}, open(mag_mag_pickle,'wb'))
        Q = {}.fromkeys([b + '_fld' for b in bands] + [b + '_yng' for b in bands] + [b + '_all' for b in bands])

        # Photometry
        for b in bands:
            Q[b + '_fld'], Q[b + '_yng'], Q[b + '_all'] = {}.fromkeys(bands), {}.fromkeys(bands), {}.fromkeys(bands)

            # Iterate through and create polynomials for field, young, and all objects
            for name, groups in zip(['_fld', '_yng', '_all'], [('fld'), ('ymg', 'low-g'), ('fld', 'ymg', 'low-g')]):

                # Create band-band plots, fit polynomials, and store coefficients in polynomial_relations.p
                for c in bands:
                    # See how many objects qualify
                    sample = \
                        zip(*self.search(['SpT'], [c, b] + (['NYMG|gravity'] if name == '_yng' else [])) or ['none'])[0]

                    # If there are more than 10 objects across spectral types L0-T0, add the relation to the dictionary
                    if c != b and len(sample) > 20 and min(sample) <= 10 and max(sample) >= 20:

                        # Pop the band on interest from the dictionary, calculate polynomial, and add it as a nested dictionary
                        try:
                            P = \
                                self.mag_plot(c, b, pop=pop, fit=[(groups, 3, 'k', '-')], weighting=False,
                                              est_mags=False)[
                                    0][1:]
                            plt.close()
                            if P[1] != 0:
                                Q[b + name][c] = {'rms': P[1], 'min': P[0][0], 'max': P[0][1], 'yparam': b + name,
                                                  'xparam': c}
                                Q[b + name][c].update({'c{}'.format(n): p for n, p in enumerate(P[2:])})
                                print('{} - {} relation added!'.format(b + name, c))
                        except:
                            pass

        pickle.dump(Q, open(mag_mag_pickle, 'wb'))

    def mag_plot(self, xparam, yparam, zparam='', add_data={}, pct_lim=1000, est_mags=True, plot_field=True, \
                 identify=[], label_objects={}, pop=[], sources=[], binaries=False, allow_null_unc=False, \
                 fit=[], weighting=True, spt=['M', 'L', 'T', 'Y'], groups=['fld', 'low-g', 'ymg'], \
                 evo_model='hybrid_solar_age', biny=False, id_NYMGs=False, legend=True, add_text=False, \
                 xlabel='', xlims='', xticks=[], invertx='', xmaglimits='', \
                 ylabel='', ylims='', yticks=[], inverty='', ymaglimits='', \
                 zlabel='', zlims='', zticks=[], invertz='', zmaglimits='', \
                 border=['#FFA821', '#FFA821', 'k', 'r', 'r', 'k', '#2B89D6', '#2B89D6', 'k', '#7F00FF', '#7F00FF', 'k'], \
                 markers=['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'], \
                 colors=['#FFA821', 'r', '#2B89D6', '#7F00FF'], \
                 fill=['#FFA821', 'w', '#FFA821', 'r', 'w', 'r', '#2B89D6', 'w', '#2B89D6', '#7F00FF', 'w', '#7F00FF'], \
                 overplot=False, grid=False, fontsize=20, figsize=(10, 8), unity=False, save='', alpha=1, \
                 verbose=False, output_data=False, return_data='polynomials', save_polynomial=''):
        '''
        Plots the given parameters for all available objects in the given data_table

        Parameters
        ----------
        xparam: str
          The key for the given parameter in the data_table dictionary to serve as the x value.
          This can be a single key like 'J' or 'teff' or the difference of two keys, e.g. 'J-Ks')
        yparam: str
          The key for the given parameter in the data_table dictionary to serve as the y value.
          This can be a single key like 'J' or 'teff' or the difference of two keys, e.g. 'J-Ks')
        zparam: str (optional)
          The key for the given parameter in the data_table dictionary to serve as the z value.
          This can be a single key like 'J' or 'teff' or the difference of two keys, e.g. 'J-Ks')
        data_table: dict
          The nested dictionary of objects to potentially plot
        add_data: dict (optional)
          A nested dictionary of additional objects to plot that are not present in the data_table
        identify: list, tuple (optional)
          A sequence of the object names to identify with a star on the plot

        '''
        D = copy.deepcopy(self.data)
        x, y, z, data_out = xparam.split('-'), yparam.split('-'), zparam.split('-'), []
        NYMG_dict = {'TW Hya': 'k', 'beta Pic': 'c', 'Tuc-Hor': 'g', 'Columba': 'm', 'Carina': 'k', 'Argus': 'y',
                     'AB Dor': 'b', 'Pleiades': 'r'}

        # Set limits on x and y axis data
        xmaglimits, ymaglimits = xmaglimits or (-np.inf, np.inf), ymaglimits or (-np.inf, np.inf)

        # Add additional sources
        D.update(add_data)

        # Pop unwanted sources
        for name in pop:
            try:
                D.pop(name)
            except:
                pass

        # Include specific sources
        if sources:
            D = {k: v for k, v in D.items() if k in sources}

        def num(Q):
            ''' Strips alphanumeric strings and Quantities of chars and units and turns them into floats '''
            if Q and not isinstance(Q, (float, int)):
                if isinstance(Q, q.quantity.Quantity):
                    Q = Q.value
                elif isinstance(Q, str):
                    Q = float(re.sub(r'[a-zA-Z]', '', Q))
                else:
                    Q = 0
            return Q

        # =======================================================================================================================================================================
        # ============================================ Sorting ==================================================================================================================
        # =======================================================================================================================================================================

        # Iterate through data and add object to appropriate group
        M_all, M_fld, M_lowg, M_ymg = [], [], [], []
        L_all, L_fld, L_lowg, L_ymg = [], [], [], []
        T_all, T_fld, T_lowg, T_ymg = [], [], [], []
        Y_all, Y_fld, Y_lowg, Y_ymg = [], [], [], []
        beta, gamma, binary, circle = [], [], [], []
        labels, rejected = [], []

        for name, v in D.items():

            # Try to add an estimated mag to the dictionary
            for b in filter(None, x + y + z):

                # If synthetic magnitude exists but survey magnitude doesn't, use the synthetic magnitude
                if v.get(b) and not v.get(b + '_unc') and b not in ['source_id'] and not allow_null_unc:
                    v[b], v[b + '_unc'] = v.get('syn_' + b), v.get('syn_' + b + '_unc')

                # Estimate photometry based on linear mag-mag relations
                if est_mags:
                    if v.get('d'):
                        # Convert MKO JHK mags to 2MASS JHKs mags if necessary
                        TMS = ['M_2MASS_J', 'M_2MASS_H', 'M_2MASS_Ks', '2MASS_J', '2MASS_H', '2MASS_Ks']
                        MKO = ['M_MKO_J', 'M_MKO_H', 'M_MKO_K', 'MKO_J', 'MKO_H', 'MKO_K']
                        Lband = np.array(['M_IRAC_ch1', 'M_WISE_W1', "M_MKO_L'"])

                        for tms, mko in zip(TMS, MKO):
                            if b == tms and not v.get(b):
                                try:
                                    v[b], v[b + '_unc'], v[b + '_ref'] = mag_mag_relations(b, v, [mko], to_flux=False)
                                except:
                                    pass
                        for mko, tms in zip(MKO, TMS):
                            if b == mko and not v.get(b):
                                try:
                                    v[b], v[b + '_unc'], v[b + '_ref'] = mag_mag_relations(b, v, [tms], to_flux=False)
                                except:
                                    pass
                        for l in Lband:
                            if l[2:] in b and any([v.get(i) for i in Lband[Lband != l]]):
                                try:
                                    v[b], v[b + '_unc'], v[b + '_ref'] = mag_mag_relations(b, v, Lband[Lband != l],
                                                                                           to_flux=False)
                                except:
                                    pass
                else:
                    # Exclude all magnitudes that were estimated from polynomials, i.e. have an absolute magnitude as the reference
                    if str(v.get(b.replace('M_', '') + '_ref')).startswith('M_'):
                        v.pop(b)

            # Check to see if all the appropriate parameters are present from the SED
            if all([v.get(i) for i in filter(None, x + y + z)]):
                if 'SpT' not in [x, y, z] and not isinstance(v.get('SpT'), (int, float)): v['SpT'] = 13
                try:
                    # Pull out x, y, and z values and caluculate differences if necessary,
                    # e.g. 'J-W2' retrieves J and W2 mags separately and calculates color
                    i, j, k = [(np.diff(list(reversed([num(v.get(m, 0)) \
                                                           if all(
                        [num(v.get(n, 0)) and num(v.get(n + '_unc', 0)) for n in p]) \
                                                           else 0 for m in p]))) if len(p) == 2 \
                                    else [num(v.get(p[0], 0))])[0] for p in [x, y, z]]

                    # Pull out uncertainties and caluclate the sum of the squares if the parameter is a color
                    i_unc, j_unc, k_unc = [np.sqrt(sum([num(v.get(e + '_unc', 0)) ** 2 for e in p])) for p in [x, y, z]]

                    # Put all the applicable data into a list to pass through plotting criteria
                    data = [name, i, i_unc, j, j_unc, k if zparam else True, k_unc if zparam else True, \
                            v.get('gravity') or (True if v.get('age_min') < 500 else False), \
                            v.get('binary'), v.get('SpT'), v.get('SpT_unc', 0.5), v.get('NYMG')]

                    # If all the necessary data is there, drop it into the appropriate category for plotting
                    if all([data[1], data[3], data[5]]) \
                            and (binaries or (not binaries and not data[-4])) \
                            and (all([data[2], data[4], data[6]]) or allow_null_unc) \
                            and all([data[1] > xmaglimits[0], data[1] < xmaglimits[1], data[3] > ymaglimits[0],
                                     data[3] < ymaglimits[1]]) \
                            and all([(100 * ((np.e ** err - 1) if param in RSR.keys() or 'bol' in param \
                                                     else (err / val)) < pct_lim) if 'SpT' not in param else True \
                                     for param, val, err in
                                     zip([xparam, yparam], [data[1], data[3]], [data[2], data[4]])]):

                        # Is it a binary?
                        if v.get('binary') and binaries: binary.append(data)

                        # Sort through and put them in the appropriate *spt* and *groups*
                        for A, Y, L, F, low, high, S in zip([M_all, L_all, T_all, Y_all], \
                                                            [M_ymg, L_ymg, T_ymg, Y_ymg], \
                                                            [M_lowg, L_lowg, T_lowg, Y_lowg], \
                                                            [M_fld, L_fld, T_fld, Y_fld], \
                                                            range(0, 40, 10), range(10, 50, 10), \
                                                            ['M', 'L', 'T', 'Y']):

                            if data[-3] >= low and data[-3] < high and S in spt:

                                # Get total count
                                if all([data[-1], 'ymg' in groups]) \
                                        or all([data[7], 'low-g' in groups]) \
                                        or all(['fld' in groups, not data[-1], not data[7]]):
                                    A.append(data)

                                # Is the object field age, low gravity, or a NYMG member?
                                Y.append(data) if data[-1] and 'ymg' in groups \
                                    else L.append(data) if data[7] and 'low-g' in groups \
                                    else F.append(data) if not data[-1] and not data[7] and 'fld' in groups \
                                    else None

                                # Distinguish between beta and gamma
                                beta.append(data) if data[7] == 'b' and 'beta' in groups \
                                    else gamma.append(data) if data[7] == 'g' and 'gamma' in groups \
                                    else None

                        # Is it in the list of objects to identify or label?
                        if data[0] in identify:
                            circle.append(data)

                        if data[0] in label_objects.keys() and any([data in i for i in [M_all, L_all, T_all, Y_all]]):
                            labels.append([label_objects[data[0]].get('label', data[0]), data[1], data[3], \
                                           label_objects[data[0]]['dx'], label_objects[data[0]]['dy']])

                        # If object didn't make it into any of the bins, put it in a rejection table
                        if not any([data in i for i in [M_all, L_all, T_all, Y_all]]): rejected.append(data)

                    else:
                        rejected.append(data)
                except:
                    print('error: ', name)

        # =======================================================================================================================================================================
        # ============================================ Plotting =================================================================================================================
        # =======================================================================================================================================================================

        # Plot each group with specified formatting
        if any([M_fld, M_lowg, M_ymg, L_fld, L_lowg, L_ymg, T_fld, T_lowg, T_ymg, Y_fld, Y_lowg, Y_ymg, beta, gamma,
                binary]):
            fill = fill or [colors[0], 'w', colors[0], colors[1], 'w', colors[1], colors[2], 'w', colors[2], colors[3],
                            'w', colors[3]]
            border = border or [colors[0], colors[0], 'k', colors[1], colors[1], 'k', colors[2], colors[2], 'k',
                                colors[3], colors[3], 'k']
            if not overplot:
                if zparam: from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=figsize)
                ax = plt.subplot(111,
                                 projection='3d' if zparam else 'aitoff' if xparam == 'ra' and yparam == 'dec' else 'rectilinear')
                plt.rc('text', usetex=True, fontsize=fontsize)
                ax.set_xlabel(r'${}$'.format(xlabel or '\mbox{Spectral Type}' if 'SpT' in xparam else xparam),
                              labelpad=fontsize * 3 / 4, fontsize=fontsize + 4)
                ax.set_ylabel(r'${}$'.format(ylabel or yparam), labelpad=fontsize * 3 / 4, fontsize=fontsize + 4)
                if zparam: ax.set_zlabel(r'${}$'.format(zlabel or zparam), labelpad=fontsize, fontsize=fontsize + 4)
            else:
                ax = overplot if hasattr(overplot, 'figure') else plt.gca()

            if biny and 'SpT' in xparam:
                M_fld, M_lowg, M_ymg = [], [], []
                L_fld, L_lowg, L_ymg = [], [], []
                T_fld, T_lowg, T_ymg = [], [], []
                Y_fld, Y_lowg, Y_ymg = [], [], []

                for ungrp, grp in zip([M_all, L_all, T_all, Y_all], [M_fld, L_fld, T_fld, Y_fld]):
                    for k, g in groupby(sorted(ungrp, key=lambda x: int(x[1])), lambda y: int(y[1])):
                        G = list(g)
                        grp.append([str(k), k, 0.5,
                                    np.average(np.array([i[3] for i in G]), weights=1. / np.array([i[4] for i in G]) \
                                        if any([i[4] for i in G]) else None), \
                                    np.sqrt(sum(np.array([i[4] for i in G]) ** 2)) \
                                        if any([i[4] for i in G]) else np.std([i[3] for i in G]), 0, 0, 0, 0, k, 0.5,
                                    None])

            # Do polynomial fits of *degree* to data
            if fit:
                for grps, degree, c, ls in fit:
                    sample = (M_fld + L_fld + T_fld + Y_fld if 'fld' in grps else []) \
                             + (M_lowg + L_lowg + T_lowg + Y_lowg if 'low-g' in grps else []) \
                             + (M_ymg + L_ymg + T_ymg + Y_ymg if 'ymg' in grps else []) \
                             + (beta if 'beta' in grps else []) \
                             + (gamma if 'gamma' in grps else [])

                    if sample and not zparam:
                        N, X, Xsig, Y, Ysig, Z, Zsig, G, B, S, Ssig, NYMG = zip(*[i for i in sample if not i[8]])
                        if return_data == 'polynomials':
                            suffix = '_fld' if 'fld' in grps and 'ymg' not in grps and 'low-g' not in grps \
                                else '_yng' if 'fld' not in grps and 'ymg' in grps and 'low-g' in grps \
                                else '_all'
                            pr = u.output_polynomial(map(float, X), map(float, Y),
                                                     sig=map(float, Ysig) if weighting else '', \
                                                     title='{} | {}'.format(spt, grps), degree=degree, x=xparam,
                                                     y=yparam + suffix, verbose=verbose, \
                                                     c=c, ls=ls, legend=False, ax=ax)
                            if save_polynomial:
                                polynomial_relation(pr['xparam'], pr['yparam'], polynomial=pr,
                                                    pickle_path=save_polynomial)

            # Plot the data
            for z, l, m, c, e in zip(*[
                [M_fld, M_lowg, M_ymg, L_fld, L_lowg, L_ymg, T_fld, T_lowg, T_ymg, Y_fld, Y_lowg, Y_ymg, beta, gamma,
                 binary, circle], \
                    ['M Field', 'M ' + r'$\beta /\gamma$', 'M NYMG', 'L Field', 'L ' + r'$\beta /\gamma$', 'L NYMG', \
                     'T Field', 'T ' + r'$\beta /\gamma$', 'T NYMG', 'Y Field', 'Y ' + r'$\beta /\gamma$', 'Y NYMG', \
                     'Beta', 'Gamma', 'Binary', 'Interesting'], \
                            markers + ['d', 'd', 's', '*'], fill + ['w', 'w', 'none', 'k'],
                        border + ['b', 'g', 'g', 'k']]):

                if z:
                    if 'NYMG' in l and id_NYMGs:
                        for k, g in groupby(sorted(z, key=lambda x: x[-1]), lambda y: y[-1]):
                            G = list(g)
                            N, X, Xsig, Y, Ysig, Z, Zsig, G, B, S, Ssig, NYMG = zip(*G)
                            if zparam:
                                ax.scatter(X, Y, Z)
                            else:
                                ax.errorbar(X, Y, ls='none', marker='d', markerfacecolor=NYMG_dict[k],
                                            markeredgecolor='k', ecolor='k', \
                                            markeredgewidth=1,
                                            markersize=10 if l == 'Binary' else 20 if l == 'Interesting' else 7,
                                            label=l, \
                                            capsize=0,
                                            zorder=0 if 'Field' in l else 3 if l in ['Binary', 'Interesting'] else 2,
                                            alpha=alpha)

                    else:
                        N, X, Xsig, Y, Ysig, Z, Zsig, G, B, S, Ssig, NYMG = zip(*z)
                        if not plot_field and 'Field' in l:
                            pass
                        else:
                            if zparam:
                                ax.scatter(X, Y, Z, marker=m, facecolor=c, edgecolor=e,
                                           linewidth=0 if 'Field' in l else 2, s=10 if l == 'Binary' else 8, \
                                           label=l,
                                           zorder=0 if 'Field' in l else 3 if l in ['Binary', 'Interesting'] else 2,
                                           alpha=alpha)
                            else:
                                ax.errorbar(X, Y, xerr=None if l == 'Binary' else [Xsig, Xsig],
                                            yerr=None if l in ['Binary', 'Interesting'] else [Ysig, Ysig], \
                                            ls='none', marker=m, markerfacecolor=c, markeredgecolor=e, ecolor=e,
                                            markeredgewidth=0 if 'Field' in l else 1, \
                                            markersize=10 if l == 'Binary' else 20 if l == 'Interesting' else 8,
                                            label=l, capsize=0, \
                                            zorder=0 if 'Field' in l else 3 if l in ['Binary', 'Interesting'] else 2,
                                            alpha=alpha)
                                plt.connect('button_press_event', interact.AnnoteFinder(X, Y, N))

                    if verbose:
                        print('\n',l,':')
                        pdata = [N, S, X, Xsig, Y, Ysig, Z, Zsig, G, B, NYMG] if zparam else [N, S, X, Xsig, Y, Ysig, G, B, NYMG]
                        colnames = ['Name', 'spt', xparam, xparam+'_unc', yparam, yparam+'_unc', zparam, zparam+'_unc', 'Gravity', 'Binary', 'Age'] \
                                   if zparam else ['Name', 'spt', xparam, xparam+'_unc', yparam, yparam+'_unc', 'Gravity', 'Binary', 'Age']
                        at.Table(pdata, names=colnames).pprint(max_width=120, max_lines=300)

                if return_data == 'params': 
                    data_out.append(z)
                
                if output_data and output_data != 'polynomials':
                    u.printer(['Name', 'SpT', xparam, xparam + '_unc', yparam, yparam + '_unc', zparam, zparam + '_unc',
                               'Gravity', 'Binary', 'Age'], \
                              zip(*[N, S, X, Xsig, Y, Ysig, Z, Zsig, G, B, NYMG]), empties=True,
                              to_txt=package + '/Files/{} v {} v {}.txt'.format(xparam, yparam, zparam)) if zparam \
                        else u.printer(['Name', xparam, xparam + '_unc', yparam, yparam + '_unc', 'Gravity', 'Binary',
                                        'Age'] if xparam == 'SpT' \
                                           else ['Name', 'SpT', xparam, xparam + '_unc', yparam, yparam + '_unc',
                                                 'Gravity', 'Binary', 'Age'],
                                       zip(*[N, X, Xsig, Y, Ysig, G, B, NYMG]) if xparam == 'SpT' \
                                           else zip(*[N, S, X, Xsig, Y, Ysig, G, B, NYMG]), empties=True,
                                       to_txt='/Files/{} v {}.txt'.format(xparam, yparam))

            # Options to format axes, draw legend and save
            if 'SpT' in xparam and spt == ['M', 'L', 'T', 'Y'] and not xticks:
                ax.set_xlim(5, 33)
                xticks = ['M6', 'M8', 'L0', 'L2', 'L4', 'L6', 'L8', 'T0', 'T2', 'T4', 'T6', 'T8', 'Y0', 'Y2']
                ax.set_xticks([6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])

            if 'SpT' in xparam and spt == ['M', 'L', 'T'] and not xticks:
                ax.set_xlim(5, 31)
                xticks = ['M6', 'L0', 'L4', 'L8', 'T2', 'T6', 'Y0']
                ax.set_xticks([6, 10, 14, 18, 22, 26, 30])

            # Axis formatting
            if grid: ax.grid(True)
            if xticks: ax.set_xticklabels(xticks)
            if xlabel: ax.set_xlabel(xlabel, labelpad=20)
            if invertx: ax.invert_xaxis()
            if xlims:
                ax.set_xlim(xlims)
            elif not xlims and all([i in RSR.keys() for i in x]):
                ax.set_xlim((min([i[1] - i[2] for i in M_all + L_all + T_all + Y_all]) - 1,
                             max([i[1] + i[2] for i in M_all + L_all + T_all + Y_all]) + 1))

            if yticks: ax.set_yticklabels(yticks)
            if ylabel: ax.set_ylabel(ylabel, labelpad=20)
            if inverty: ax.invert_yaxis()
            if ylims:
                ax.set_ylim(ylims)
            elif not ylims and all([i in RSR.keys() for i in y]):
                ax.set_ylim((min([i[3] - i[4] for i in M_all + L_all + T_all + Y_all]) - 1,
                             max([i[3] + i[4] for i in M_all + L_all + T_all + Y_all]) + 1))

            if zparam:
                if zticks: ax.set_zticklabels(zticks)
                if zlabel: ax.set_zlabel(zlabel, labelpad=20)
                if invertz: ax.invery_zaxis()
                if zlims:
                    ax.set_zlim(zlims)
                elif not zlims and all([i in RSR.keys() for i in z]):
                    ax.set_zlim((min([i[5] - i[6] for i in M_all + L_all + T_all + Y_all]) - 1,
                                 max([i[5] + i[6] for i in M_all + L_all + T_all + Y_all]) + 1))

            # Axes text
            if not zparam:
                if add_text: ax.annotate(add_text[0], xy=add_text[1], xytext=add_text[2], fontsize=add_text[3])
                if labels:
                    for l, x, y, dx, dy in labels:
                        ax.annotate(l, xy=(x, y), xytext=(x + dx, y + dy), fontsize=14)
                if unity:
                    X, Y = ax.get_xlim(), ax.get_ylim()
                    ax.plot(X, Y, c='k', ls='--')
                    ax.set_xlim(X), ax.set_ylim(Y)

                plt.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.98)

            # Plot the legend (Breaks with 3D projection)
            if legend and not zparam:
                if binaries:
                    u.manual_legend(
                        ['M Field', 'L Field', 'T Field', 'Binary', r'M$\beta /\gamma$', r'L$\beta /\gamma$',
                         r'T$\beta /\gamma$', 'M NYMG', 'L NYMG', 'T NYMG'],
                        ['#FFA821', 'r', '#2B89D6', 'w', 'w', 'w', 'w', '#FFA821', 'r', '#2B89D6'], overplot=ax,
                        markers=['o', 'o', 'o', 's', 'o', 'o', 'o', 'o', 'o', 'o'],
                        sizes=[8, 8, 8, 12, 8, 8, 8, 8, 8, 8, 8, 8],
                        edges=['#FFA821', 'r', '#2B89D6', 'g', '#FFA821', 'r', '#2B89D6', 'k', 'k', 'k'],
                        errors=[True, True, True, False, True, True, True, True, True, True],
                        styles=['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'], ncol=3, loc=legend)
                else:
                    u.manual_legend(spt + (['Field', r'$\beta /\gamma$', 'NYMG'] if groups != ['fld'] else []),
                                    colors + ['0.5', 'w', '0.5'], sizes=[8, 8, 8, 8, 8, 8, 8], overplot=ax,
                                    markers=['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
                                    edges=['#FFA821', 'r', '#2B89D6', '#7F00FF', '0.5', '0.5', '0.3'],
                                    errors=[False, False, False, False, True, True, True],
                                    ncol=1 if groups == ['fld'] else 2, loc=0)

            # Saving, returning, and printing
            if save: 
                plt.savefig(save if '.png' in save else (save + '{} vs {}.png'.format(yparam, xparam)))

            view = [['Field', 'NYMG', 'low_g', 'Total'],
                    [len(M_fld), len(M_ymg), len(M_lowg), len(M_fld + M_ymg + M_lowg)],
                    [len(L_fld), len(L_ymg), len(L_lowg), len(L_fld + L_ymg + L_lowg)],
                    [len(T_fld), len(T_ymg), len(T_lowg), len(T_fld + T_ymg + T_lowg)],
                    [len(Y_fld), len(Y_ymg), len(Y_lowg), len(Y_fld + Y_ymg + Y_lowg)],
                    [len(M_fld + L_fld + T_fld + Y_fld), len(M_ymg + L_ymg + T_ymg + Y_ymg),
                     len(M_lowg + L_lowg + T_lowg + Y_lowg), 
                     len(M_fld + M_ymg + M_lowg + L_fld + L_ymg + L_lowg + T_fld + T_ymg + T_lowg + Y_fld + Y_ymg + Y_lowg)]]
            presults = at.Table(view, names=['-','M', 'L', 'T', 'Y', 'Total'])
            print('\n')
            presults.pprint()
            
            
            if rejected and verbose:
                print('\nRejected:')
                rej = at.Table(list(zip(*rejected)), dtype=[str,float,float,float,float,float,float,str,bool,float,float,str],
                        names=['name', xparam, xparam+'_unc', yparam, yparam+'_unc', zparam or 'z',
                        zparam+'_unc' if zparam else 'z_unc', 'gravity', 'binary', 'spt', 'spt_unc', 'NYMG'])
                rej.pprint(max_width=120, max_lines=300)
                
            if return_data and data_out: 
                return data_out

        else:
            print("No objects with {} and {} values.".format(xparam, yparam))

    def RESET(self):
        """
        Empties the data_pickle after a prompt
        """
        sure = raw_input("Are you sure you want to delete all data from {} pickle? [No,Yes]".format(self.path))
        if sure.lower() == 'yes':
            try:
                pickle.dump({}, open(self.path, 'wb'))
                print('All data removed from {} pickle!'.format(self.path))
            except:
                print('Ut oh! Could NOT delete all data from {} pickle!'.format(self.path))

        else:
            print("You must respond 'Yes' to delete all data from {} pickle!".format(self.path))

    def search(self, keys, requirements, sources=[], spt='', fmt='array', to_txt=False, delim='|', keysort='', \
               verbose=False):
        '''
        Returns list of all values for *keys* of objects that satisfy all *requirements*

        Parameters
        ----------
        keys: list, tuple
          A sequence of the dictionary keys to be returned
        requirements: list, tuple
          A sequence of the dictionary keys to be evaluated as True or False.
          | (or) and ~ (not) operators recognized, e.g ['WISE_W1|IRAC_ch1'] and ['~publication_shortname']
        sources: sequence (optional)
          A list of the sources to include exclusively
        fmt: str
          Returns an array, dictionary, or Astropy table of the results given 'array', 'dict', and 'table' respectively
        to_txt: bool
          Writes an ascii file with delimiter **delim to the path supplied by **to_txt
        delim: str
          The delimiter to use when writing data to a text file
        keysort: str (option)
          Sorts the columns by the given key
        verbose: bool
          Print the details

        Returns
        -------
        result: list, dict, table
          A container of the values for all objects that satisfy the given requirements

        '''
        L = copy.deepcopy(self.data)
        result, rejected = [], []

        # Option to provide a list of sources to include
        if sources:
            for k, v in L.items():
                if k not in sources: L.pop(k)

        # Fetch the data that satisfy the requirements
        for n, d in L.items():
            if all([any([True if (not d.get(j.replace('~', '')) and j.startswith('~')) \
                                 else d.get(j) for j in i.split('|')]) for i in requirements]):
                if spt:
                    if d.get('SpT') >= spt[0] and d.get('SpT') <= spt[1]:
                        result.append([d.get(k) for k in keys])
                else:
                    result.append([d.get(k) for k in keys])
            else:
                rejected.append(n)

        # Sort the columns
        if keysort in keys:
            result = np.asarray(sorted(result, key=lambda x: x[keys.index(keysort)]))

        if fmt == 'table' and not to_txt:
            result = at.Table(np.asarray(result), names=keys)
            if keysort: result.sort(keysort)
        if fmt == 'dict' and not to_txt:
            result = {n: {k: v for k, v in d.items() if k in keys} for n, d in L.items() \
                      if all([any([True if (not d.get(j.replace('~', '')) and j.startswith('~')) \
                                       else d.get(j) for j in i.split('|')]) for i in requirements])}

        print('{}/{} records found satisfying {}.'.format(len(result), len(L), requirements))
        if verbose: print(rejected)

        # Print to file or return data
        if to_txt:
            np.savetxt(to_txt, unitless(result), header=delim.join(keys), delimiter=delim, fmt='%s')
        else:
            return result

    def subsample(self, sources):
        """
        Select a subsample from the GetData() instace given a sequence of source keys

        Parameters
        ----------
        source_list: sequence
          The list of keys to include in the GetData() instance

        """
        subset = {}
        for source in sources:
            if source in self.data:
                subset[source] = self.data[source]
            else:
                print('{} not in {}'.format(source, self.path))

        if len(subset) != len(sources):
            if raw_input('Not all specified sources were included. Proceed anyway? [y/n] ') == 'y':
                self.data = subset
            else:
                print('Subsample not created.')
        else:
            self.data = subset

    def spec_plot(self, sources=[], um=(0.5, 14.5), spt=(5, 33), teff=(0, 9999), SNR=0.5,
                  groups=['fld', 'ymg', 'low-g'], plot_phot=True, norm_to='', app=False, 
                  add_nans=[1.4, 1.85, 2.6, 4.3, 5.05], binaries=False, pop=[], highlight=[], 
                  cmap=u.truncate_colormap(plt.cm.brg_r, 0.5, 1.), cbar=True, figsize=(12, 8),
                  fontsize=18, overplot=False, legend='None', save='', low_SNR=True, 
                  ylabel='', xlabel='', lw=1, plot_integrals=False, zorder=1, **kwargs):
        """
        Plot flux calibrated or normalized SEDs for visual comparison.

        Parameters
        ----------
        sources: sequence (optional)
          A list of sources names to include exclusively
        um: sequence
          The wavelength range in microns to include in the plot
        spt: sequence
          The numeric spectral type range to include
        teff: sequence
          The effective temperture range to include
        SNR: int, float
          The signal-to-noise ratio above which the spectra should be masked
        groups: sequence
          The gravity groups to include, including 'fld' for field gravity, 'low-g' for low gravity
          designations, and 'ymg' for members of nearby young moving groups
        norm_to: sequence (optional)
          The wavelength range in microns to which all spectra should be normalized
        app: bool
          Plot apparent fluxes instead of absolute
        plot_phot: bool
          Plot the photometry
        add_nans: sequence
          A sequence of wavelength positions in microns to insert NaN values for nicer plotting
        binaries: bool
          Include known binaries
        cmap: colormap object, list
          The matplotlib colormap to use or a list of manual colors
        pop: sequence (optional)
          The sources to exclude from the plot
        highlight: sequence (optional)
          The wavelength ranges to highlight to point out interesting spectral features,
          e.g. [(6.38,6.55),(11.7,12.8),(10.3,11.3)] for MIR spectra
        cbar: bool
          Plot the color bar
        legend: int
          The 0-9 location to plot the legend. Does not plot legend if 'None'
        figsize: sequence
          The (x,y) dimensions for the plot
        fontsize: int
          The size of the font
        overplot: matplotlib figure
          The axes to plot the figure on
        xlabel: str (optional)
          The x-axis label
        ylabel: str (optional)
          The y-axis label
        save: str (optional)
          The path to save the plot

        """
        if overplot:
            ax = overplot if hasattr(overplot, 'figure') else plt.gca()
        else:
            fig, ax = plt.subplots(figsize=figsize, **kwargs)
            if cbar:
                cbar = ax.contourf([[0, 0], [0, 0]], range(teff[0], teff[1], int((teff[1] - teff[0]) / 20.)), cmap=cmap)
                ax.cla()
        L = copy.deepcopy(self.data)

        # If you want to specify the sources to include
        if sources:
            for k in L.keys():
                if k not in sources: L.pop(k)

        # Iterate through the list and plot the sources that satisfy *kwarg criteria
        plots, ylims, cidx = [], [], 0
        for k, v in L.items():
            try:
                # Get the spectrum
                spec = np.asarray(
                    [i.value if hasattr(i, 'unit') else i for i in v.get('SED_spec_' + ('app' if app else 'abs'))])

                # Add NaN to gaps
                for w in add_nans:
                    spec = np.concatenate([spec.T, [[w, np.nan, np.nan]]])
                    spec = np.asarray(spec[np.argsort(spec.T)[0]])
                    idx = np.where(spec == [w, np.nan, np.nan])[0][0]
                    try:
                        spec[idx] = [spec[idx + 1][0] * 0.999, np.nan, np.nan]
                    except:
                        pass
                    spec = spec.T

                    # Make the spectrum mask
                mask = np.logical_and(spec[0] > um[0], spec[0] < um[1])
                norm_mask = np.logical_and(spec[0] > norm_to[0], spec[0] < norm_to[1]) if norm_to else mask
                norm = 1. / np.trapz(spec[1][norm_mask], x=spec[0][norm_mask]) if norm_to else 1.

                # Get temperature
                try:
                    Teff = v.get('teff', 0 * q.K).value
                except AttributeError:
                    Teff = 0

                # Plot the spectra which satisfy the criteria
                if any(mask) \
                        and (((app or (not app and Teff > teff[0] and Teff < teff[1])) \
                                      and (binaries or (not binaries and not v.get('binary'))) \
                                      and v.get('SpT') >= spt[0] and v.get('SpT') <= spt[1] \
                                      and any(['low-g' in groups and v.get('gravity'), \
                                               'ymg' in groups and v.get('NYMG'), \
                                               'fld' in groups and not v.get('gravity') and not v.get('NYMG')]) \
                                      and k not in pop)
                             or (k in sources)):
                    try:
                        # Pick the color
                        if isinstance(cmap, (list,tuple)):
                            color = cmap[cidx]
                        else:
                            color = cmap((1. * Teff - teff[0]) / (teff[1] - teff[0]), 1.) if Teff else '0.5'

                        # Plot the integral surface
                        if plot_integrals:
                            intgrl = np.asarray([i.value if hasattr(i, 'unit') else i for i in
                                                 v.get('SED_' + ('app' if app else 'abs'))])
                            ax.plot(intgrl[0], intgrl[1], ls='--', color=color, zorder=zorder)

                        # Plot the high SNR spectra
                        hiSNRflux = np.ma.masked_where(
                            np.logical_and((spec[1][mask] / spec[2][mask]) < SNR, spec[1][mask] != np.nan),
                            spec[1][mask] * norm)
                        ax.step(spec[0][mask], hiSNRflux, lw=2 if color=='r' else 1, color=color, zorder=zorder+(2 if color=='r' else 1))
                        
                        # Plot the low SNR spectra
                        if low_SNR:
                            lwSNRflux = np.ma.masked_where(
                                np.logical_and((spec[1][mask] / spec[2][mask]) > SNR, spec[1][mask] != np.nan),
                                spec[1][mask] * norm)
                            ax.step(spec[0][mask], lwSNRflux, lw=lw, color=color, alpha=0.2, zorder=zorder)

                        # Plot photometry as well
                        if plot_phot:
                            phot = np.asarray([i.value if hasattr(i, 'unit') else i \
                                               for i in v.get('SED_phot_' + ('app' if app else 'abs'))])
                            ax.errorbar(phot[0], phot[1] * norm, yerr=phot[2] * norm, marker='o', markersize=7,
                                        markeredgewidth=1, \
                                        markeredgecolor='k', ls='none', color=color, zorder=zorder)

                        # Print the results and store the count and max and min y-axis values
                        ylims += list(hiSNRflux)
                        plots.append(
                            ['{} ({}) {}'.format(k, v.get('spectral_type'), '{} K'.format(Teff) if Teff else '-'),
                             color, 1, \
                             [k, v.get('spectral_type'), v.get('teff') or '-']])

                        # Advance the index for manual colors
                        cidx += 1

                    except IOError:
                        pass
            except IOError:
                pass

        # Show bounds on wavelength range used to normalize
        # if norm_to: ax.axvline(x=norm_to[0], color='0.8'), ax.axvline(x=norm_to[1], color='0.8')

        if any(plots):

            # Sort the list by spectral type then Teff
            plots = sorted(plots, key=lambda x: x[-1][-1], reverse=True)
            to_print = [i.pop() for i in plots]

            # Labels
            if not overplot:
                plt.rc('text', usetex=True, fontsize=fontsize)
                ax.set_ylabel(ylabel or r'$f_\lambda$' + (r'$/F_\lambda ({}-{} \mu m)$'.format(
                    *norm_to) if norm_to else r'$(\lambda) [erg s^{-1} cm^{-2} A^{-1}]$'))
                ax.set_xlabel(r'$\lambda [\mu m]$')

            # Format y-axis
            ax.set_yscale('log'), ax.set_xscale('log')
            Y = (min(ylims) * 0.6, max(ylims) * 1.4)

            for x in highlight:
                ax.fill_between(x, [Y[0]] * 2, [Y[1]] * 2, color='k', alpha=0.1, zorder=zorder)

            ax.set_ylim(Y), ax.set_xlim(um)

            # Plot the colorbar
            if cbar and not overplot:
                C = fig.colorbar(cbar)
                C.ax.set_ylabel(r'$T_{eff}(K)$')

            # Draw the legend
            if legend != 'None':
                labels, colors, sizes = zip(*plots)
                u.manual_legend(labels, colors, fontsize=fontsize - 6, sizes=sizes, overplot=ax,
                                markers=['-'] * len(labels), styles=['l'] * len(labels), loc=legend)

            # Print the results, save the figure, and return the axes
            result = np.asarray([[n + 1] + r for n, r in enumerate(to_print)])
            result = at.Table(result, names=['#','Name','SpT','Teff'])
            result.pprint()

            if save: plt.savefig(save)
            return ax, result

        else:
            print('No spectra fulfilled that criteria.')
            return
