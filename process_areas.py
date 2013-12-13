#!/usr/bin/env python

import bz2
import glob
import numpy as np
import matplotlib.pyplot as pp
import plotutils.plotutils as pu
import scipy.stats as ss

if __name__ == '__main__':
    cls = np.array([0.5, 0.75, 0.9])
    cls_header = ['area({0:d})'.format(int(round(100.0*cl))) for cl in cls]

    rad2deg = 180.0/np.pi

    data = []
    for file in glob.glob('*/areas.dat'):
        data.append(np.recfromtxt(file, names=True))
    new_data = np.zeros(len(data), dtype=data[0].dtype)
    for i in range(len(data)):
        new_data[i] = data[i][()]
    data = new_data

    with bz2.BZ2File('2015-TaylorF2-MCMC-areas.dat.bz2', 'w') as out:
        out.write('simulation_id\tp_value\tsearched_area\t' + '\t'.join(cls_header) + '\n')
        for d in data:
            out.write('{0:s}\t{1:g}\t{2:g}\t{3:g}\t{4:g}\t{5:g}\n'.format(d['simulation_id'],
                                                                          d['p_value'],
                                                                          d['searched_area'],
                                                                          d['area50'],
                                                                          d['area75'],
                                                                          d['area90']))

    ks_stat, ks_p = ss.kstest(data['p_value'], lambda x: x)
    
    pp.clf()
    pu.plot_cumulative_distribution(data['p_value'], '-k')
    pp.plot(np.linspace(0,1,10), np.linspace(0,1,10), '--k')
    pp.xlabel(r'$p_\mathrm{inj}$')
    pp.ylabel(r'$P(p_\mathrm{inj})$')
    pp.title('2015 MCMC TaylorF2 (K-S p-value {0:g})'.format(ks_p))
    pp.savefig('2015-TaylorF2-MCMC-p-p.pdf')

    pp.clf()
    pu.plot_cumulative_distribution(data['searched_area']*rad2deg*rad2deg, '-k')
    pp.xscale('log')
    pp.xlabel(r'Searched Area (deg$^2$)')
    pp.savefig('2015-TaylorF2-MCMC-searched-area.pdf')

    pp.clf()
    pu.plot_cumulative_distribution(data['area50'], label=str('50\%'))
    pu.plot_cumulative_distribution(data['area75'], label=str('75\%'))
    pu.plot_cumulative_distribution(data['area90'], label=str('90\%'))    
    pp.xscale('log')
    pp.xlabel(r'Credible Area (deg$^2$)')
    pp.legend(loc='upper left')
    pp.savefig('2015-TaylorF2-MCMC-credible-area.pdf')
