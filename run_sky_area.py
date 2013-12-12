#!/usr/bin/env python

from optparse import OptionParser
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import table
from glue.ligolw import utils
import healpy as hp
import matplotlib as mpl
import numpy as np
import os
import pickle
import sky_area_clustering as sac

mpl.use('Agg')
import matplotlib.pyplot as pp

def plot_skymap(output, skypost, pixresol=np.pi/180.0):
    nside = 1
    while hp.nside2resol(nside) > pixresol:
        nside *= 2

    thetas, phis = hp.pix2ang(nside, np.arange(hp.nside2npix(nside), dtype=np.int))
    pixels = np.column_stack((phis, np.pi/2.0 - thetas))

    pix_post = skypost.posterior(pixels)

    pp.clf()
    hp.mollview(pix_post)
    pp.savefig(output)

def plot_assign(output, skypost):
    k = skypost.k

    pp.clf()
    for i in range(k):
        sel = skypost.assign == i
        pp.plot(skypost.kde_pts[sel, 0], skypost.kde_pts[sel, 1], ',')

    pp.xlabel(r'$\alpha$')
    pp.ylabel(r'$\sin \delta$')

    pp.savefig(output)

def save_areas(output_areas, output_ps, skypost, sim_id, ra, dec, cls=[0.5, 0.75, 0.9]):
    if output_ps is None or sim_id is None or ra is None or dec is None:
        levels = cls
        areas = skypost.sky_areas(cls)
        np.savetxt(output_areas, np.column_stack((np.array(cls), areas[:-1])))
    else:
        p_value = skypost.p_values(np.array([[ra,dec]]))[0]
        levels = np.concatenate((cls, [p_value]))

        areas = skypost.sky_area(levels)

        np.savetxt(output_areas, np.column_stack((np.array(cls), areas[:-1])))
        with open(output_ps, 'w') as out:
            out.write('# sim_id p searched_area\n')
            out.write('{0:d} {1:g} {2:g}\n'.format(int(sim_id), p_value, areas[-1]))

if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option('--outdir', help='output directory')
    parser.add_option('--samples', help='posterior samples file')

    parser.add_option('--inj', help='injection XML')
    parser.add_option('--eventnum', default=0, type='int', help='event number')

    parser.add_option('--loadpost', help='filename for pickled posterior state')

    parser.add_option('--maxpts', type='int', help='maximum number of posterior points to use')

    parser.add_option('--trials', type='int', default=10, help='maximum number of trials to build sky posterior')

    parser.add_option('--noskyarea', action='store_true', default=False, help='turn off sky area computation')

    (args, remaining) = parser.parse_args()

    with open(args.samples, 'r') as inp:
        names = inp.readline().split()
        data = np.loadtxt(inp, dtype=[(n,np.float) for n in names])
    pts = np.column_stack((data['ra'], data['dec']))

    if args.maxpts is not None:
        pts = np.random.permutation(pts)[:args.maxpts, :]
    
    if args.loadpost is None:
        for i in range(args.trials):
            try:
                skypost = sac.ClusteredKDEPosterior(pts)
                break
            except:
                skypost = None
                continue
        if skypost is None:
            print 'Could not generate sky posterior'
            exit(1)
    else:
        with open(args.loadpost, 'r') as inp:
            skypost = pickle.load(inp)

    try:
        os.makedirs(args.outdir)
    except:
        pass

    print 'pickling ...'
    with open(os.path.join(args.outdir, 'skypost.obj'), 'w') as out:
        pickle.dump(skypost, out)

    print 'plotting skymap ...' 
    plot_skymap(os.path.join(args.outdir, 'skymap.pdf'), skypost)
    
    print 'plotting cluster assignments ...'
    plot_assign(os.path.join(args.outdir, 'assign.pdf'), skypost)
    
    if args.noskyarea:
        pass
    else:
        print 'saving sky areas ...'
        if args.inj is not None:
            injs = table.get_table(utils.load_filename(args.inj),
                                   lsctables.SimInspiralTable.tableName)
            inj = injs[args.eventnum]

            save_areas(os.path.join(args.outdir, 'areas.dat'), 
                       os.path.join(args.outdir, 'p.dat'),
                       skypost, 
                       inj.simulation_id, inj.longitude, inj.latitude)
        else:
            save_areas(os.path.join(args.outdir, 'areas.dat'),
                       None, 
                       skypost,
                       None, None, None)
