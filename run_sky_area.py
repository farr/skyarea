#!/usr/bin/env python

from argparse import ArgumentParser
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import table
from glue.ligolw import utils
import healpy as hp
import matplotlib.pyplot as pp
import numpy as np
import os
import pickle
import sky_area_clustering as sac

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
        pp.plot(skypost.pts[sel, 0], skypost.pts[sel, 1], ',')

    pp.xlabel(r'$\alpha$')
    pp.ylabel(r'$\sin \delta$')

    pp.savefig(output)

def save_areas(output, skypost, cls=[0.5, 0.75, 0.9]):
    areas = skypost.sky_area(cls)

    np.savetxt(output, np.column_stack((np.array(cls), areas)))

def save_pvalue(output, skypost, ra, dec):
    p = skypost.p_values(np.array([[ra, dec]]))

    np.savetxt(output, p)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--outdir', required=True, help='output directory')
    parser.add_argument('--samples', required=True, help='posterior samples file')

    parser.add_argument('--inj', help='injection XML')
    parser.add_argument('--eventnum', default=0, type=int, help='event number')

    parser.add_argument('--loadpost', help='filename for pickled posterior state')

    parser.add_argument('--maxpts', type=int, help='maximum number of posterior points to use')

    parser.add_argument('--noskyarea', action='store_true', help='turn off sky area computation')

    args = parser.parse_args()

    data = np.genfromtxt(args.samples, names=True)
    pts = np.column_stack((data['ra'], data['dec']))

    if args.maxpts is not None:
        pts = np.random.permutation(pts)[:args.maxpts, :]
    
    if args.loadpost is None:
        skypost = sac.ClusteredKDEPosterior(pts)
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
        save_areas(os.path.join(args.outdir, 'areas.dat'), skypost)

    if args.inj is not None:
        injs = table.get_table(utils.load_filename(args.inj),
                               lsctables.SimInspiralTable.tableName)
        inj = injs[args.eventnum]

        print 'saving injection p-value ...'
        save_pvalue(os.path.join(args.outdir, 'p.dat'), skypost, inj.longitude, inj.latitude)
