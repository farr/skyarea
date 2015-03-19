#!/usr/bin/env python
from __future__ import print_function

from optparse import OptionParser
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import table
from glue.ligolw import utils
from lalinference import fits
import healpy as hp
import matplotlib as mpl
import numpy as np
import os
import pickle
import sky_area.sky_area_clustering as sac

mpl.use('Agg')
import matplotlib.pyplot as pp


class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass

lsctables.use_in(LIGOLWContentHandler)

def plot_skymap(output, skypost, pixresol=np.pi/180.0, nest=True):
    nside = 1
    while hp.nside2resol(nside) > pixresol:
        nside *= 2

    thetas, phis = hp.pix2ang(nside, np.arange(hp.nside2npix(nside), dtype=np.int), nest=nest)
    pixels = np.column_stack((phis, np.pi/2.0 - thetas))

    pix_post = skypost.posterior(pixels)

    pp.clf()
    hp.mollview(pix_post, nest=nest)
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

def save_areas(output, skypost, sim_id, ra, dec, cls=[0.5, 0.75, 0.9]):
    if sim_id is None or ra is None or dec is None:
        p_value = 0.0
        levels = cls
        areas = skypost.sky_area(cls)
        areas = np.concatenate((areas, [0.0]))
    else:
        p_value = skypost.p_values(np.array([[ra,dec]]))[0]
        levels = np.concatenate((cls, [p_value]))
        areas = skypost.sky_area(levels)

    rad2deg = 180.0/np.pi

    # Final areas in degrees
    areas = areas*rad2deg*rad2deg

    str_cls = ['area({0:d})'.format(int(round(100.0*cl))) for cl in cls]

    with open(output, 'w') as out:
        print(
            'simulation_id', 'p_value', 'searched_area', *str_cls,
            sep='\t', file=out)
        print(
            str(sim_id), p_value, areas[-1], *areas[:-1],
            sep='\t', file=out)

if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option('--outdir', help='output directory', default='.')
    parser.add_option('--samples', help='posterior samples file')

    parser.add_option('--inj', help='injection XML')
    parser.add_option('--eventnum', default=0, type='int', help='event number [default: %default]')

    parser.add_option('--loadpost', help='filename for pickled posterior state')

    parser.add_option('--maxpts', type='int', help='maximum number of posterior points to use')

    parser.add_option('--trials', type='int', default=50, help='maximum number of trials to build sky posterior [default: %default]')

    parser.add_option('--noskyarea', action='store_true', default=False, help='turn off sky area computation')

    parser.add_option('--enable-distance-map', action='store_true', default=False, help='enable output of healpy map of distance mean and s.d.')

    parser.add_option('--nside', type=int, default=512, help='HEALPix resolution [default: %default]')

    parser.add_option('--objid', help='event ID to store in FITS header')

    parser.add_option('--seed', type=int, default=None, help='use specified random seed')

    (args, remaining) = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    data = np.recfromtxt(args.samples, names=True)
    pts = np.column_stack((data['ra'], data['dec']))

    if args.maxpts is not None:
        pts = np.random.permutation(pts)[:args.maxpts, :]

    if args.loadpost is None:
        for i in range(args.trials):
            try:
                skypost = sac.ClusteredSkyKDEPosterior(pts)
                break
            except:
                skypost = None
                continue
        if skypost is None:
            print('Could not generate sky posterior')
            exit(1)
    else:
        with open(args.loadpost, 'r') as inp:
            skypost = pickle.load(inp)

    try:
        os.makedirs(args.outdir)
    except:
        pass

    print('pickling ...')
    with open(os.path.join(args.outdir, 'skypost.obj'), 'w') as out:
        pickle.dump(skypost, out)

    print('plotting skymap ...')
    plot_skymap(os.path.join(args.outdir, 'skymap.pdf'), skypost)

    print('plotting cluster assignments ...')
    plot_assign(os.path.join(args.outdir, 'assign.pdf'), skypost)

    if args.noskyarea:
        pass
    else:
        print('saving sky areas ...')
        if args.inj is not None:
            xmldoc = utils.load_filename(args.inj,
                                         contenthandler=LIGOLWContentHandler)
            injs = table.get_table(xmldoc,
                                   lsctables.SimInspiralTable.tableName)
            inj = injs[args.eventnum]

            save_areas(os.path.join(args.outdir, 'areas.dat'),
                       skypost,
                       inj.simulation_id, inj.longitude, inj.latitude)
        else:
            save_areas(os.path.join(args.outdir, 'areas.dat'),
                       skypost,
                       None, None, None)

    fits_nest = True

    if not args.enable_distance_map:
        fits.write_sky_map(os.path.join(args.outdir, 'skymap.fits.gz'),
                           skypost.as_healpix(args.nside, nest=fits_nest), 
                           creator=parser.get_prog_name(),
                           objid=args.objid, gps_time=data['time'].mean(),
                           nest=fits_nest)
    else:
        print('Constructing 3D clustered posterior.')
        skypost3d = sac.Clustered3DKDEPosterior(np.column_stack((data['ra'], data['dec'], data['dist'])))

        print('Producing distance map')
        map3d = skypost3d.as_healpix(args.nside, nest=fits_nest)
        mapsky = skypost.as_healpix(args.nside, nest=fits_nest)

        hpmap = np.column_stack((mapsky, map3d))
        
        hp.write_map(os.path.join(args.outdir, 'skymap.fits.gz'),
                     hpmap.T,
                     nest=fits_nest)
                         
