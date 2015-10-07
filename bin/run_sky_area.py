#!/usr/bin/env python
from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')

from optparse import OptionParser
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import table
from glue.ligolw import utils
import lalinference.cmap
from lalinference import fits
from lalinference import plot
import healpy as hp
import numpy as np
import os
import pickle
import sky_area.sky_area_clustering as sac

import matplotlib.pyplot as pp

class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass

lsctables.use_in(LIGOLWContentHandler)

def plot_skymap(output, skypost, pixresol=np.pi/180.0, nest=True,inj=None, fast=True):
    nside = 1
    while hp.nside2resol(nside) > pixresol:
        nside *= 2

    pix_post = skypost.as_healpix(nside, nest=nest, fast=fast)

    fig = pp.figure(frameon=False)
    ax = pp.subplot(111, projection='astro mollweide')
    ax.cla()
    ax.grid()
    plot.healpix_heatmap(pix_post, nest=nest, vmin=0.0, vmax=np.max(pix_post), cmap=pp.get_cmap('cylon'))

    if inj is not None:
        # If using an injection file, also plot an X at the true position
        pp.plot(inj['ra'], inj['dec'], 'kx', ms=30, mew=1)

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

def save_areas(output, skypost, sim_id, ra, dec, cls=[0.5, 0.75, 0.9], fast=True):

    if sim_id is None or ra is None or dec is None:
        p_value = 0.0
        levels = cls
        areas = skypost.sky_area(cls, fast=fast)
        areas = np.concatenate((areas, [0.0]))
    else:
        p_value = skypost.p_values(np.array([[ra,dec]]))[0]
        levels = np.concatenate((cls, [p_value]))
        areas = skypost.sky_area(levels, fast=fast)

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

    parser.add_option('--fitsoutname', help='filename for the FITS file', default='skymap.fits.gz')

    parser.add_option('--pdf', action='store_true', default=False, help='output plots in PDF format [default: PNG]')

    parser.add_option('--inj', help='injection XML')
    parser.add_option('--eventnum', default=0, type='int', help='event number [default: %default]')

    parser.add_option('--loadpost', help='filename for pickled posterior state')

    parser.add_option('--maxpts', type='int', help='maximum number of posterior points to use')

    parser.add_option('--trials', type='int', default=50, help='maximum number of trials to build sky posterior [default: %default]')

    parser.add_option('--slowskyarea', default=False, action='store_true', help='use a much slower but robust sky area algorithm')

    parser.add_option('--slowsmoothskymaps', default=False, action='store_true', help='use a faster algorithm for producing skymaps (that are "blocky")')

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

    # First check if injection file is given and fill and auxiliary dictionary
    injpos=None
    if args.inj is not None:
      xmldoc = utils.load_filename(args.inj,
                                    contenthandler=LIGOLWContentHandler)
      try:
        print('Checking if using a sim_inspiral table...')
        injs = table.get_table(xmldoc,
                   lsctables.SimInspiralTable.tableName)
        inj = injs[args.eventnum]
        injpos={'ra':inj.longitude,'dec':inj.latitude,'id':inj.simulation_id}
        print(' yes')
      except:
        print('Checking if using a sim_burst table...')
        injs = table.get_table(xmldoc,
                     lsctables.SimBurstTable.tableName)
        inj = injs[args.eventnum]
        injpos={'ra':inj.ra,'dec':inj.dec,'id':inj.simulation_id}
        print(' yes')

    print('plotting skymap ...')
    if args.pdf:
        skymap_out = os.path.join(args.outdir, 'skymap.pdf')
    else:
        skymap_out = os.path.join(args.outdir, 'skymap.png')
    plot_skymap(skymap_out, skypost,inj=injpos, fast=not(args.slowsmoothskymaps))

    print('plotting cluster assignments ...')
    if args.pdf:
        assign_out = os.path.join(args.outdir, 'assign.pdf')
    else:
        assign_out = os.path.join(args.outdir, 'assign.png')
    plot_assign(assign_out, skypost)

    print('saving sky areas ...')
    if injpos is not None:
        save_areas(os.path.join(args.outdir, 'areas.dat'),
                   skypost,
                   injpos['id'], injpos['ra'], injpos['dec'], fast=not(args.slowskyarea))

    else:
        save_areas(os.path.join(args.outdir, 'areas.dat'),
                   skypost,
                   None, None, None, fast=not(args.slowskyarea))

    fits_nest = True

    if not args.enable_distance_map:
        hpmap = skypost.as_healpix(args.nside, nest=fits_nest, fast=not(args.slowsmoothskymaps))
    else:
        print('Constructing 3D clustered posterior.')
        try:
            xyz = np.column_stack((data['ra'], data['dec'], data['dist']))
        except ValueError:
            print("ERROR, cannot use skypost3d with LIB output. Exiting..\n")
            import sys
            sys.exit(1)
        skypost3d = sac.Clustered3DKDEPosterior(xyz)

        print('pickling ...')
        with open(os.path.join(args.outdir, 'skypost3d.obj'), 'w') as out:
            pickle.dump(skypost3d, out)

        print('Producing distance map')
        hpmap = skypost3d.as_healpix(args.nside, nest=fits_nest)
    names=data.dtype.names 
    if 'time' in names:
      gps_time=data['time'].mean()
    elif 'time_mean' in names:
      gps_time=data['time_mean'].mean()
    elif 'time_maxl' in names:
      gps_time=data['time_maxl'].mean()
    else:
      print("Cannot find time, time_mean, or time maxl variable in posterior. Not saving sky_pos obj.\n")
      exit(0)

    fits.write_sky_map(os.path.join(args.outdir, args.fitsoutname),
                       hpmap, creator=parser.get_prog_name(),
                       objid=args.objid, gps_time=gps_time,
                       nest=fits_nest)
