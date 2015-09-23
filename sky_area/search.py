"""Utilities for computing useful quantities for searches on the sky.

"""

import healpy as hp
import healpy.sphtfunc as hps
import numpy as np

def _find_nside(beam, pix_per_beam):
    nside = 4
    while beam/hp.nside2resol(nside) < pix_per_beam:
        nside *= 2

    return nside

def search_map(ras, decs, beam, nest=True, pix_per_beam=10):
    """Returns a healpix map optimised for searching on the sky.  It
    represents the Gaussian-beam convolved posterior.

    :param ras: RA posterior samples.

    :param decs: Corresponding DEC samples.

    :param beam: The beam FWHM in radians.

    :param nest: Whether to output the map in nested (default) or ring
      pixel ordering.

    :param pix_per_beam: The number of pixels in the output map per
      beam (default 10).

    :return: An array representing the (unnormalised) posterior
      convolved with a Gaussian beam of the given size.

    """

    nside = _find_nside(beam, pix_per_beam)

    thetas = np.pi/2.0 - decs

    # Create the map in ring coordinates first.
    hmap = np.bincount(hp.ang2pix(nside, thetas, ras))
    if hmap.shape[0] < hp.nside2npix(nside):
        hmap = np.concatenate((hmap, np.zeros(hp.nside2npix(nside)-hmap.shape[0])))

    hmap = hmap / float(thetas.shape[0]) / hp.nside2pixarea(nside)

    chmap = hps.smoothing(hmap, fwhm=beam, pol=False)

    if nest:
        chmap = hp.reorder(chmap, r2n=True)
    return chmap
