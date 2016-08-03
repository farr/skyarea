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

    :return: An array representing the posterior convolved with a
      Gaussian beam of the given size.  The array is normalised as a
      probability density per square degree.

    """

    nside = _find_nside(beam, pix_per_beam)

    thetas = np.pi/2.0 - decs

    # Create the map in ring coordinates first.
    hmap = np.bincount(hp.ang2pix(nside, thetas, ras))
    if hmap.shape[0] < hp.nside2npix(nside):
        hmap = np.concatenate(
            (hmap, np.zeros(hp.nside2npix(nside) - hmap.shape[0])))

    hmap = hmap / float(thetas.shape[0]) / hp.nside2pixarea(nside)

    chmap = hps.smoothing(hmap, fwhm=beam, pol=False)

    if nest:
        chmap = hp.reorder(chmap, r2n=True)

    norm = np.sum(chmap) * hp.nside2pixarea(nside, degrees=True)

    return chmap / norm


def search_map_searched_area_pt(smap, ra, dec, nest=True):
    """Returns the area on the sky required to be imaged in a greedy
    search according to the map ``smap`` before imaging the point
    ``(ra, dec)``.

    :param smap: The map used for the search.

    :param ra: The RA of the point of interest.

    :param dec: The DEC of the point of interest.

    :param nest: ``True`` if the map is in nested order (default).

    :return: The area (in square degrees) that must be searched when
      following a greedy algorithm before the point at ``(ra, dec)``
      is imaged.

    """

    smap = np.atleast_1d(smap)

    nside = hp.npix2nside(smap.shape[0])

    theta = np.pi/2.0 - dec

    ptind = hp.ang2pix(nside, theta, ra, nest=nest)

    ptlevel = smap[ptind]

    nabove = np.sum(smap >= ptlevel)

    return nabove*hp.nside2pixarea(nside, degrees=True)


def search_map_searched_area_cl(smap, cl):
    """Returns the area that must be searched greedily using ``smap`` to
    reach the credible level ``cl`` (fraction between 0 and 1 of the
    probability in the map).

    Note that the resulting credible area will be biased because the
    posterior pixel counts used to construct the search map are
    subject to Poisson fluctuations and we search greedily, resulting
    in the upward-fluctuations being searched first.  This bias tends
    to reduce the searched area compared to the true area that would
    be computed with perfect knowledge of the distribution on the sky.
    See https://dcc.ligo.org/LIGO-P1400054 .

    :param smap: The search map (need not be normalised).

    :param cl: Fraction (between 0 and 1) of the probability to be
      covered in the search.

    :return: The area (in square degrees) that must be searched to
      reach the desired coverage of the distribution.

    """

    smap = np.atleast_1d(smap)
    nside = hp.npix2nside(smap.shape[0])

    # Normalise the map to sum to 1:
    smap = smap / np.sum(smap)

    cum_probs = np.cumsum(np.sort(smap)[::-1])

    nsearched = np.sum(cum_probs <= cl)

    return nsearched*hp.nside2pixarea(nside, degrees=True)


def search_map_credible_level_pt(smap, ra, dec, nest=True):
    """Returns the credible level at which the given point would be found
    in a greedy search.  Note that this credible level has an
    intrinsic bias due to the combination of greedy search and Poisson
    fluctuations in the histogram counts used to make the map.  See
    https://dcc.ligo.org/LIGO-P1400054 .

    :param smap: The search map.

    :param ra: The RA of the point of interest.

    :param dec: The DEC of the point of interest.

    :param nest: ``True`` if the search map is in healpix nested order
      (default).

    :return: The fraction of the probability (between 0 and 1) that
      would be covered in a greedy search using ``smap`` before
      imaging the point at ``(ra, dec)``

    """

    smap = np.atleast_1d(smap)
    nside = hp.npix2nside(smap.shape[0])

    # Normalise
    smap = smap / np.sum(smap)

    theta = np.pi/2.0 - dec

    ptidx = hp.ang2pix(nside, theta, ra, nest=nest)
    ptden = smap[ptidx]

    return np.sum(smap[smap >= ptden])
