from __future__ import print_function
import warnings
import bisect as bs
import healpy as hp
import numpy as np
import numpy.linalg as nl
import scipy.integrate as si
from scipy.stats import gaussian_kde
from lalinference.bayestar import distance
from lalinference.healpix_tree import (
    HEALPIX_MACHINE_ORDER, HEALPIX_MACHINE_NSIDE, HEALPixTree)


def km_assign(mus, cov, pts):
    """Implements the assignment step in the k-means algorithm.  Given a
    set of centers, ``mus``, a covariance matrix used to produce a
    metric on the space, ``cov``, and a set of points, ``pts`` (shape
    ``(npts, ndim)``), assigns each point to its nearest center,
    returning an array of indices of shape ``(npts,)`` giving the
    assignments.

    """
    k = mus.shape[0]
    n = pts.shape[0]

    dists = np.zeros((k, n))

    for i, mu in enumerate(mus):
        dx = pts - mu
        dists[i, :] = np.sum(dx * nl.solve(cov, dx.T).T, axis=1)

    return np.argmin(dists, axis=0)


def km_centroids(pts, assign, k):
    """Implements the centroid-update step of the k-means algorithm.
    Given a set of points, ``pts``, of shape ``(npts, ndim)``, and an
    assignment of each point to a region, ``assign``, and the number
    of means, ``k``, returns an array of shape ``(k, ndim)`` giving
    the centroid of each region.

    """

    mus = np.zeros((k, pts.shape[1]))
    for i in range(k):
        sel = assign == i
        if np.sum(sel) > 0:
            mus[i, :] = np.mean(pts[sel, :], axis=0)
        else:
            mus[i, :] = pts[np.random.randint(pts.shape[0]), :]

    return mus


def k_means(pts, k):
    """Implements k-means clustering on the set of points.

    :param pts: Array of shape ``(npts, ndim)`` giving the points on
      which k-means is to operate.

    :param k: Positive integer giving the number of regions.

    :return: ``(centroids, assign)``, where ``centroids`` is an ``(k,
      ndim)`` array giving the centroid of each region, and ``assign``
      is a ``(npts,)`` array of integers between 0 (inclusive) and k
      (exclusive) indicating the assignment of each point to a region.

    """
    assert pts.shape[0] > k, 'must have more points than means'

    cov = np.cov(pts, rowvar=0)

    mus = np.random.permutation(pts)[:k, :]
    assign = km_assign(mus, cov, pts)
    while True:
        old_mus = mus
        old_assign = assign

        mus = km_centroids(pts, assign, k)
        assign = km_assign(mus, cov, pts)

        if np.all(assign == old_assign):
            break

    return mus, assign


class ClusteredSkyKDEPosterior(object):
    r"""Represents a kernel-density estimate of a sky-position PDF that has
    been decomposed into clusters, using a different kernel for each
    cluster.

    The estimated PDF is

    .. math::

      p\left( \vec{\theta} \right) = \sum_{i = 0}^{k-1} \frac{N_i}{N}
      \sum_{\vec{x} \in C_i} N\left[\vec{x}, \Sigma_i\right]\left( \vec{\theta}
      \right)

    where :math:`C_i` is the set of points belonging to cluster
    :math:`i`, :math:`N_i` is the number of points in this cluster,
    :math:`\Sigma_i` is the optimally-converging KDE covariance
    associated to cluster :math:`i`.

    The number of clusters, :math:`k` is chosen to maximize the `BIC
    <http://en.wikipedia.org/wiki/Bayesian_information_criterion>`_
    for the given set of points being drawn from the clustered KDE.
    The points are assigned to clusters using the k-means algorithm,
    with a decorrelated metric.  The overall clustering behavior is
    similar to the well-known `X-Means
    <http://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf>`_ algorithm.

    In order to produce an unbiased estimate of credible areas, the
    algorithm follows a two-step process.  The set of input points is
    divided into two independent sets.  The first of these sets is
    used to establish a clustered KDE as described above; then the
    second set of points is ranked under this clustered KDE to
    establish a mapping from KDE contours to credible levels.  The
    different point sets are accessible as the ``self.kde_pts`` and
    ``self.ranking_pts`` arrays in the object.

    """

    def __init__(self, pts, ntrials=5, means=None, assign=None, acc=1e-2):
        """Set up the posterior with the given RA-DEC points.

        :param pts: The sky points, in RA-DEC coordinates.

        :param ntrials: If optimizing the assignments of points to
          clusters, this many trials will be used for each k (cluster
          number) to determine the optimal clustering.

        :param means: If not ``None``, use these points as centroids
          of the clusters.

        :param assign: If not ``None``, use these assignments into
          clusters.  If either ``means`` or ``assign`` is ``None``,
          then the choice of cluster number (k) and assignments are
          optimized using a BIC criterion on the model that ``pts``
          are drawn from the given clustered KDE.

        :param acc: The (relative) accuracy with which to compute sky
          areas.

        """
        self._acc = acc

        pts = pts.copy()
        pts[:, 1] = np.sin(pts[:, 1])
        self._pts = pts

        ppts = np.random.permutation(pts)

        self._kde_pts = ppts[::2]
        self._ranking_pts = ppts[1::2]
        self._ntrials = ntrials

        if means is None or assign is None:
            self._set_up_optimal_k()
        else:
            self._set_up_kmeans(means.shape[0], means, assign)

        self._set_up_greedy_order()

    @property
    def acc(self):
        """Integration accuracy for sky/searched areas.

        """
        return self._acc

    @acc.setter
    def acc(self, a):
        self._acc = a

    @property
    def ntrials(self):
        """Returns the number of trials at each k over which the cluster
        assignments have been optimized.

        """
        return self._ntrials

    @property
    def pts(self):
        r"""Returns the points in :math:`(\alpha, \sin(\delta))` space.

        """
        return self._pts

    @property
    def kde_pts(self):
        """Return the subset of points used to construct the KDE.

        """
        return self._kde_pts

    @property
    def ranking_pts(self):
        """Return the set of points used that are ranked under the KDE to
        establish credible levels.

        """
        return self._ranking_pts

    @property
    def k(self):
        """Returns the optimized number of clusters.

        """
        return self._k

    @property
    def assign(self):
        """Returns the cluster assignment number for each point.

        """
        return self._assign

    @property
    def means(self):
        """Returns the cluster centroids.

        """
        return self._means

    @property
    def kdes(self):
        """Returns the scipy KDE object associated with each cluster.

        """
        return self._kdes

    @property
    def weights(self):
        """Returns the weight assigned to each cluster's KDE in the final
        posterior.

        """
        return self._weights

    @property
    def greedy_order(self):
        """Returns the ordering of ``self.ranking_pts`` from highest to lowest
        posterior values.

        """
        return self._greedy_order

    @property
    def greedy_posteriors(self):
        """Returns the posterior values at ``self.ranking_pts`` in greedy order.

        """
        return self._greedy_posteriors

    def _set_up_optimal_k(self):
        self._set_up_kmeans(1)
        low_bic = self._bic()
        low_assign = self.assign
        low_means = self.means
        low_k = 1

        mid_bic = self._set_up_optimal_kmeans(2, self.ntrials)
        mid_assign = self.assign
        mid_means = self.means
        mid_k = 2

        high_bic = self._set_up_optimal_kmeans(4, self.ntrials)
        high_assign = self.assign
        high_means = self.means

        low_k, mid_k, high_k = 1, 2, 4

        while high_bic > mid_bic:
            print('extending ks: ', (low_k, mid_k, high_k))
            print('with bics: ', (low_bic, mid_bic, high_bic))

            low_k, mid_k = mid_k, high_k
            low_bic, mid_bic = mid_bic, high_bic
            low_means, mid_means = mid_means, high_means
            low_assign, mid_assign = mid_assign, high_assign

            high_k = 2*mid_k
            high_bic = self._set_up_optimal_kmeans(high_k, self.ntrials)
            high_means = self.means
            high_assign = self.assign

        while high_k - low_k > 2:
            print('shrinking ks: ', (low_k, mid_k, high_k))
            print('with bics: ', (low_bic, mid_bic, high_bic))

            if high_k - mid_k > mid_k - low_k:
                k = mid_k + (high_k - mid_k)/2
                bic = self._set_up_optimal_kmeans(k, self.ntrials)
                means = self.means
                assign = self.assign

                if bic > mid_bic:
                    low_k, mid_k = mid_k, k
                    low_bic, mid_bic = mid_bic, bic
                    low_means, mid_means = mid_means, means
                    low_assign, mid_assign = mid_assign, assign
                else:
                    high_k = k
                    high_bic = bic
                    high_means = means
                    high_assign = assign
            else:
                k = low_k + (mid_k - low_k)/2
                bic = self._set_up_optimal_kmeans(k, self.ntrials)
                means = self.means
                assign = self.assign

                if bic > mid_bic:
                    mid_k, high_k = k, mid_k
                    mid_bic, high_bic = bic, mid_bic
                    mid_means, high_means = means, mid_means
                    mid_assign, high_assign = assign, mid_assign
                else:
                    low_k = k
                    low_bic = bic
                    low_means = means
                    low_assign = assign

        print('Found best k, BIC: ', mid_k, mid_bic)
        self._set_up_kmeans(mid_k, mid_means, mid_assign)

    def _set_up_optimal_kmeans(self, k, ntrials):
        best_bic = np.NINF

        for i in range(ntrials):
            self._set_up_kmeans(k)
            bic = self._bic()

            print('k = ', k, 'ntrials = ', ntrials, 'bic = ', bic)

            if bic >= best_bic:
                best_means = self.means
                best_assign = self.assign
                best_bic = bic

        self._set_up_kmeans(k, means=best_means, assign=best_assign)
        return best_bic

    def _set_up_kmeans(self, k, means=None, assign=None):
        self._k = k

        if means is None or assign is None:
            self._means, self._assign = k_means(self.kde_pts, k)
        else:
            self._means = means
            self._assign = assign

        self._kdes = []
        self._weights = []
        ndim = self.kde_pts.shape[1]
        for i in range(k):
            sel = (self.assign == i)
            # If there are fewer points than degrees of freedom, then don't
            # bother adding a KDE for that cluster; its covariance would be
            # singular.
            if np.sum(sel) > ndim:
                self._kdes.append(gaussian_kde(self.kde_pts[sel, :].T))
                self._weights.append(float(np.sum(sel)))
        self._weights = np.array(self.weights)

        # Normalize the weights
        self._weights = self._weights / np.sum(self._weights)

    def _set_up_greedy_order(self):
        pts = self.ranking_pts.copy()
        pts[:, 1] = np.arcsin(pts[:, 1])

        posts = self.posterior(pts)
        self._greedy_order = np.argsort(posts)[::-1]
        self._greedy_posteriors = posts[self.greedy_order]

    def posterior(self, pts):
        """Returns the clustered KDE estimate of the sky density per steradian
        at the given points in RA-DEC.

        """
        pts = pts.copy()
        pts = np.atleast_2d(pts)
        pts[:, 1] = np.sin(pts[:, 1])

        post = np.zeros(pts.shape[0])

        ras = pts[:, 0]
        sin_decs = pts[:, 1]

        for dra in [0.0, 2.0*np.pi, -2.0*np.pi]:
            pts = np.column_stack((ras+dra, sin_decs))
            post += self._posterior(pts)

            pts = np.column_stack((ras+dra, 2.0 - sin_decs))
            post += self._posterior(pts)

            pts = np.column_stack((ras+dra, -2.0 - sin_decs))
            post += self._posterior(pts)

        return post

    def _posterior(self, pts):
        post = np.zeros(pts.shape[0])

        for kde, weight in zip(self.kdes, self.weights):
            post += weight*kde(pts.T)

        return post

    def __call__(self, pts):
        """Synonym for ``self.posterior()``.

        """
        return self.posterior(pts)

    def _bic(self):
        """Returns the BIC for the point set being drawn from the clustered
        KDE.

        """

        npts, ndim = self.kde_pts.shape

        # The number of parameters is:
        #
        # * ndim for each centroid location
        #
        # * (ndim+1)*ndim/2 Kernel covariances for each cluster
        #
        # * one weighting factor for the cluster (minus one for the
        #   overall constraint that the weights must sum to one)
        nparams = self.k*ndim + self.k*((ndim+1)*(ndim)/2) + self.k - 1

        pts = self.kde_pts.copy()
        pts[:, 1] = np.arcsin(pts[:, 1])

        return np.sum(np.log(self.posterior(pts))) - nparams/2.0*np.log(npts)

    def _split_range(self, n, nmax=100000):
        if n < nmax:
            return [(0, n)]
        else:
            lows = list(range(0, n, nmax))
            highs = lows[1:]
            highs.append(n)

            return list(zip(lows, highs))

    def _adaptive_grid(self, order=HEALPIX_MACHINE_ORDER):
        theta = np.arccos(self.pts[:, 1])
        phi = self.pts[:, 0]
        ipix = hp.ang2pix(
            HEALPIX_MACHINE_NSIDE, theta, phi, nest=True).astype(np.uint64)
        return HEALPixTree(ipix, max_samples_per_pixel=1, max_order=order)

    def _bayestar_adaptive_grid(self, top_nside=16, rounds=8):
        """Implement of the BAYESTAR adaptive mesh refinement scheme as
        described in Section VI of Singer & Price 2016, PRD, 93, 024013
        (http://dx.doi.org/10.1103/PhysRevD.93.024013).

        FIXME: Consider refactoring BAYESTAR itself to perform the adaptation
        step in Python.
        """
        top_npix = hp.nside2npix(top_nside)
        nrefine = top_npix // 4
        cells = zip([0] * nrefine, [top_nside // 2] * nrefine, range(nrefine))
        for _ in range(rounds - 1):
            cells = sorted(cells, key=lambda (p, n, i): p / n**2)
            new_nside, new_ipix = np.transpose([
                (nside * 2, ipix * 4 + i)
                for _, nside, ipix in cells[-nrefine:] for i in range(4)])
            theta, phi = hp.pix2ang(new_nside, new_ipix, nest=True)
            ra = phi
            dec = 0.5 * np.pi - theta
            p = self.posterior(np.column_stack((ra, dec)))
            cells[-nrefine:] = zip(p, new_nside, new_ipix)
        return cells

    def _as_healpix_slow(self, nside, nest=True):
        npix = hp.nside2npix(nside)
        thetas, phis = hp.pix2ang(nside, np.arange(npix), nest=nest)
        pixels = np.column_stack((phis, np.pi/2.0 - thetas))
        pixel_posts = self.posterior(pixels)
        return pixel_posts / np.sum(pixel_posts)

    def _as_healpix_fast(self, nside, nest=True):
        """Returns a healpix map of the posterior density, by default in
        nested order.

        """
        cell_post, cell_nside, cell_ipix = zip(*self._bayestar_adaptive_grid())

        max_nside = max(cell_nside)
        max_npix = hp.nside2npix(max_nside)
        map = np.empty(max_npix)

        for cpost, cnside, cipix in zip(cell_post, cell_nside, cell_ipix):
            cnpix = hp.nside2npix(cnside)
            cnpts = (max_npix // cnpix)
            ipix0 = cipix * cnpts
            ipix1 = (cipix + 1) * cnpts
            map[ipix0:ipix1] = cpost

        # FIXME: we're ignoring the requested value of nside
        warnings.warn('Ignoring user-provided value of nside because it is not '
                      'currently implemented for the fast pixelization.',
                      RuntimeWarning, stacklevel=2)

        if nest:
            pass  # Map is already in nested order
        else:
            map = hp.pixelfunc.reorder(map, n2r=True)

        return map / np.sum(map)

    def as_healpix(self, nside, nest=True, fast=True):
        """Return a healpix map of the posterior at the given resolution.

        :param nside: The resolution parameter.

        :param nest: If ``True``, map is in nested order.

        :param fast: If ``True`` produce a map more quickly, at the
          cost of some pixellation.

        """
        if fast:
            return self._as_healpix_fast(nside, nest=nest)
        else:
            return self._as_healpix_slow(nside, nest=nest)

    def _fast_area_within(self, levels):
        grid = self._adaptive_grid()

        nside, ipix = np.transpose(list(grid.visit(extra=False)))
        theta, phi = hp.pix2ang(nside, ipix, nest=True)
        ra = phi
        dec = 0.5 * np.pi - theta
        plevels = self.posterior(np.column_stack((ra, dec)))
        pareas = hp.nside2pixarea(nside)

        areas = []
        for l in levels:
            areas.append(np.sum(pareas[plevels >= l]))

        return np.array(areas)

    def _area_within_nside(self, levels, nside):
        npix = hp.nside2npix(nside)
        pixarea = hp.nside2pixarea(nside)

        areas = 0.0
        for low, high in self._split_range(npix):
            thetas, phis = hp.pix2ang(
                nside, np.arange(low, high, dtype=np.int))
            pixels = np.column_stack((phis, np.pi/2.0 - thetas))

            pixel_posts = self.posterior(pixels)

            sub_areas = np.array(
                [pixarea*np.sum(pixel_posts > l) for l in levels])
            areas = areas + sub_areas

        return areas

    def _area_within(self, levels, nside_max=512):
        levels = np.atleast_1d(levels)

        nside = 1
        old_areas = np.zeros(levels.shape[0])
        while True:
            nside *= 2
            areas = self._area_within_nside(levels, nside)

            extrap_areas = (4.0*areas - old_areas)/3.0

            error = np.abs((areas - extrap_areas)/extrap_areas)

            print('Calculated sky area at nside = ', nside)
            print('Areas are ', extrap_areas)
            print()

            if np.all(areas > 0) and np.all(error < self.acc):
                return extrap_areas
            elif nside >= nside_max:
                print('Ending sky area calculation at nside = ', nside)
                return extrap_areas
            else:
                old_areas = areas

    def sky_area(self, cls, fast=True):
        """Returns the sky area occupied by the given list of credible levels.
        If ``fast``, then use a fast algorithm that is usually
        accurate but not guaranteed to converge to the correct answer.

        """
        cls = np.atleast_1d(cls)
        idxs = [int(round(cl*self.ranking_pts.shape[0])) for cl in cls]
        missed = False
        if idxs[-1] == len(self.greedy_posteriors):
            # this can happen if the injected position is totally missed
            idxs[-1] -= 1
            missed = True

        post_levels = [self.greedy_posteriors[i] for i in idxs]

        if fast:
            out = self._fast_area_within(post_levels)
        else:
            out = self._area_within(post_levels)

        if missed:
            # if missed set the searched are to be the whole sky
            out[-1] = 4*np.pi
        return out

    def searched_area(self, pts, fast=True):
        """Returns the sky area that must be searched using a greedy algorithm
        before encountering the given points in the sky.  If ``fast``,
        then use a fast algorithm that is usually accurate but not
        guaranteed to converge to the correct answer.

        """
        post_levels = self.posterior(pts)

        if fast:
            return self._fast_area_within(post_levels)
        else:
            return self._area_within(post_levels)

    def p_values(self, pts):
        """Returns the posterior greedy p-values (quantile in the posterior
        distribution) for the given points.

        """

        post_levels = self.posterior(pts)

        # Need smallest to largest, not other way around
        greedy_levels = self.greedy_posteriors[::-1]
        n = greedy_levels.shape[0]

        indexes = []
        for pl in post_levels:
            indexes.append(bs.bisect(greedy_levels, pl))

        return 1.0 - np.array(indexes) / float(n)


class Clustered3DKDEPosterior(ClusteredSkyKDEPosterior):
    """Like :class:`ClusteredSkyKDEPosterior`, but clusters in 3D
    space.  Can compute volumetric posterior density (per cubic Mpc),
    and also produce Healpix maps of the mean and standard deviation
    of the log-distance.  Does not currently produce credible volumes.

    """

    def __init__(self, pts, ntrials=5, means=None, assign=None, acc=1e-2):
        """Initialise the posterior object.

        :param pts: A ``(npts, 3)`` shaped array.  The first column is
          RA in radians, then DEC in radians, then distance in Mpc.

        :param ntrials: The number of trials to make at each k for
          optimising the clustering.

        :param means: If given, use these means as the clustering centroids.

        :param assign: If given, use these assignments for the clustering.
        """
        self._acc = acc

        pts = pts.copy()
        _pts = pts.copy()[:, :-1]
        _pts[:, 1] = np.sin(_pts[:, 1])
        self._pts = _pts

        indices = np.arange(len(pts))
        np.random.shuffle(indices)

        self._kde_pts = self._pts_to_xyzpts(pts[indices][::2])
        self._ranking_pts = _pts[indices][1::2]
        self._ntrials = ntrials

        if means is None or assign is None:
            self._set_up_optimal_k()
        else:
            self._set_up_kmeans(means.shape[0], means, assign)

        self._set_up_greedy_order()

    @staticmethod
    def _pts_to_xyzpts(pts):
        ras = pts[:, 0]
        decs = pts[:, 1]
        ds = pts[:, 2]

        xyzpts = np.column_stack((ds*np.cos(ras)*np.cos(decs),
                                  ds*np.sin(ras)*np.cos(decs),
                                  ds*np.sin(decs)))

        return xyzpts

    def posterior(self, pts):
        """Given an array of positions in RA, DEC, dist, compute the 3D
        volumetric posterior density (per Mpc) at those points.

        """

        datasets = [kde.dataset for kde in self.kdes]
        inverse_covariances = [kde.inv_cov for kde in self.kdes]
        weights = self.weights

        pts = np.column_stack((pts, np.ones(len(pts))))

        prob, _, _ = np.transpose([distance.cartesian_kde_to_moments(
            n, datasets, inverse_covariances, weights)
            for n in self._pts_to_xyzpts(pts)])

        return prob

    def posterior_cartesian(self, pts):
        return self._posterior(pts)

    def posterior_spherical(self, pts):
        return self.posterior_cartesian(self._pts_to_xyzpts(pts))

    def conditional_posterior(self, ra, dec, ds):
        """Returns a slice through the smoothed posterior at the given
        RA, DEC as a function of distance.  WARNING: the returned
        posterior is not normalised.

        """

        ds = np.atleast_1d(ds)
        ras = ra + 0*ds
        decs = dec + 0*ds

        pts = np.column_stack((ras, decs, ds))

        return self.posterior_spherical(pts)

    def _bic(self):
        """Returns the BIC for the point set being drawn from the clustered
        KDE.

        """

        npts, ndim = self.kde_pts.shape

        # The number of parameters is:
        #
        # * ndim for each centroid location
        #
        # * (ndim+1)*ndim/2 Kernel covariances for each cluster
        #
        # * one weighting factor for the cluster (minus one for the
        #   overall constraint that the weights must sum to one)
        nparams = self.k*ndim + self.k*((ndim+1)*(ndim)/2) + self.k - 1

        pts = self.kde_pts.copy()

        return (np.sum(np.log(self.posterior_cartesian(pts))) -
                nparams/2.0*np.log(npts))

    def _as_healpix_slow(self, nside, nest=True):
        r"""Returns a healpix map with the mean and standard deviations
        of :math:`d` for any pixel containing at least one posterior
        sample.

        """
        npix = hp.nside2npix(nside)
        pixarea = hp.nside2pixarea(nside)
        datasets = [kde.dataset for kde in self.kdes]
        inverse_covariances = [kde.inv_cov for kde in self.kdes]
        weights = self.weights

        # Compute marginal probability, conditional mean, and conditional
        # standard deviation in all directions.
        prob, mean, std = np.transpose([distance.cartesian_kde_to_moments(
            np.asarray(hp.pix2vec(nside, ipix, nest=nest)),
            datasets, inverse_covariances, weights)
            for ipix in range(npix)])

        prob *= pixarea
        # Normalize marginal probability...
        # just to be safe. It should be normalized already.
        prob /= prob.sum()

        # Apply method of moments to find location parameter, scale parameter,
        # and normalization.
        distmu, distsigma, distnorm = distance.moments_to_parameters(mean, std)

        # Done!
        return prob, distmu, distsigma, distnorm

    def _as_healpix_fast(self, nside, nest=True):
        """Returns a healpix map of the posterior density, by default in
        nested order.

        """
        _, cell_nside, cell_ipix = zip(*self._bayestar_adaptive_grid())
        cell_nside = np.asarray(cell_nside)
        cell_ipix = np.asarray(cell_ipix)

        max_nside = max(cell_nside)
        max_npix = hp.nside2npix(max_nside)
        map = np.empty(max_npix)

        map = [np.empty(max_npix),
               np.empty(max_npix),
               np.empty(max_npix),
               np.empty(max_npix)]

        pts = np.transpose(hp.pix2vec(cell_nside, cell_ipix, nest=True))

        datasets = [kde.dataset for kde in self.kdes]
        inverse_covariances = [kde.inv_cov for kde in self.kdes]
        weights = self.weights

        # Compute marginal probability, conditional mean, and conditional
        # standard deviation in all directions.
        prob, mean, std = np.transpose([distance.cartesian_kde_to_moments(
            pt, datasets, inverse_covariances, weights)
            for pt in pts])

        # Apply method of moments to find location parameter, scale parameter,
        # and normalization.
        distmu, distsigma, distnorm = distance.moments_to_parameters(mean, std)

        for prob, distmu, distsigma, distnorm, cnside, cipix in zip(
                prob, distmu, distsigma, distnorm, cell_nside, cell_ipix):
            cnpix = hp.nside2npix(cnside)
            cnpts = (max_npix // cnpix)
            ipix0 = cipix * cnpts
            ipix1 = (cipix + 1) * cnpts
            map[0][ipix0:ipix1] = prob
            map[1][ipix0:ipix1] = distmu
            map[2][ipix0:ipix1] = distsigma
            map[3][ipix0:ipix1] = distnorm

        map[0] *= hp.nside2pixarea(max_nside)
        # Normalize marginal probability...
        # just to be safe. It should be normalized already.
        map[0] /= map[0].sum()

        # FIXME: we're ignoring the requested value of nside
        warnings.warn('Ignoring user-provided value of nside because it is not '
                      'currently implemented for the fast pixelization.',
                      RuntimeWarning, stacklevel=2)

        if nest:
            pass  # Map is already in nested order
        else:
            map = hp.pixelfunc.reorder(map, n2r=True)

        return map
