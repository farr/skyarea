from __future__ import print_function
from .eigenframe import EigenFrame
from astropy.coordinates import SkyCoord
from astropy.table import Table
import healpy as hp
import numpy as np
import numpy.linalg as nl
from scipy.stats import gaussian_kde
from lalinference.bayestar import distance, moc
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
        try:
            dists[i, :] = np.sum(dx * nl.solve(cov, dx.T).T, axis=1)
        except nl.LinAlgError:
            dists[i, :] = np.nan

    return np.nanargmin(dists, axis=0)


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

    """

    def __init__(self, pts, ntrials=5, means=None, assign=None):
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

        """
        # Find eigenbasis of sample points
        pts = SkyCoord(*pts.T, unit='rad')
        self._frame = EigenFrame.for_coords(pts)
        pts = pts.transform_to(self._frame).spherical
        self._pts = np.column_stack((pts.lon.rad, np.sin(pts.lat.rad)))

        self._ntrials = ntrials

        if means is None or assign is None:
            self._set_up_optimal_k()
        else:
            self._set_up_kmeans(means.shape[0], means, assign)

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

    def _set_up_optimal_k(self):
        self._set_up_kmeans(1)
        ks = [1]
        bics = [self._bic()]
        assigns = [self.assign]
        means = [self.means]
        for k in range(2, 41):
            self._set_up_optimal_kmeans(k, self.ntrials)
            ks.append(k)
            bics.append(self._bic())
            assigns.append(self.assign)
            means.append(self.means)
        i = np.nanargmax(bics)
        bic = bics[i]
        k = ks[i]
        assign = assigns[i]
        means = means[i]
        print('Found best k, BIC: ', k, bic)
        self._set_up_kmeans(k, means, assign)

    def _set_up_optimal_kmeans(self, k, ntrials):
        self._set_up_kmeans(k)

        best_means = self.means
        best_assign = self.assign
        best_bic = self._bic()

        print('k = ', k, 'ntrials = ', ntrials, 'bic = ', best_bic)

        for i in range(ntrials - 1):
            self._set_up_kmeans(k)
            bic = self._bic()
            assert np.isnan(bic) or bic < np.inf

            print('k = ', k, 'ntrials = ', ntrials, 'bic = ', bic)

            # The second clause is necessary to work around the corner case
            # that the initial value of best_bic is nan.
            if bic > best_bic or (np.isnan(best_bic) and not np.isnan(bic)):
                best_means = self.means
                best_assign = self.assign
                best_bic = bic

        self._set_up_kmeans(k, means=best_means, assign=best_assign)
        return best_bic

    def _set_up_kmeans(self, k, means=None, assign=None):
        self._k = k

        if means is None or assign is None:
            self._means, self._assign = k_means(self.pts, k)
        else:
            self._means = means
            self._assign = assign

        self._kdes = []
        self._weights = []
        for i in range(k):
            sel = (self.assign == i)
            pts = self.pts[sel, :]
            ndim = pts.shape[1]
            # Equivalent to but faster than len(set(pts))
            nuniq = len(np.unique(
                pts.view('V{}'.format(ndim * pts.dtype.itemsize))))
            # Skip if there are fewer unique points than dimensions
            if nuniq <= ndim:
                continue
            try:
                kde = gaussian_kde(pts.T)
            except (nl.LinAlgError, ValueError):
                # If there are fewer unique points than degrees of freedom,
                # then the KDE will fail because the covariance matrix is
                # singular. In that case, don't bother adding that cluster.
                pass
            else:
                self._kdes.append(kde)
                self._weights.append(float(np.sum(sel)))
        self._weights = np.array(self.weights)

        # Normalize the weights
        self._weights = self._weights / np.sum(self._weights)

    def posterior(self, pts):
        """Returns the clustered KDE estimate of the sky density per steradian
        at the given points in RA-DEC.

        """
        pts = SkyCoord(*pts.T, unit='rad').transform_to(self._frame).spherical
        pts = np.column_stack((pts.lon.rad, np.sin(pts.lat.rad)))
        return self._posterior(pts)

    def _posterior(self, pts):
        post = np.zeros(pts.shape[0])

        ras = pts[:, 0]
        sin_decs = pts[:, 1]

        for dra in [0.0, 2.0*np.pi, -2.0*np.pi]:
            pts = np.column_stack((ras+dra, sin_decs))
            post += self._posterior1(pts)

            pts = np.column_stack((ras+dra, 2.0 - sin_decs))
            post += self._posterior1(pts)

            pts = np.column_stack((ras+dra, -2.0 - sin_decs))
            post += self._posterior1(pts)

        return post

    def _posterior1(self, pts):
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

        pts = self.pts
        npts, ndim = pts.shape

        # The number of parameters is:
        #
        # * ndim for each centroid location
        #
        # * (ndim+1)*ndim/2 Kernel covariances for each cluster
        #
        # * one weighting factor for the cluster (minus one for the
        #   overall constraint that the weights must sum to one)
        nparams = self.k*ndim + self.k*((ndim+1)*(ndim)/2) + self.k - 1

        return np.sum(np.log(self._posterior(pts))) - nparams/2.0*np.log(npts)

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

    def as_healpix(self):
        """Returns a healpix map of the posterior density."""
        post, nside, ipix = zip(*self._bayestar_adaptive_grid())
        post = np.asarray(list(post))
        nside = np.asarray(list(nside))
        ipix = np.asarray(list(ipix))

        # Make sure that sky map is normalized (it should be already)
        post /= np.sum(post * hp.nside2pixarea(nside))

        # Convert from NESTED to UNIQ pixel indices
        order = np.log2(nside).astype(int)
        uniq = moc.nest2uniq(order.astype(np.int8), ipix.astype(np.uint64))

        # Done!
        return Table([uniq, post], names=['UNIQ', 'PROBDENSITY'])


class Clustered3DKDEPosterior(ClusteredSkyKDEPosterior):
    """Like :class:`ClusteredSkyKDEPosterior`, but clusters in 3D
    space.  Can compute volumetric posterior density (per cubic Mpc),
    and also produce Healpix maps of the mean and standard deviation
    of the log-distance.

    """

    def __init__(self, pts, ntrials=5, means=None, assign=None):
        """Initialise the posterior object.

        :param pts: A ``(npts, 3)`` shaped array.  The first column is
          RA in radians, then DEC in radians, then distance in Mpc.

        :param ntrials: The number of trials to make at each k for
          optimising the clustering.

        :param means: If given, use these means as the clustering centroids.

        :param assign: If given, use these assignments for the clustering.
        """
        self._pts = self._pts_to_xyzpts(pts)

        self._ntrials = ntrials

        if means is None or assign is None:
            self._set_up_optimal_k()
        else:
            self._set_up_kmeans(means.shape[0], means, assign)

    @staticmethod
    def _pts_to_xyzpts(pts):
        ras = pts[:, 0]
        decs = pts[:, 1]
        ds = pts[:, 2]

        xyzpts = np.column_stack((ds*np.cos(ras)*np.cos(decs),
                                  ds*np.sin(ras)*np.cos(decs),
                                  ds*np.sin(decs)))

        return xyzpts

    def _posterior(self, pts):
        post = np.zeros(pts.shape[0])

        for kde, weight in zip(self.kdes, self.weights):
            post += weight*kde(pts.T)

        return post

    def posterior(self, pts, distances=False):
        """Given an array of positions in RA, DEC, compute the marginal sky
        posterior and optinally the conditional distance parameters.

        """
        datasets = [kde.dataset for kde in self.kdes]
        inverse_covariances = [kde.inv_cov for kde in self.kdes]
        weights = self.weights

        pts = np.column_stack((pts, np.ones(len(pts))))

        prob, mean, std = np.transpose([distance.cartesian_kde_to_moments(
            n, datasets, inverse_covariances, weights)
            for n in self._pts_to_xyzpts(pts)])

        if distances:
            mu, sigma, norm = distance.moments_to_parameters(mean, std)
            return prob, mu, sigma, norm
        else:
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

        pts = self.pts
        npts, ndim = pts.shape

        # The number of parameters is:
        #
        # * ndim for each centroid location
        #
        # * (ndim+1)*ndim/2 Kernel covariances for each cluster
        #
        # * one weighting factor for the cluster (minus one for the
        #   overall constraint that the weights must sum to one)
        nparams = self.k*ndim + self.k*((ndim+1)*(ndim)/2) + self.k - 1

        return (np.sum(np.log(self.posterior_cartesian(pts))) -
                nparams/2.0*np.log(npts))

    def as_healpix(self):
        """Returns a healpix map of the posterior density."""
        _, nside, ipix = zip(*self._bayestar_adaptive_grid())
        nside = np.asarray(list(nside))
        ipix = np.asarray(list(ipix))

        # Compute marginal probability and distance parameters.
        theta, phi = hp.pix2ang(nside, ipix, nest=True)
        pts = np.column_stack((phi, 0.5 * np.pi - theta))
        post, mu, sigma, norm = self.posterior(pts, distances=True)

        # Make sure that sky map is normalized (it should be already)
        post /= np.sum(post * hp.nside2pixarea(nside))

        # Convert from NESTED to UNIQ pixel indices
        order = np.log2(nside).astype(int)
        uniq = moc.nest2uniq(order.astype(np.int8), ipix.astype(np.uint64))

        # Done!
        return Table([uniq, post, mu, sigma, norm],
                     names=['UNIQ', 'PROBDENSITY', 'DISTMU', 'DISTSIGMA',
                            'DISTNORM'])
