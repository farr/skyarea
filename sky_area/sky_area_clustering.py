from __future__ import division
from .eigenframe import EigenFrame
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.utils.console import ProgressBar
from astropy.utils.misc import NumpyRNGContext
import healpy as hp
import numpy as np
import numpy.linalg as nl
from scipy.stats import gaussian_kde
from lalinference.bayestar import distance, moc
from functools import partial
from six.moves import copyreg

__all__ = ('Clustered2DSkyKDE', 'Clustered3DSkyKDE', 'Clustered2Plus1DSkyKDE')


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


def _cluster(cls, pts, trials, i, seed):
    k = i // trials
    if k == 0:
        raise ValueError('Expected at least one cluster')
    try:
        if k == 1:
            assign = np.zeros(len(pts), dtype=np.intp)
        else:
            with NumpyRNGContext(i + seed):
                _, assign = k_means(pts, k)
        obj = cls(pts, assign=assign)
    except np.linalg.LinAlgError:
        return -np.inf,
    else:
        return obj.bic, k, obj.kdes


class _mapfunc(object):

    def __init__(self, func):
        self._func = func

    def __call__(self, i_arg):
        i, arg = i_arg
        return i, self._func(arg)


class ClusteredKDE(object):

    def __init__(self, pts, max_k=40, trials=5, assign=None,
                 multiprocess=False):
        self.multiprocess = multiprocess
        if assign is None:
            print('clustering ...')
            # Make sure that each thread gets a different random number state.
            # We start by drawing a random integer s in the main thread, and
            # then the i'th subprocess will seed itself with the integer i + s.
            #
            # The seed must be an unsigned 32-bit integer, so if there are n
            # threads, then s must be drawn from the interval [0, 2**32 - n).
            seed = np.random.randint(0, 2**32 - max_k * trials)
            func = partial(_cluster, type(self), pts, trials, seed=seed)
            self.bic, self.k, self.kdes = max(
                self._map(func, range(trials, (max_k + 1) * trials)))
        else:
            # Build KDEs for each cluster, skipping degenerate clusters
            self.kdes = []
            npts, ndim = pts.shape
            self.k = assign.max() + 1
            for i in range(self.k):
                sel = (assign == i)
                cluster_pts = pts[sel, :]
                # Equivalent to but faster than len(set(pts))
                # FIXME: replace with the following in Numpy >= 1.13.0:
                #   nuniq = len(np.unique(cluster_pts, axis=0))
                nuniq = len(np.unique(
                    np.ascontiguousarray(cluster_pts).view(
                        'V{}'.format(ndim * pts.dtype.itemsize))))
                # Skip if there are fewer unique points than dimensions
                if nuniq <= ndim:
                    continue
                try:
                    kde = gaussian_kde(cluster_pts.T)
                except (np.linalg.LinAlgError, ValueError):
                    # If there are fewer unique points than degrees of freedom,
                    # then the KDE will fail because the covariance matrix is
                    # singular. In that case, don't bother adding that cluster.
                    pass
                else:
                    self.kdes.append(kde)

            # Calculate BIC
            # The number of parameters is:
            #
            # * ndim for each centroid location
            #
            # * (ndim+1)*ndim/2 Kernel covariances for each cluster
            #
            # * one weighting factor for the cluster (minus one for the
            #   overall constraint that the weights must sum to one)
            nparams = self.k*ndim + self.k*((ndim+1)*(ndim)/2) + self.k - 1
            with np.errstate(divide='ignore'):
                self.bic = (
                    np.sum(np.log(self.eval_kdes(pts))) -
                    nparams/2.0*np.log(npts))

    def eval_kdes(self, pts):
        pts = pts.T
        return sum(w * kde(pts) for w, kde in zip(self.weights, self.kdes))

    def __call__(self, pts):
        return self.eval_kdes(pts)

    @property
    def weights(self):
        """Get the cluster weights: the fraction of the points within each
        cluster."""
        w = np.asarray([kde.n for kde in self.kdes])
        return w / np.sum(w)

    def _map(self, func, items):
        # FIXME: ProgressBar.map(..., multiprocess=True) uses imap_unordered,
        # but we want the result to come back in order. This should be fixed,
        # or at least correctly documented, in Astropy.
        if self.multiprocess:
            _, result = zip(*sorted(ProgressBar.map(_mapfunc(func),
                                                    list(enumerate(items)),
                                                    multiprocess=True)))
            return list(result)
        else:
            return ProgressBar.map(func, items, multiprocess=False)


class SkyKDE(ClusteredKDE):

    @classmethod
    def transform(cls, pts):
        """Override in sub-classes to transform points."""
        raise NotImplementedError

    def __init__(self, pts, max_k=40, trials=5, assign=None,
                 multiprocess=False):
        if assign is None:
            pts = self.transform(pts)
        super(SkyKDE, self).__init__(
            pts, max_k=max_k, trials=trials, assign=assign,
            multiprocess=multiprocess)

    def __call__(self, pts):
        return super(SkyKDE, self).__call__(self.transform(pts))

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
        for iround in range(rounds - 1):
            print('adaptive refinement round {} of {} ...'.format(
                  iround + 1, rounds - 1))
            cells = sorted(cells, key=lambda (p, n, i): p / n**2)
            new_nside, new_ipix = np.transpose([
                (nside * 2, ipix * 4 + i)
                for _, nside, ipix in cells[-nrefine:] for i in range(4)])
            theta, phi = hp.pix2ang(new_nside, new_ipix, nest=True)
            ra = phi
            dec = 0.5 * np.pi - theta
            p = self(np.column_stack((ra, dec)))
            cells[-nrefine:] = zip(p, new_nside, new_ipix)
        return cells

    def as_healpix(self):
        """Returns a HEALPix multi-order map of the posterior density."""
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


class _Clustered2DSkyKDEMeta(type):
    """This metaclass is required to make classes picklable because
    our __new__ method dynamically creates a class that is not literally
    present in the module."""
def _Clustered2DSkyKDEMeta_pickle(cls):
    return type, (cls.__name__, cls.__bases__, {'frame': cls.frame})
copyreg.pickle(_Clustered2DSkyKDEMeta, _Clustered2DSkyKDEMeta_pickle)


class Clustered2DSkyKDE(SkyKDE):
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

    __metaclass__ = _Clustered2DSkyKDEMeta
    frame = None

    @classmethod
    def transform(cls, pts):
        pts = SkyCoord(*pts.T, unit='rad').transform_to(cls.frame).spherical
        return np.column_stack((pts.lon.rad, np.sin(pts.lat.rad)))

    def __new__(cls, pts, *args, **kwargs):
        frame = EigenFrame.for_coords(SkyCoord(*pts.T, unit='rad'))
        name = '{:s}_{:x}'.format(cls.__name__, id(frame))
        new_cls = type(name, (cls,), {'frame': frame})
        return super(Clustered2DSkyKDE, cls).__new__(
            new_cls, pts, *args, **kwargs)

    @classmethod
    def __reduce__(cls):
        """This method is required to make instances picklable because
        our __new__ method dynamically creates a class that is not literally
        present in the module."""
        return type, (cls.__name__, cls.__bases__, {'frame': cls.frame})

    def eval_kdes(self, pts):
        base = super(Clustered2DSkyKDE, self).eval_kdes
        dphis = (0.0, 2.0*np.pi, -2.0*np.pi)
        phi, z = pts.T
        return sum(base(np.column_stack((phi+dphi, z))) for dphi in dphis)


class Clustered3DSkyKDE(SkyKDE):
    """Like :class:`Clustered2DSkyKDE`, but clusters in 3D
    space.  Can compute volumetric posterior density (per cubic Mpc),
    and also produce Healpix maps of the mean and standard deviation
    of the log-distance."""

    @classmethod
    def transform(cls, pts):
        return SkyCoord(*pts.T, unit='rad').cartesian.xyz.value.T

    def __call__(self, pts, distances=False):
        """Given an array of positions in RA, DEC, compute the marginal sky
        posterior and optinally the conditional distance parameters."""
        func = partial(distance.cartesian_kde_to_moments,
                       datasets=[_.dataset for _ in self.kdes],
                       inverse_covariances=[_.inv_cov for _ in self.kdes],
                       weights=self.weights)
        probdensity, mean, std = zip(*self._map(func, self.transform(pts)))
        if distances:
            mu, sigma, norm = distance.moments_to_parameters(mean, std)
            return probdensity, mu, sigma, norm
        else:
            return probdensity

    def posterior_spherical(self, pts):
        """Evaluate the posterior probability density in spherical polar
        coordinates, as a function of (ra, dec, distance)."""
        return super(Clustered3DSkyKDE, self).__call__(pts)

    def as_healpix(self):
        """Returns a HEALPix multi-order map of the posterior density
        and conditional distance distribution parameters."""
        m = super(Clustered3DSkyKDE, self).as_healpix()
        order, ipix = moc.uniq2nest(m['UNIQ'])
        nside = 2 ** order.astype(int)
        theta, phi = hp.pix2ang(nside, ipix.astype(np.int64), nest=True)
        p = np.column_stack((phi, 0.5 * np.pi - theta))
        print('evaluating distance layers ...')
        _, m['DISTMU'], m['DISTSIGMA'], m['DISTNORM'] = self(p, distances=True)
        return m


class Clustered2Plus1DSkyKDE(Clustered3DSkyKDE):
    """A hybrid sky map estimator that uses a 2D clustered KDE for the marginal
    distribution as a function of (RA, Dec) and a 3D clustered KDE for the
    conditional distance distribution."""

    def __init__(self, pts, max_k=40, trials=5, assign=None,
                 multiprocess=False):
        if assign is None:
            self.twod = Clustered2DSkyKDE(
                pts, max_k=max_k, trials=trials, assign=assign,
                multiprocess=multiprocess)
        super(Clustered2Plus1DSkyKDE, self).__init__(
            pts, max_k=max_k, trials=trials, assign=assign,
            multiprocess=multiprocess)

    def __call__(self, pts, distances=False):
        probdensity = self.twod(pts)
        if distances:
            base = super(Clustered2Plus1DSkyKDE, self)
            _, distmu, distsigma, distnorm = base.__call__(pts, distances=True)
            return probdensity, distmu, distsigma, distnorm
        else:
            return probdensity

    def posterior_spherical(self, pts):
        """Evaluate the posterior probability density in spherical polar
        coordinates, as a function of (ra, dec, distance)."""
        base = super(Clustered2Plus1DSkyKDE, self)
        return self(pts) * base.posterior_spherical(pts) / base.__call__(pts)
