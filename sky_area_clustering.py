import bisect as bs
import healpy as hp
import numpy as np
import numpy.linalg as nl
from scipy.stats import gaussian_kde

def km_assign(mus, cov, pts):
    """Implements the assignment step in the k-means algorith.  Given a
    set of centers, ``mus``, a covariance matrix used to produce a
    metric on the space, ``cov``, and a set of points, ``pts`` (shape
    ``(npts, ndim)``), assigns each point to its nearest center,
    returning an array of indices of shape ``(npts,)`` giving the
    assignments.

    """
    k = mus.shape[0]
    n = pts.shape[0]

    dists = np.zeros((k,n))

    for i,mu in enumerate(mus):
        dx = pts - mu
        dists[i,:] = np.sum(dx*nl.solve(cov, dx.T).T, axis=1)

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
        sel = assign==i
        if np.count_nonzero(sel) > 0:
            mus[i,:] = np.mean(pts[sel, :], axis=0)
        else:
            mus[i,:] = pts[np.random.randint(pts.shape[0]), :]

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

class ClusteredKDEPosterior(object):
    r"""Represents a kernel-density estimate of a sky-position PDF that has
    been decomposed into clusters, using a different kernel for each
    cluster.

    The estimated PDF is 
    
    .. math::

      p\left( \vec{\theta} \right) = \sum_{i = 0}^{k-1} \frac{N_i}{N} \sum_{\vec{x} \in C_i} N\left[\vec{x}, \Sigma_i\right]\left( \vec{\theta} \right)

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
        pts = pts.copy()
        pts[:,1] = np.sin(pts[:,1])
        self._pts = pts
        self._ntrials = ntrials

        if means is None or assign is None:
            self._set_up_kmeans(1)
            low_bic = self._bic()
            low_assign = self.assign
            low_means = self.means
            low_k = 1

            mid_bic = self._set_up_optimal_kmeans(2, ntrials)
            mid_assign = self.assign
            mid_means = self.means
            mid_k = 2

            if low_bic > mid_bic: # Only need one subdivision
                self._set_up_kmeans(1)
            else:
                high_bic = self._set_up_optimal_kmeans(4, ntrials)
                high_assign = self.assign
                high_means = self.means

                low_k, mid_k, high_k = 1, 2, 4

                while high_bic > mid_bic:
                    print 'extending ks: ', (low_k, mid_k, high_k)
                    print 'with bics: ', (low_bic, mid_bic, high_bic)

                    low_k, mid_k = mid_k, high_k
                    low_bic, mid_bic = mid_bic, high_bic
                    low_means, mid_means = mid_means, high_means
                    low_assign, mid_assign = mid_assign, high_assign

                    high_k = 2*mid_k
                    while True:
                        try:
                            high_bic = self._set_up_optimal_kmeans(high_k, ntrials)
                            high_means = self.means
                            high_assign = self.assign
                        except:
                            high_k = mid_k + (high_k - mid_k)/2
                            if high_k >= mid_k + 1:
                                continue
                            else:
                                raise
                        break

                while high_k - low_k > 2:
                    print 'shrinking ks: ', (low_k, mid_k, high_k)
                    print 'with bics: ', (low_bic, mid_bic, high_bic)

                    if high_k - mid_k > mid_k - low_k:
                        k = mid_k + (high_k - mid_k)/2
                        bic = self._set_up_optimal_kmeans(k, ntrials)
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
                        bic = self._set_up_optimal_kmeans(k, ntrials)
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
            
                print 'Found best k, BIC: ', mid_k, mid_bic
                self._set_up_kmeans(mid_k, mid_means, mid_assign)
        else:
            self._set_up_kmeans(means.shape[0], means, assign)

        self._set_up_greedy_order()
        
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

    @property
    def greedy_order(self):
        """Returns the ordering of ``self.pts`` from highest to lowest
        posterior values.

        """
        return self._greedy_order

    @property
    def greedy_posteriors(self):
        """Returns the posterior values at ``self.pts`` in greedy order.

        """
        return self._greedy_posteriors

    def _set_up_optimal_kmeans(self, k, ntrials):
        best_bic = np.NINF
        
        for i in range(ntrials):
            self._set_up_kmeans(k)
            bic = self._bic()

            if bic > best_bic:
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
            if np.count_nonzero(sel) == 0:
                self._kdes.append(lambda x : 0.0)
                self._weights.append(0.0)
            else:
                self._kdes.append(gaussian_kde(self.pts[sel,:].T))
                self._weights.append(float(np.count_nonzero(sel))/float(self.pts.shape[0]))
        self._weights = np.array(self.weights)

    def _set_up_greedy_order(self):
        pts = self.pts.copy()
        pts[:,1] = np.arcsin(pts[:,1])

        posts = self.posterior(pts)
        self._greedy_order = np.argsort(posts)[::-1]
        self._greedy_posteriors = posts[self.greedy_order]

    def posterior(self, pts):
        """Returns the clustered KDE estimate of the sky density per steradian
        at the given points in RA-DEC.

        """
        pts = pts.copy()
        pts[:,1] = np.sin(pts[:,1])
        
        post = np.zeros(pts.shape[0])

        ras = pts[:,0]
        sin_decs = pts[:,1]

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

        ndim = self.pts.shape[1]
        npts = self.pts.shape[0]

        nparams = self.k*(ndim + (ndim+1)*ndim/2) # We fit both the
                                                  # positions and the
                                                  # covariances of the
                                                  # kernels

        pts = self.pts.copy()
        pts[:,1] = np.arcsin(pts[:,1])

        return np.sum(np.log(self.posterior(pts))) - nparams/2.0*np.log(self.pts.shape[0])

    def _area_within_nside(self, levels, nside):
        thetas, phis = hp.pix2ang(nside, np.arange(hp.nside2npix(nside), dtype=np.int))
        pixels = np.column_stack((phis, np.pi/2.0-thetas))
        pixel_posts = self.posterior(pixels)

        return hp.nside2pixarea(nside)*np.array([np.count_nonzero(pixel_posts > lev) for lev in levels])

    def _area_within(self, levels, acc = 1e-2):
        old_areas = self._area_within_nside(levels, 1)
        nside = 2
        while True:
            areas = self._area_within_nside(levels, nside)

            if np.all(areas > 0) and np.all(np.abs((areas-old_areas)/areas) < acc):
                break
            else:
                old_areas = areas
                nside *= 2

        return areas
        

    def sky_area(self, cls):
        """Returns the sky area occupied by the given list of credible levels.

        """
        post_levels = [self.greedy_posteriors[int(round(cl*self.pts.shape[0]))] for cl in cls]

        return self._area_within(post_levels)

    def searched_area(self, pts):
        """Returns the sky area that must be searched using a greedy algorithm
        before encountering the given points in the sky.

        """
        post_levels = self.posterior(pts)

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

        return 1.0 - np.array(indexes)/float(n)
