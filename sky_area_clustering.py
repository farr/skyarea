import numpy as np
import numpy.linalg as nl

def sky_eval(pdf, pts):
    """Returns the value of the probability density in ``pdf`` evaluated
    at the ra-sin_dec points ``pts`` properly accounting for
    periodicity in the sky coordinates.

    """

    ps = pdf(pts)

    for dra in [-2.0*np.pi, 2.0*np.pi]:
        ra = pts[:,0] + dra

        sin_dec = 2.0 - pts[:,1]
        ps += pdf(np.column_stack((ra, sin_dec)))

        sin_dec = -2.0 - pts[:,1]        
        ps += pdf(np.column_stack((ra, sin_dec)))
        
    return ps

def log_gaussian(xs, mu, cov):
    """Returns the log of normal PDF with the given mean and covariance at
    the given points.

    """

    dx = xs - mu

    return -np.log(2.0*np.pi) - 0.5*nl.slogdet(cov)[1] - 0.5*np.dot(dx, nl.solve(cov, dx))

def km_assign(mus, cov, pts):
    k = mus.shape[0]
    n = pts.shape[0]

    dists = np.zeros((k,n))

    for i,mu in enumerate(mus):
        dx = pts - mu
        dists[i,:] = np.sum(dx*nl.solve(cov, dx.T).T, axis=1)

    return np.argmin(dists, axis=0)

def km_centroids(pts, assign, k):
    mus = np.zeros((k, pts.shape[1]))
    for i in range(k):
        sel = assign==i
        if np.count_nonzero(sel) > 0:
            mus[i,:] = np.mean(pts[sel, :], axis=0)
        else:
            mus[i,:] = pts[np.random.randint(pts.shape[0]), :]

    return mus

def k_means(pts, k):
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
