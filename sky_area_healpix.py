import acor
import bisect
import healpy
import numpy as np

class HealpixSkyPosterior(object):
    r"""Class that represents the posterior 

    """

    def __init__(self, pts, nside=None):
        self._pts = pts
        npts = self.pts.shape[0]
        
        if nside is None:
            self._nside = self.optimal_nside(pts[:npts/2,:], pts[npts/2:,:])
        else:
            self._nside = nside

        self._den_map = self.density_map(pts[:npts/2, :])
        self._gr_order = self.greedy_order(pts[npts/2:, :])

        dV = healpy.nside2pixarea(self.nside)
        self._cum_den = dV*np.cumsum(self.den_map[self.gr_order])[np.argsort(self.gr_order)]

    @property
    def pts(self):
        return self._pts

    @property
    def nside(self):
        return self._nside

    @property
    def den_map(self):
        return self._den_map

    @property
    def gr_order(self):
        return self._gr_order

    @property
    def cum_den(self):
        return self._cum_den

    def pts2pix(self, pts, nside=None):
        if nside is None:
            nside = self.nside

        return healpy.ang2pix(nside, np.pi/2.0-pts[:,1], pts[:,0])

    def density_map(self, pts, nside=None):
        if nside is None:
            nside = self.nside

        pixels = self.pts2pix(pts, nside)
        dV = healpy.nside2pixarea(nside)
        
        density = np.zeros(healpy.nside2npix(nside))
        
        for pix in pixels:
            density[pix] += 1.0

        zero_sel = density == 0.0
        nzero = np.count_nonzero(zero_sel)
        if nzero > 0:
            density[zero_sel] = np.random.uniform(low=0.0, high=1.0/nzero, size=nzero)
            # put an average of 1/2 a count in total of zero region

        density /= dV*np.sum(density)

        return density

    def log_likelihood(self, pts, den_map, nside=None):
        if nside is None:
            nside = self.nside

        pixels = self.pts2pix(pts, nside)

        ll = 0.0
        for pix in pixels:
            ll += np.log(den_map[pix])

        return ll

    def optimal_nside(self, pts1, pts2):
        tau = max(acor.acor(pts2[:,0])[0], acor.acor(pts2[:,1])[0])
        step = max(1, int(round(tau)))
        pts2 = pts2[::step]

        nside = 2
        last_ll = np.NINF
        while True:
            den_map = self.density_map(pts1, nside)
            ll = self.log_likelihood(pts2, den_map, nside)

            if ll < last_ll:
                break
            last_ll = ll
            nside *= 2

        return nside / 2 # the best nside is half the current value
        
    def greedy_order(self, pts, nside=None):
        if nside is None:
            nside = self.nside

        dmap = self.density_map(pts)

        return np.argsort(dmap)[::-1]

    def credible2area(self, p):
        assert p >= 0 and p <= 1, 'p must be between [0,1]'

        dV = healpy.nside2pixarea(self.nside)
        index = bisect.bisect(self.cum_den, p)

        return index*dV

    def pt2credible(self, pts):
        pts = np.atleast_2d(pts)

        pix = self.pts2pix(pts)
        
        return self.cum_den[pix]
