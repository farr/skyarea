import bisect
import healpy
import numpy as np
from scipy.stats import gaussian_kde

class SkyPosterior(object):
    def __init__(self, pts):
        """Initialize a posterior function with the given RA-DEC samples.

        :param pts: An array of shape ``(N,2)`` giving the RA and DEC
          samples on the sky.

        """
        self._pts = pts.copy()
        self._pts[:,1] = np.sin(self._pts[:,1])
        self._kde = gaussian_kde(self.pts.T)
        

    @property
    def pts(self):
        """Returns the sample points in RA-sin(DEC) space.

        """
        return self._pts

    @property
    def kde(self):
        """Returns the KDE function on RA-sin(DEC) used to estimate the
        posterior.

        """
        return self._kde

    def posterior(self, pts):
        r"""Returns a kernel-density estimate of the posterior density per sr
        on the sky.  Note that this is not the density in RA-DEC
        space, which differs by a factor of :math:`\cos \delta`.

        """
        
        pts_sin = pts.copy()
        pts_sin[:,1] = np.sin(pts_sin[:,1])

        post = np.zeros(pts.shape[0])
        for ra_inc in [0, 2*np.pi, -2*np.pi]:
            ps = pts_sin.copy()
            ps[:,0] += ra_inc

            ps_low = ps.copy()
            ps_low[:,1] = -2.0 - ps[:,1]

            ps_high = ps.copy()
            ps_high[:,1] = 2.0 - ps[:,1]

            post += self.kde(ps.T) + self.kde(ps_low.T) + self.kde(ps_high.T)

        return post

    def posterior_level(self, cfs):
        """Returns the posterior contour levels that correspond to the given
        credible fractions on the sky.

        """

        cfs = np.atleast_1d(cfs)

        pts = self.pts.copy()
        pts[:,1] = np.arcsin(pts[:,1]) # transform back to RA-DEC

        posts = self.posterior(pts)
        iposts = np.argsort(posts)[::-1]

        plevels = []
        for cf in cfs:
            plevels.append(posts[iposts[int(round(cf*self.pts.shape[0]))]])

        return np.array(plevels)

    def sky_area(self, cfs):
        """Returns the sky area in sr contained in the given credible
        fractions.  

        """
        cfs = np.atleast_1d(cfs)

        pls = self.posterior_level(cfs)
        
        nside = 2
        old_areas = np.zeros(cfs.shape)

        while True:
            theta, phi = healpy.pix2ang(nside, np.arange(0, healpy.nside2npix(nside), dtype=np.int))
            ras = phi
            decs = np.pi/2.0 - theta

            pts = np.column_stack((ras, decs))

            post_at_pts = self.posterior(pts)
            areas = []
            for pl in pls:
                nabove = np.count_nonzero(post_at_pts > pl)
                areas.append(nabove*healpy.nside2pixarea(nside))
            areas = np.array(areas)

            print nside, areas

            if np.all(areas > 0) and np.all(np.abs((areas-old_areas)/areas) < 1e-2):
                break

            old_areas = areas
            nside *= 2

        return areas

    def sky_area_fixed_resolution(self, nside, cfs):
        cfs = np.atleast_2d(cfs)

        counts = np.zeros(healpy.nside2npix(nside), dtype=np.int)
        for p in self.pts:
            ra = p[0]
            dec = np.arcsin(p[1])

            theta = np.pi/2.0 - dec
            phi = ra

            counts[healpy.ang2pix(nside, theta, phi)] += 1

        theta_cen, phi_cen = healpy.pix2ang(nside, np.arange(0, healpy.nside2npix(nside), dtype=np.int))
        ra_cen = phi_cen
        dec_cen = np.pi/2.0 - theta_cen

        post_cen = self.posterior(np.column_stack((ra_cen, dec_cen)))
        greedy_order = np.argsort(post_cen)[::-1]
        counts_greedy = counts[greedy_order]
        counts_greedy_cum = np.cumsum(counts_greedy)

        areas = []
        for cf in cfs:
            count_threshold = int(round(cf*healpy.nside2npix(nside)))
            nsearched = bisect.bisect_right(counts_greedy_cum, count_threshold)
            areas.append(nsearched*healpy.nside2pixarea(nside))

        return np.array(areas)

    def to_degrees(self, sr):
        return sr*180.0/np.pi*180.0/np.pi
