import bisect
import numpy as np

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
        
        n = 10
        old_areas = np.array([float('-inf') for cf in cfs])

        while True:
            ras = np.linspace(0, 2*np.pi, n)
            sin_decs = np.linspace(-1, 1, n/2)
            dv = 4.0*np.pi/((n-1)*(n/2-1))

            ra_cen = 0.5*(ras[:-1] + ras[1:])
            sin_dec_cen = 0.5*(sin_decs[:-1] + sin_decs[1:])

            RA_CEN,SIN_DEC_CEN = np.meshgrid(ra_cen, sin_dec_cen)

            pts = np.column_stack((RA_CEN.flatten(), SIN_DEC_CEN.flatten()))
            pts[:,1] = np.arcsin(pts[:,1])

            posts = self.posterior(pts)

            areas = []
            for pl in pls:
                areas.append(dv*np.sum(posts > pl))
            areas=np.array(areas)

            if np.all(areas > 0) and np.all(np.abs((areas-old_areas)/areas) < 0.01):
                break

            old_areas = areas
            n *= 2

        return areas

    def to_degrees(self, sr):
        return sr*180.0/np.pi*180.0/np.pi
