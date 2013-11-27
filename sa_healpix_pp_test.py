import healpy as hp
import numpy as np
import sky_area_healpix as sap

def draw_points(npts):
    sigma = np.exp(np.random.uniform(low=np.log(0.01), high=0.0))
    ra0 = np.random.uniform(low=0.0, high=2.0*np.pi)
    dec0 = np.arcsin(np.random.uniform(low=-1.0, high=1.0))
    x0 = np.array([np.cos(ra0)*np.cos(dec0),
                   np.sin(ra0)*np.cos(dec0),
                   np.sin(dec0)])

    pts = x0 + np.random.normal(scale=sigma, size=(npts+1,3))
    norm2 = np.sum(pts*pts, axis=1)
    norm = np.sqrt(norm2)
    pts /= np.reshape(norm, (-1, 1))

    ras = np.arctan2(pts[:,1], pts[:,0])
    decs = np.arcsin(pts[:,2])
    
    pts = np.column_stack((ras, decs))

    return pts
    

if __name__ == '__main__':
    ntests = 100
    npts = 10000

    ps = []
    for i in range(ntests):
        pts = draw_points(npts)
        sky = sap.HealpixSkyPosterior(pts[1:,:])
        p0 = pts[0,:]
        ps.append(sky.pt2credible(p0)[0])
    ps = np.array(ps).reshape((-1,1))

    np.savetxt('ps.dat', ps)
