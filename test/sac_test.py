import numpy as np
import sky_area_clustering as sac

def project_pts(pts):
    """Returns the RA-DEC positions of three-dimensional points (not
    necessarily on the sphere).

    """
    norms = np.sqrt(np.sum(pts*pts, axis=1))
    pts = pts / norms.reshape((-1, 1))

    thetas = np.arccos(pts[:,2])
    phis = np.arctan2(pts[:,1], pts[:,0])

    phis[phis < 0] += 2.0*np.pi

    return np.column_stack((phis, np.pi/2.0 - thetas))

def draw_pts(n):
    theta = np.arccos(np.random.uniform(low=-1, high=1))
    phi = np.random.uniform(low=0, high=2.0*np.pi)

    center = np.array([np.cos(phi)*np.sin(theta),
                       np.sin(phi)*np.sin(theta),
                       np.cos(theta)])

    cov = np.exp(np.random.uniform(low=np.log(0.3), high=np.log(3.0), size=(3,3)))
    cov = np.dot(cov, cov.T)

    return project_pts(np.random.multivariate_normal(center, cov, n))

def p_value(n):
    pts = draw_pts(n+1)
    pt = pts[0,:]
    pts = pts[1:,:]

    skypost = sac.ClusteredKDEPosterior(pts)

    return skypost.p_values(np.reshape(pt, (1, 2)))[0]
