from astropy.coordinates import (
    BaseCoordinateFrame, CartesianRepresentation, DynamicMatrixTransform,
    frame_transform_graph, ICRS)
try:
    from astropy.coordinates import CartesianRepresentationAttribute
except ImportError:
    from astropy.coordinates.baseframe import \
        CartesianRepresentationFrameAttribute \
        as CartesianRepresentationAttribute
from astropy.units import dimensionless_unscaled
import numpy as np

__all__ = ('EigenFrame',)


class EigenFrame(BaseCoordinateFrame):
    """A coordinate frame that has its axes aligned with the principal
    components of a cloud of points."""

    e_x = CartesianRepresentationAttribute(
        default=CartesianRepresentation(1, 0, 0, unit=dimensionless_unscaled),
        unit=dimensionless_unscaled)
    e_y = CartesianRepresentationAttribute(
        default=CartesianRepresentation(0, 1, 0, unit=dimensionless_unscaled),
        unit=dimensionless_unscaled)
    e_z = CartesianRepresentationAttribute(
        default=CartesianRepresentation(0, 0, 1, unit=dimensionless_unscaled),
        unit=dimensionless_unscaled)

    default_representation = CartesianRepresentation

    @classmethod
    def for_coords(cls, coords):
        """Create a coordinate frame that has its axes aligned with the
        principal components of a cloud of points.

        Parameters
        ----------
        coords : astropy.cordinates.SkyCoord
            A cloud of points

        Returns
        -------
        frame : EigenFrame
            A new coordinate frame
        """
        obj = cls()
        v = coords.icrs.cartesian.xyz.value
        _, R = np.linalg.eigh(np.dot(v, v.T))
        R = R[:, ::-1]  # Order by descending eigenvalue
        e_x, e_y, e_z = CartesianRepresentation(R, unit=dimensionless_unscaled)
        return cls(e_x=e_x, e_y=e_y, e_z=e_z)


@frame_transform_graph.transform(DynamicMatrixTransform, ICRS, EigenFrame)
def icrs_to_eigenframe(from_coo, to_frame):
    return np.row_stack((to_frame.e_x.xyz.value,
                         to_frame.e_y.xyz.value,
                         to_frame.e_z.xyz.value))
