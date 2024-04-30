import numpy as np

__all__ = ["get_relative_position"]


def get_relative_position(loc1, loc2):
    """Returns the relative position (range, azimuth and elevation) between 2 locations.

    Parameters
    ----------
    loc1, loc2: array_like, shape (3,)
        Location of the 2 points.
    """
    loc1 = np.asarray(loc1).reshape(3)
    loc2 = np.asarray(loc2).reshape(3)
    dxyz = dx, dy, dz = loc2 - loc1
    range = np.linalg.norm(dxyz)
    az = np.arctan2(dy, dx)
    el = np.arcsin(dz / range)
    return range, az, el
