import netket as nk
import numpy as np
from netket.utils.group._point_group import PGSymmetry, PointGroup
from netket.utils.group._semigroup import Identity


def P4132():
    """
    Point group of the Hyperkagome lattice P4132 (No. 213).
    Made up of a C2 rotation, C3 rotation and S4 non-symmporphic screw.
    See Huang et al. PRB 95 054404 (2017) for details and definitions of symmetry operations.
    """
    # C2 symmetry
    W = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
    w = 0.75 * np.array([1, 1, 1])
    C2_rotation = PGSymmetry(W, w)
    C2 = PointGroup([Identity(), C2_rotation], ndim=3)
    # C3 symmetry along (1,1,1)
    C3 = nk.utils.group.axial.C(n=3, axis=np.array([1, 1, 1]))
    rotations = nk.utils.group._point_group.product(C2, C3)
    # S4 nonsymmorphic screw
    W = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    w = np.array([0.25, 0.25, 0.75])
    S4_rotation = PGSymmetry(W, w)
    S4_1 = PointGroup([S4_rotation], ndim=3, unit_cell=np.array([0, 0, 0]))
    S4_2 = nk.utils.group._point_group.product(S4_1, S4_1)
    S4_3 = nk.utils.group._point_group.product(S4_2, S4_1)
    S4 = PointGroup(
        [Identity(), S4_1[0], S4_2[0], S4_3[0]], ndim=3, unit_cell=np.array([0, 0, 0])
    )
    point_group = nk.utils.group._point_group.product(rotations, S4)
    return point_group
