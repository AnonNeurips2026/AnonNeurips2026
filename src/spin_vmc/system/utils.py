# 2D spin models
import numpy as np
import netket as nk
from netket.utils.group._point_group import PGSymmetry, PointGroup
from netket.utils.group._semigroup import Identity
from netket.utils.types import Array


# Functions for defining symmetries of Shastry-Sutherland model
def reflect_and_translate(angle: float, t: Array) -> PGSymmetry:
    """
    Return `PGSymmetry` representing a translation followed by a 2D reflection across an axis at angle `angle` to the +x direction through the (0,0) point.
    This is equivalent to reflecting in the axis at angle `angle` passing through (0,0) and then translating by W@`t`, where W is
    the reflection matrix.
    Args:
        angle: the angle between the +x axis and the reflection axis.
        t: the translation vector
    """
    axis = np.radians(angle) * 2  # the mirror matrix is written in terms of 2φ
    W = np.asarray(
        [[np.cos(axis), np.sin(axis)], [np.sin(axis), -np.cos(axis)]]
    )  # the reflection matrix
    w = W @ t
    return PGSymmetry(W, w)  # transformation is W@x + w


def reflect_and_translate_group(angle: float, t: Array) -> PGSymmetry:
    """
    Returns the Z_2 `PointGroup`containing the identity and reflect_and_translate(angle,t)

    Arguments:
        trans: translation vector
        origin: a point on the glide axis, defaults to the origin
    The output is only a valid `PointGroup` after supplying a `unit_cell`
    consistent with the glide axis; otherwise, operations like `product_table`
    will fail.
    """
    return PointGroup([Identity(), reflect_and_translate(angle, t)], ndim=2)

def is_symmetric(symm_op, graph: nk.graph.Graph) -> bool:
    """
    Check if all edges of the graph are left invariant by the symmetry operation
    Arguments:
        symm_op: Symmetry operation
        graph: nk.graph.Graph with graph.edges(return_color=True) defined
    Returns:
        True if all edges are left invariant by the symmetry operation, False otherwise
    """
    edges_old = np.array(
        graph.edges(return_color=True)
    )  # edges_old[n,:] = [i,j,k] with (i,j) the nodes and k the color of the edge
    old_indices = np.arange(graph.n_nodes)
    new_indices = symm_op(
        old_indices
    )  # new_indices[i] = j means that the ith node is mapped to the jth node by the symmetry operation
    edges_new = np.zeros(
        (edges_old.shape[0], edges_old.shape[1] - 1)
    )  # add edges with nodes swapped to this

    for old_index in old_indices:
        new_index = new_indices[old_index]
        edges_new[edges_old[:, :2] == old_index] = (
            new_index  # replace the old index the new index
        )
        # print(f"Replacing {edges_old[:,:2][edges_old[:,:2]==old_index]} with {new_index}")

    edges_new = np.hstack(
        (edges_new, edges_old[:, 2][:, np.newaxis])
    )  # add the last column specifying edge colors which remains the same

    # Now compare edges_new and edges_old, they should contain all of the same edges
    edges_check = edges_old.copy()
    # count = 0
    for row in edges_new:
        found_index = np.where(np.all(edges_old == row, axis=1))[
            0
        ]  # the index of where the row appears in edges_old
        if len(found_index) == 0:  # if not found
            row = np.array([row[1], row[0], row[2]])  # permute i,j
            found_index = np.where(np.all(edges_old == row, axis=1))[
                0
            ]  # try to find the row again
        edges_check[found_index] = -1  # set the row to -1

    return np.all(
        edges_check == -1
    )  # if all rows are -1, then all of the original edges are contained in edges_new
