from netket.graph import Lattice
from netket.utils.group import PermutationGroup, Identity
from netket.graph.space_group import Translation
from functools import reduce
from typing import Optional


def custom_translations_along_axis(
    lattice: Lattice, axis: int, n: int, max_translation: int = None
) -> PermutationGroup:
    """
    The group of translations by n*lattice.basis_vector[axis] as a `PermutationGroup`
    acting on the sites of `lattice.`
    This is a modification of the original `_translations_along_axis` in `netket.graph.space_group`
    """
    if lattice._pbc[axis]:
        displacement_vector = n * lattice.basis_vectors[axis]
        if max_translation is None:
            max_translation = lattice.extent[axis] // n
        trans_list = [Identity()]
        # note that we need the preimages in the permutation
        trans_perm = lattice.id_from_position(lattice.positions - displacement_vector)
        trans_by_n = Translation(
            inverse_permutation_array=trans_perm, displacement=displacement_vector
        )

        for _ in range(1, max_translation):
            trans_list.append(trans_list[-1] @ trans_by_n)

        return PermutationGroup(trans_list, degree=lattice.n_nodes)
    else:
        return PermutationGroup([Identity()], degree=lattice.n_nodes)


def translation_group_from_axis_translations(
    lattice: Lattice, n: tuple[int], max_translations: Optional[tuple[int]] = None
):
    axes = tuple(range(lattice.ndim))
    if max_translations is None:  # use None along all axes
        max_translations = tuple([None] * lattice.ndim)
    translation_by_axis = [
        custom_translations_along_axis(lattice, i, n[i], max_translations[i])
        for i in axes
    ]
    translation_group = reduce(PermutationGroup.__matmul__, translation_by_axis)
    return translation_group
