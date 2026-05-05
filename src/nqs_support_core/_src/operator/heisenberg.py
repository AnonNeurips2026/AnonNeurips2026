from netket.operator import LocalOperator
from netket.hilbert import AbstractHilbert
from collections.abc import Sequence
from netket.operator.spin import sigmaz, sigmap, sigmam


def heisenberg_edges(
    hilbert: AbstractHilbert,
    edges: list[tuple[int, int, int]],
    J: float | Sequence[float] = 1.0,
) -> LocalOperator:
    """
    Constructs a Heisenberg operator on the pairs of nodes of the nn graph given a hilbert space.

    Args:
        hilbert: Hilbert space the operator acts on.
        graph: The graph upon which this hamiltonian is defined by nodes and their distances
        J: The strength of the coupling. Default is 1.
            Can pass a sequence of coupling strengths with coloured graphs:
            edges of colour n will have coupling strength J[n]
    """
    if type(J) is float:
        J = (J,)

    colors = [c for _, _, c in edges]
    max_color = max(colors)
    assert max_color + 1 == len(J)
    sz = sigmaz
    sp = sigmap  # 1/2 (sx + isy)
    sm = sigmam  # 1/2 (sx - isy)

    ham = 0
    for i, j, c in edges:
        ham += J[c] * (
            sz(hilbert, i) @ sz(hilbert, j)
            + 2 * sp(hilbert, i) @ sm(hilbert, j)
            + 2 * sm(hilbert, i) @ sp(hilbert, j)
        )

    return ham
