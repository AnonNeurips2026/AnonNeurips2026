from netket.hilbert import SpinOrbitalFermions
from netket.graph import Graph
from netket.operator.fermion import destroy as f
from netket.operator.fermion import create as fdag
from netket.operator.fermion import number as nf


def SpinonHeisenberg(hilbert: SpinOrbitalFermions, graph: Graph, J: float):
    r"""
    Heisenberg Hamiltonian for spinons, with coupling, J, between all edges of graph
    H = J \sum_{i,j} [1/2 *(f^i,up fi,down f^j,down fj,up + h.c) + 1/4*(ni,up - ni,down)*(nj,up-nj,down)]
    """
    ham = 0
    for i, j in graph.edges():
        ham += 0.5 * (
            fdag(hilbert, i, sz=+1)
            * f(hilbert, i, sz=-1)
            * fdag(hilbert, j, sz=-1)
            * f(hilbert, j, sz=+1)
        )
        ham += 0.5 * (
            fdag(hilbert, j, sz=+1)
            * f(hilbert, j, sz=-1)
            * fdag(hilbert, i, sz=-1)
            * f(hilbert, i, sz=+1)
        )
        ham += (
            0.25
            * (nf(hilbert, i, sz=+1) - nf(hilbert, i, sz=-1))
            * (nf(hilbert, j, sz=+1) - nf(hilbert, j, sz=-1))
        )

    ham *= J
    return ham
