from netket.operator.spin import sigmax, sigmay, sigmaz, sigmam, sigmap
from netket.operator._local_operator import LocalOperator as _LocalOperator
from netket.hilbert import DiscreteHilbert as _DiscreteHilbert
from netket.utils.types import DType as _DType
from netket.graph import Graph
from netket.hilbert import DiscreteHilbert


def Sx(hilbert: _DiscreteHilbert, site: int, dtype: _DType = None) -> _LocalOperator:
    """
    Builds the :math:`S^x` operator acting on the `site`-th of the Hilbert
    space `hilbert`, such that it's eigenvalues are +- S
    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    return sigmax(hilbert, site, dtype) / 2


def Sy(hilbert: _DiscreteHilbert, site: int, dtype: _DType = None) -> _LocalOperator:
    """
    Builds the :math:`S^y` operator acting on the `site`-th of the Hilbert
    space `hilbert`, such that it's eigenvalues are +- S
    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    return sigmay(hilbert, site, dtype) / 2


def Sz(hilbert: _DiscreteHilbert, site: int, dtype: _DType = None) -> _LocalOperator:
    """
    Builds the :math:`S^z` operator acting on the `site`-th of the Hilbert
    space `hilbert`, such that it's eigenvalues are +- S
    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    return sigmaz(hilbert, site, dtype) / 2


def Sm(hilbert: _DiscreteHilbert, site: int, dtype: _DType = None) -> _LocalOperator:
    """
    Builds the :math:`(S^x-iS^y)` operator acting on the `site`-th of the Hilbert
    space `hilbert`, corresponding to the S^x, S^y above.
    It is constructed as a ladder operator between S^z eigenstates, without needing complex elements.
    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    return sigmam(hilbert, site, dtype)


def Sp(hilbert: _DiscreteHilbert, site: int, dtype: _DType = None) -> _LocalOperator:
    """
    Builds the :math:`(S^x+iS^y)` operator acting on the `site`-th of the Hilbert
    space `hilbert`, corresponding to the S^x, S^y above.
    It is constructed as a ladder operator between S^z eigenstates, without needing complex elements.
    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    return sigmap(hilbert, site, dtype)


##Correlators
class Correlators:
    def __init__(
        self,
        op1: _LocalOperator,
        op2: _LocalOperator,
        hilbert_space: DiscreteHilbert,
        graph: Graph,
        transl_invariant: bool,
        name: str = "O1O2",
    ):
        self.N = graph.n_nodes
        self.op1 = op1
        self.op2 = op2
        self.hilbert_space = hilbert_space
        if self.bond_op1(0, 1).is_hermitian:  # e.g S_i^z S_j^z
            self.correlator = self.bond_op1
        else:
            if (
                self.bond_op1(0, 1) + self.bond_op1(1, 0)
            ).is_hermitian:  # e.g S_i^+ S_j^- + h.c
                self.correlator = self.bond_op2
            else:
                raise ValueError(
                    "Cannot construct hermitian local operator from given op1, op2"
                )

        if transl_invariant:  # only need to measure relative to one site
            self.ij_pairs = [(0, j) for j in range(self.N)]

        else:  # for all i <= j, assumes (op1(i)*op2(j) + h.c) = (op1(j)*op2(i) + h.c)
            self.ij_pairs = [(i, j) for i in range(self.N) for j in range(i + 1)]

        self.local_operators = {}
        for i, j in self.ij_pairs:
            self.local_operators[f"{name}({i,j})"] = self.correlator(i, j)

    def bond_op1(self, i, j):
        """
        O1_i*O2_j
        """
        return self.op1(self.hilbert_space, i) * self.op2(self.hilbert_space, j)

    def bond_op2(self, i, j):
        """
        O1_i*O2_j+O1_j*O2_i
        """
        return self.op1(self.hilbert_space, i) * self.op2(
            self.hilbert_space, j
        ) + self.op1(self.hilbert_space, j) * self.op2(self.hilbert_space, i)
