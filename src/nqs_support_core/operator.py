__all__ = [
    "Rx",
    "Ry",
    "Hadamard",
    "SumOperator",
    "Sx",
    "Sy",
    "Sz",
    "Sm",
    "Sp",
    "Correlators",
    "SpinonHeisenberg",
    "heisenberg_edges",
]

from nqs_support_core._src.operator.singlequbit_gates import Rx, Ry, Hadamard
from nqs_support_core._src.operator.sum import SumOperator
from nqs_support_core._src.operator.spin import Sx, Sy, Sz, Sm, Sp, Correlators
from nqs_support_core._src.operator.spinon import SpinonHeisenberg
from nqs_support_core._src.operator.heisenberg import heisenberg_edges

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

from netket.experimental.operator import (
    ParticleNumberConservingFermioperator2nd as _deprecated_ParticleNumberConservingFermioperator2nd,
)
from netket.experimental.operator import (
    ParticleNumberAndSpinConservingFermioperator2nd as _deprecated_ParticleNumberAndSpinConservingFermioperator2nd,
)


_deprecations = {
    # July 2025, netket 3.18
    "ParticleNumberConservingFermioperator2ndJax": (
        "nqs_support_core.operator.ParticleNumberConservingFermioperator2ndJax is deprecated: use "
        "netket.experimental.operator.ParticleNumberConservingFermioperator2nd (netket >= 3.18)",
        _deprecated_ParticleNumberConservingFermioperator2nd,
    ),
    "ParticleNumberConservingFermioperator2ndSpinJax": (
        "nqs_support_core.operator.ParticleNumberConservingFermioperator2ndSpinJax is deprecated: use "
        "netket.experimental.operator.ParticleNumberAndSpinConservingFermioperator2nd (netket >= 3.18)",
        _deprecated_ParticleNumberAndSpinConservingFermioperator2nd,
    ),
}

__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
