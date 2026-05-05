from functools import partial

# flake8: noqa: E402
from nqxpack._src.lib_v1.custom_types import (
    register_serialization,
)


# Graph

from nqs_support_core._src.vqs.full_summ.mixed_state import FullSumMixedState
from nqxpack._src.registry.netket import serialize_fullsumstate, deserialize_vstate

register_serialization(
    FullSumMixedState,
    partial(serialize_fullsumstate, mixed_state=True),
    partial(deserialize_vstate, FullSumMixedState),
)

# Note: nqs_support_core sampler rules register themselves for serialization
# in their respective modules to avoid circular imports
