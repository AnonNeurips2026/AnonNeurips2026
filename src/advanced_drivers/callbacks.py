from advanced_drivers._src.callbacks import (
    AbstractCallback as AbstractCallback,
    EarlyStopping as EarlyStopping,
    Timeout as Timeout,
    InvalidLossStopping as InvalidLossStopping,
    ConvergenceStopping as ConvergenceStopping,
)

from advanced_drivers._src.callbacks.auto_chunk_size import (
    AutoChunkSize as AutoChunkSize,
)
from advanced_drivers._src.callbacks.auto_slurm_requeue import (
    AutoSlurmRequeue as AutoSlurmRequeue,
)

from advanced_drivers._src.callbacks.autodiagshift import (
    PI_controller_diagshift as PI_controller_diagshift,
)

from advanced_drivers._src.callbacks.autodiagshift_fs import (
    PI_controller_diagshift_fs as PI_controller_diagshift_fs,
)


def __getattr__(name):
    """Lazy import for CheckpointCallback (requires netket_checkpoint)."""
    if name == "CheckpointCallback":
        from advanced_drivers._src.callbacks.checkpoint import CheckpointCallback
        return CheckpointCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
