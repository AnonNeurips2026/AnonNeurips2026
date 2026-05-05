import os
from pathlib import Path


LATTICE_SIZE = 8
TRAIN_SAMPLES = 8192
PHASE_SAMPLES = (8192, 8192, 6000)
HELDOUT_SAMPLES = 100_000
LEARNING_RATE = 0.03
WARMUP_STEPS = 100
DEFAULT_SR_LAMBDA_GRID = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
SR_VARIANTS = ("single", "same_batch", "indep_batch")
SR_WEIGHT_SCHEMES = ("uniform", "gcv", "stacking")
NGD_METHODS = ("sr", "spring", "pii", "rgn")
LAMBDA_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
TFIM_LAMBDA_GRID = [
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    3.1622776601683795e-5,
    1e-4,
    3.1622776601683795e-4,
    1e-3,
    3.1622776601683795e-3,
    1e-2,
    3.1622776601683795e-2,
    1e-1,
]
TFIM_HELDOUT_SAMPLES = 120_000
TFIM_TOTAL_STEPS = 2000
TFIM_CHECKPOINT_EVERY = 100
TFIM_DIAGNOSTIC_STEPS = tuple(range(1100, TFIM_TOTAL_STEPS + 1, TFIM_CHECKPOINT_EVERY))
SHARED_CHECKPOINT_SOURCE_TRAIN_LAMBDA = 1e-4
SHARED_FIG2_DIAGNOSTIC_LAMBDAS = (
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
)
SHARED_FIG3_LAMBDA_GRIDS = {
    2: (1e-3, 1e-4),
    3: (1e-3, 1e-4, 1e-5),
    4: (1e-2, 1e-3, 1e-4, 1e-5),
    5: (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
}
J1J2_EARLY_DELTA_BANK_CHUNK_SIZE_BWD = 4096
J1J2_LATE_DELTA_BANK_CHUNK_SIZE_BWD = 2000
TFIM_DELTA_BANK_CHUNK_SIZE_BWD = 6000

VIT_DEPTH = 6
VIT_D_MODEL = 48
VIT_HEADS = 8
VIT_EXPANSION = 4

PHASE_LENGTHS = (200, 3500, 3500)
TOTAL_STEPS = sum(PHASE_LENGTHS)
CHECKPOINT_STEPS = (
    40, 80, 120, 160, 200,
    900, 1600, 2300, 3000, 3700,
    4400, 5100, 5800, 6500, 7200,
)
PHASE0_CHECKPOINT_STEPS = (PHASE_LENGTHS[0],)
J1J2_DIAGNOSTIC_STEPS = tuple(step for step in CHECKPOINT_STEPS if step > PHASE_LENGTHS[0])

SYSTEM_ORDER = ("j1j2", "tfim")
ACTIVE_SYSTEM_CHOICES = ("j1j2", "tfim", "ising", "ising1d", "tfim_fnqs", "ssm", "shastry")
SYSTEM_TITLES = {
    "j1j2": r"$J_1$-$J_2$ (8$\times$8, $J_2/J_1 = 0.5$)",
    "tfim": r"1D TFIM FNQS ($L=100$, $h \in [0.8, 1.2]$)",
    "ssm": r"Shastry-Sutherland (8$\times$8, $J'/J = 0.8$)",
}
SYSTEM_MODEL_TYPES = {
    "j1j2": "vit",
    "tfim": "fnqs",
    "ssm": "vit",
}
PII_E0_ESTIMATES_PER_SITE = {
    "j1j2": -1.995843,
    "ssm": -2.386983,
    "tfim": -1.280757,
}
SYSTEM_PHASES = {
    "j1j2": (
        {"phase_index": 0, "symmetry_index": 0, "name": "identity", "length": PHASE_LENGTHS[0]},
        {"phase_index": 1, "symmetry_index": 1, "name": "translations", "length": PHASE_LENGTHS[1]},
        {"phase_index": 2, "symmetry_index": 2, "name": "translations_c4", "length": PHASE_LENGTHS[2]},
    ),
    "tfim": (
        {"phase_index": 0, "symmetry_index": 0, "name": "fnqs", "length": 2000},
    ),
    "ssm": (
        {"phase_index": 0, "symmetry_index": 0, "name": "identity", "length": PHASE_LENGTHS[0]},
        {"phase_index": 1, "symmetry_index": 1, "name": "c4", "length": PHASE_LENGTHS[1]},
        {"phase_index": 2, "symmetry_index": 2, "name": "full_point_group", "length": PHASE_LENGTHS[2]},
    ),
}

_SYSTEM_ALIASES = {
    "j1j2": "j1j2",
    "tfim": "tfim",
    "ising": "tfim",
    "ising1d": "tfim",
    "tfim_fnqs": "tfim",
    "ssm": "ssm",
    "shastry": "ssm",
    "shastry-sutherland": "ssm",
}


def normalize_system_name(name):
    key = name.lower()
    if key not in _SYSTEM_ALIASES:
        raise ValueError(f"Unknown system: {name}")
    return _SYSTEM_ALIASES[key]


def default_pii_tau(name, n_sites: int, *, scale: float = 1.01) -> float:
    """Default projected inverse-iteration shift for arXiv:2507.10835-style PII.

    The paper uses a shift below the ground-state energy. These experiments
    estimate that as ``scale * E0_est_per_site * n_sites``; with negative
    energies, ``scale=1.01`` places the shift slightly below the estimate.
    Environment variables ``PII_TAU_J1J2``, ``PII_TAU_SSM``, and
    ``PII_TAU_TFIM`` override the default.
    """
    normalized = normalize_system_name(name)
    env_name = f"PII_TAU_{normalized.upper()}"
    if env_name in os.environ:
        return float(os.environ[env_name])
    if normalized not in PII_E0_ESTIMATES_PER_SITE:
        raise ValueError(f"No PII ground-state estimate registered for system={name!r}")
    return float(scale) * float(PII_E0_ESTIMATES_PER_SITE[normalized]) * int(n_sites)


def legacy_system_names(name):
    normalized = normalize_system_name(name)
    if normalized == "ssm":
        return ("ssm", "shastry")
    return (normalized,)


def model_type_for_system(name):
    normalized = normalize_system_name(name)
    return SYSTEM_MODEL_TYPES[normalized]


def lambda_grid_for_system(name):
    normalized = normalize_system_name(name)
    if normalized == "tfim":
        return TFIM_LAMBDA_GRID
    return LAMBDA_GRID


def heldout_samples_for_system(name):
    normalized = normalize_system_name(name)
    if normalized == "tfim":
        return TFIM_HELDOUT_SAMPLES
    return HELDOUT_SAMPLES


def default_source_steps_for_system(name):
    normalized = normalize_system_name(name)
    if normalized == "tfim":
        return TFIM_DIAGNOSTIC_STEPS
    return J1J2_DIAGNOSTIC_STEPS


def canonical_shared_fig3_lambda_grid(k: int):
    k = int(k)
    if k not in SHARED_FIG3_LAMBDA_GRIDS:
        raise ValueError(
            f"Unsupported shared Figure 3 grid size k={k}; expected one of {sorted(SHARED_FIG3_LAMBDA_GRIDS)}"
        )
    return SHARED_FIG3_LAMBDA_GRIDS[k]


def shared_fig3_default_method_ids(k: int):
    k = int(k)
    return (
        "single_oracle",
        f"indep_single_lambda_k{k}",
        f"same_batch_multilambda_gcv_k{k}",
        f"indep_multilambda_uniform_k{k}",
        f"indep_multilambda_gcv_k{k}",
    )


def default_delta_bank_chunk_size_bwd(name, *, n_samples: int | None = None):
    normalized = normalize_system_name(name)
    if normalized == "tfim":
        return TFIM_DELTA_BANK_CHUNK_SIZE_BWD

    if normalized not in ("j1j2", "ssm"):
        return None

    if n_samples is None:
        return J1J2_LATE_DELTA_BANK_CHUNK_SIZE_BWD

    n_samples = int(n_samples)
    if (
        n_samples >= J1J2_EARLY_DELTA_BANK_CHUNK_SIZE_BWD
        and n_samples % J1J2_EARLY_DELTA_BANK_CHUNK_SIZE_BWD == 0
    ):
        return J1J2_EARLY_DELTA_BANK_CHUNK_SIZE_BWD
    if (
        n_samples >= J1J2_LATE_DELTA_BANK_CHUNK_SIZE_BWD
        and n_samples % J1J2_LATE_DELTA_BANK_CHUNK_SIZE_BWD == 0
    ):
        return J1J2_LATE_DELTA_BANK_CHUNK_SIZE_BWD
    return n_samples


def format_lambda_dir(diag_shift):
    if diag_shift <= 0:
        raise ValueError(f"Diagonal shift must be positive, got {diag_shift}")
    exponent = round(float(f"{diag_shift:e}".split("e")[-1]))
    exact_power = 10.0 ** exponent
    if abs(diag_shift - exact_power) <= max(1e-15, abs(diag_shift) * 1e-12):
        return f"{diag_shift:.0e}"
    return f"{diag_shift:.3e}"


def parse_sr_lambda_grid(raw):
    if raw is None:
        return tuple(float(x) for x in DEFAULT_SR_LAMBDA_GRID)
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = list(raw)
    if not values:
        raise ValueError("sr_lambda_grid must contain at least one value")
    grid = tuple(float(value) for value in values)
    if any(value <= 0.0 for value in grid):
        raise ValueError(f"All SR lambda-grid values must be positive, got {grid}")
    return grid


def lambda_grid_tag(values) -> str:
    grid = parse_sr_lambda_grid(values)
    return "__".join(format_lambda_dir(value) for value in grid)


def make_system(name, L=LATTICE_SIZE):
    from spin_vmc.system import ShastrySutherland, SquareHeisenberg

    normalized = normalize_system_name(name)
    if normalized == "j1j2":
        return SquareHeisenberg(
            L=L,
            J=[1.0, 0.5],
            sign_rule=[False, False],
            patching=True,
        )
    if normalized == "ssm":
        return ShastrySutherland(L=L, J=[1.0, 0.8])
    raise ValueError(f"Unknown system: {name}")


def _looks_like_large_scale_root(path: Path) -> bool:
    return (
        (path / "j1j2_results").exists()
        or (path / "tfim_results").exists()
        or (path / "ssm_results").exists()
    )


def large_scale_root_dir(base):
    base = Path(base)
    cursor = base
    if base.exists() and not base.is_dir():
        cursor = base.parent

    named_large_scale = None
    while True:
        if _looks_like_large_scale_root(cursor):
            return cursor
        if cursor.name == "large_scale" and named_large_scale is None:
            named_large_scale = cursor
        if cursor.parent == cursor:
            break
        cursor = cursor.parent

    if named_large_scale is not None:
        return named_large_scale
    return base / "large_scale"


def system_results_root(base_root, system):
    base_root = large_scale_root_dir(base_root)
    system_name = normalize_system_name(system)
    return base_root / f"{system_name}_results"


def system_mixture_results_root(base_root, system):
    base_root = large_scale_root_dir(base_root)
    system_name = normalize_system_name(system)
    return base_root / f"{system_name}_mixture_results"


def canonical_results_root_candidates(base_root, system):
    base_root = Path(base_root)
    system_name = normalize_system_name(system)
    candidates = [system_results_root(base_root, system_name)]
    if system_name == "j1j2":
        candidates.append(base_root / f"{system_name}_8x8")
    elif system_name == "tfim":
        candidates.append(base_root / "tfim_1d")
    seen = set()
    unique = []
    for path in candidates:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return tuple(unique)


def canonical_mixture_results_root_candidates(base_root, system):
    base_root = Path(base_root)
    system_name = normalize_system_name(system)
    candidates = [system_mixture_results_root(base_root, system_name)]
    seen = set()
    unique = []
    for path in candidates:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return tuple(unique)


def default_collect_dir(base):
    return large_scale_root_dir(base) / "results"


def phase_specs_for_system(name):
    normalized = normalize_system_name(name)
    phase_specs = []
    step_start = 0
    for spec in SYSTEM_PHASES[normalized]:
        spec_copy = dict(spec)
        spec_copy["step_start"] = step_start
        spec_copy["step_end"] = step_start + spec_copy["length"]
        phase_specs.append(spec_copy)
        step_start = spec_copy["step_end"]
    return tuple(phase_specs)


def phase_spec_for_step(name, step):
    normalized = normalize_system_name(name)
    for spec in phase_specs_for_system(normalized):
        if step <= spec["step_end"]:
            return spec
    return phase_specs_for_system(normalized)[-1]


def results_dir_for_run(base_root, system, diag_shift):
    base_root = large_scale_root_dir(base_root)
    system_name = normalize_system_name(system)
    lambda_dir = format_lambda_dir(diag_shift)
    return base_root / f"{system_name}_results" / lambda_dir / "results"


def results_dir_for_mixture_run(
    base_root,
    system,
    *,
    sr_variant,
    sr_weight_scheme,
    sr_lambda_grid,
):
    base_root = large_scale_root_dir(base_root)
    system_name = normalize_system_name(system)
    if sr_variant not in SR_VARIANTS or sr_variant == "single":
        raise ValueError(f"results_dir_for_mixture_run requires a mixture sr_variant, got {sr_variant}")
    if sr_weight_scheme not in SR_WEIGHT_SCHEMES:
        raise ValueError(f"Unsupported SR weight scheme: {sr_weight_scheme}")
    grid_tag = lambda_grid_tag(sr_lambda_grid)
    run_name = f"{sr_variant}__{sr_weight_scheme}__{grid_tag}"
    return base_root / f"{system_name}_mixture_results" / run_name / "results"
