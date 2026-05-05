from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NETKET_EXPERIMENTAL_SHARDING", "1")

from flax import serialization
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax

import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS
from sr_filtering_nqs.large_scale.core.common import parse_sr_lambda_grid


ROOT = Path(__file__).resolve().parents[2]

TFIM_SYSTEM = "tfim"
TFIM_MODEL_TYPE = "fnqs"
TFIM_LENGTH = 100
TFIM_H_MIN = 0.8
TFIM_H_MAX = 1.2

TFIM_TRAIN_H_COUNT = 6000
TFIM_TRAIN_SAMPLES = 12_000
TFIM_DIAGNOSTIC_H_COUNT = 6000
TFIM_DIAGNOSTIC_SAMPLES_PER_H = 20
TFIM_DIAGNOSTIC_SAMPLES = TFIM_DIAGNOSTIC_H_COUNT * TFIM_DIAGNOSTIC_SAMPLES_PER_H
TFIM_H_HELDOUT_SEED = 0

TFIM_TOTAL_STEPS = 2000
TFIM_CHECKPOINT_EVERY = 100
TFIM_CHECKPOINT_STEPS = tuple(
    range(TFIM_CHECKPOINT_EVERY, TFIM_TOTAL_STEPS + 1, TFIM_CHECKPOINT_EVERY)
)
TFIM_DIAGNOSTIC_STEPS = tuple(range(1100, TFIM_TOTAL_STEPS + 1, TFIM_CHECKPOINT_EVERY))
TFIM_LEARNING_RATE = 0.005
TFIM_USE_NTK = True
TFIM_ON_THE_FLY = True
TFIM_LINEAR_SOLVER = "cholesky"

TFIM_PATCH_SIZE = 4
TFIM_MODEL_CONFIG = {
    "num_layers": 6,
    "d_model": 72,
    "heads": 12,
    "b": TFIM_PATCH_SIZE,
    "n_coups": 1,
    "complex": False,
    "disorder": False,
    "transl_invariant": True,
    "two_dimensional": False,
}

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


def lambda_grid():
    return list(TFIM_LAMBDA_GRID)


def validation_sample_count(config: dict) -> int:
    return int(config["validation_h_count"]) * int(config["validation_samples_per_h"])


def model_config_from_run_config(config: dict) -> dict:
    model_config = dict(TFIM_MODEL_CONFIG)
    model_config.update(config.get("model_config", {}))
    return model_config


def _to_numpy_tree(tree):
    def convert(x):
        if isinstance(x, (jax.Array, np.ndarray)):
            return np.asarray(x)
        return x

    return jax.tree_util.tree_map(convert, tree)


def make_graph(*, length: int = TFIM_LENGTH):
    return nk.graph.Hypercube(length=length, n_dim=1, pbc=True)


def make_hilbert(*, length: int = TFIM_LENGTH):
    graph = make_graph(length=length)
    return nk.hilbert.Spin(0.5, N=graph.n_nodes)


def make_parameter_space(*, h_min: float = TFIM_H_MIN, h_max: float = TFIM_H_MAX, n_coups: int = 1):
    return nkf.ParameterSpace(N=n_coups, min=h_min, max=h_max)


def make_model(*, length: int = TFIM_LENGTH, model_config: dict | None = None):
    config = dict(TFIM_MODEL_CONFIG if model_config is None else model_config)
    patch_size = int(config["b"])
    if length % patch_size != 0:
        raise ValueError(f"length={length} must be divisible by patch_size={patch_size}")
    return ViTFNQS(
        L_eff=length // patch_size,
        **config,
    )


def make_sampler(hilbert, *, n_chains: int):
    return nk.sampler.MetropolisSampler(
        hilbert=hilbert,
        rule=nk.sampler.rules.LocalRule(),
        n_chains=n_chains,
        sweep_size=hilbert.size,
    )


def make_parameter_array(*, count: int, h_min: float, h_max: float):
    return jnp.linspace(h_min, h_max, count, dtype=jnp.float64).reshape(-1, 1)


def make_training_parameter_array(config: dict | None = None):
    if config is None:
        count = TFIM_TRAIN_H_COUNT
        h_min = TFIM_H_MIN
        h_max = TFIM_H_MAX
    else:
        count = int(config["train_h_count"])
        h_min = float(config["h_min"])
        h_max = float(config["h_max"])
    return make_parameter_array(count=count, h_min=h_min, h_max=h_max)


def make_operator(
    *,
    length: int = TFIM_LENGTH,
    h_min: float = TFIM_H_MIN,
    h_max: float = TFIM_H_MAX,
    n_coups: int = 1,
):
    hilbert = make_hilbert(length=length)
    parameter_space = make_parameter_space(h_min=h_min, h_max=h_max, n_coups=n_coups)

    ha_x = sum(nkf.operator.sigmax(hilbert, i) for i in range(hilbert.size))
    ha_zz = sum(
        nkf.operator.sigmaz(hilbert, i) @ nkf.operator.sigmaz(hilbert, (i + 1) % hilbert.size)
        for i in range(hilbert.size)
    )

    def create_operator(params):
        assert params.shape == (n_coups,)
        h = params[0]
        return -h * ha_x - ha_zz

    return nkf.operator.ParametrizedOperator(hilbert, parameter_space, create_operator)


def make_state(
    *,
    length: int = TFIM_LENGTH,
    h_min: float = TFIM_H_MIN,
    h_max: float = TFIM_H_MAX,
    n_coups: int = 1,
    model_config: dict | None = None,
    parameter_array=None,
    n_samples: int = TFIM_TRAIN_SAMPLES,
    seed: int = 0,
    chunk_size: int | None = None,
):
    hilbert = make_hilbert(length=length)
    sampler = make_sampler(hilbert, n_chains=n_samples)
    if parameter_array is None:
        parameter_array = make_parameter_array(
            count=TFIM_TRAIN_H_COUNT,
            h_min=h_min,
            h_max=h_max,
        )
    parameter_array = jnp.asarray(parameter_array, dtype=jnp.float64)
    vstate = nkf.FoundationalQuantumState(
        sampler=sampler,
        model=make_model(length=length, model_config=model_config),
        parameter_space=make_parameter_space(h_min=h_min, h_max=h_max, n_coups=n_coups),
        n_samples=n_samples,
        n_replicas=int(parameter_array.shape[0]),
        seed=seed,
        chunk_size=chunk_size,
    )
    vstate.parameter_array = parameter_array
    return vstate


def make_state_from_config(
    config: dict,
    *,
    parameter_array=None,
    n_samples: int | None = None,
    seed: int | None = None,
    chunk_size: int | None = None,
):
    if seed is None:
        seed = int(config.get("seed", 0))
    if chunk_size is None:
        chunk_size = int(config.get("chunk_size", config.get("n_samples", TFIM_TRAIN_SAMPLES)))
    if n_samples is None:
        n_samples = int(config["n_samples"])
    return make_state(
        length=int(config["length"]),
        h_min=float(config["h_min"]),
        h_max=float(config["h_max"]),
        n_coups=int(model_config_from_run_config(config).get("n_coups", 1)),
        model_config=model_config_from_run_config(config),
        parameter_array=parameter_array,
        n_samples=n_samples,
        seed=seed,
        chunk_size=chunk_size,
    )


def make_driver(
    *,
    diag_shift: float,
    variational_state,
    learning_rate: float = TFIM_LEARNING_RATE,
    chunk_size_bwd: int | None = None,
    use_ntk: bool = TFIM_USE_NTK,
    on_the_fly: bool = TFIM_ON_THE_FLY,
    linear_solver: str = TFIM_LINEAR_SOLVER,
    lr_schedule: str = "constant",
    total_steps: int = TFIM_TOTAL_STEPS,
    momentum: float | None = None,
    proj_reg: float | None = None,
    length: int = TFIM_LENGTH,
    h_min: float = TFIM_H_MIN,
    h_max: float = TFIM_H_MAX,
    n_coups: int = 1,
):
    if on_the_fly and not use_ntk:
        raise ValueError("on_the_fly=True requires use_ntk=True")
    if lr_schedule == "constant":
        optimizer = optax.sgd(learning_rate)
    elif lr_schedule == "cosine":
        optimizer = optax.sgd(
            optax.cosine_decay_schedule(
                init_value=float(learning_rate),
                decay_steps=int(total_steps),
            )
        )
    else:
        raise ValueError(f"Unsupported lr_schedule={lr_schedule!r}")
    driver = nkf.VMC_NG(
        make_operator(length=length, h_min=h_min, h_max=h_max, n_coups=n_coups),
        optimizer,
        variational_state=variational_state,
        diag_shift=diag_shift,
        proj_reg=proj_reg,
        momentum=momentum,
        chunk_size_bwd=chunk_size_bwd,
        linear_solver_fn=resolve_linear_solver(linear_solver),
        mode="real",
        use_ntk=use_ntk,
        on_the_fly=on_the_fly,
    )
    driver.collect_residual_info = False
    return driver


def make_driver_from_config(config: dict, *, variational_state):
    model_config = model_config_from_run_config(config)
    ngd_method = str(config.get("ngd_method", "sr"))
    return make_driver(
        diag_shift=float(config["diag_shift"]),
        variational_state=variational_state,
        learning_rate=float(config.get("learning_rate", TFIM_LEARNING_RATE)),
        chunk_size_bwd=int(config.get("chunk_size_bwd", config.get("n_samples", TFIM_TRAIN_SAMPLES))),
        use_ntk=bool(config.get("use_ntk", TFIM_USE_NTK)),
        on_the_fly=bool(config.get("on_the_fly", TFIM_ON_THE_FLY)),
        linear_solver=str(config.get("linear_solver", TFIM_LINEAR_SOLVER)),
        lr_schedule=str(config.get("lr_schedule", "constant")),
        total_steps=int(config.get("total_steps", TFIM_TOTAL_STEPS)),
        momentum=(
            float(config.get("spring_momentum", 0.9))
            if ngd_method == "spring"
            else None
        ),
        proj_reg=(
            None
            if ngd_method != "spring" or config.get("spring_proj_reg") is None
            else float(config["spring_proj_reg"])
        ),
        length=int(config["length"]),
        h_min=float(config["h_min"]),
        h_max=float(config["h_max"]),
        n_coups=int(model_config.get("n_coups", 1)),
    )


def resolve_linear_solver(name: str):
    return {
        "cholesky": nk.optimizer.solver.cholesky,
        "pinv_smooth": nk.optimizer.solver.pinv_smooth,
        "pinv": nk.optimizer.solver.pinv,
        "LU": nk.optimizer.solver.LU,
    }[name]


def count_parameters(params) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))


def heldout_h_values_path(base_root, config: dict | None = None) -> Path:
    from sr_filtering_nqs.large_scale.core.common import large_scale_root_dir

    if config is None:
        count = TFIM_DIAGNOSTIC_H_COUNT
        seed = TFIM_H_HELDOUT_SEED
    else:
        count = int(config["validation_h_count"])
        seed = int(config["heldout_h_seed"])
    return large_scale_root_dir(base_root) / "tfim_validation" / f"heldout_h_values_R{count}_seed{seed}.npy"


def load_or_create_heldout_h_values(base_root, config: dict | None = None):
    if config is None:
        count = TFIM_DIAGNOSTIC_H_COUNT
        seed = TFIM_H_HELDOUT_SEED
        h_min = TFIM_H_MIN
        h_max = TFIM_H_MAX
    else:
        count = int(config["validation_h_count"])
        seed = int(config["heldout_h_seed"])
        h_min = float(config["h_min"])
        h_max = float(config["h_max"])

    path = heldout_h_values_path(base_root, config)
    if path.exists():
        return jnp.asarray(np.load(path), dtype=jnp.float64)

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    h_values = rng.uniform(h_min, h_max, size=(count, 1))
    np.save(path, h_values)
    return jnp.asarray(h_values, dtype=jnp.float64)


def make_validation_state(base_root, *, variables, config: dict, seed: int = 0):
    h_values = load_or_create_heldout_h_values(base_root, config)
    vstate = make_state_from_config(
        config,
        parameter_array=h_values,
        n_samples=validation_sample_count(config),
        seed=seed,
        chunk_size=int(config.get("validation_chunk_size", config.get("chunk_size", config.get("n_samples", TFIM_TRAIN_SAMPLES)))),
    )
    vstate.variables = variables
    return vstate, h_values


def serialize_state(vstate):
    return _to_numpy_tree(serialization.to_state_dict(vstate))


def restore_training_state(state_dict, *, config: dict, seed: int | None = None):
    template = make_state_from_config(
        config,
        parameter_array=make_training_parameter_array(config),
        n_samples=int(config["n_samples"]),
        seed=int(config.get("seed", 0) if seed is None else seed),
        chunk_size=int(config.get("chunk_size", config.get("n_samples", TFIM_TRAIN_SAMPLES))),
    )
    return serialization.from_state_dict(template, state_dict)


def reconstruct_training_driver(ckpt_payload):
    config = ckpt_payload["config"]
    vstate = restore_training_state(
        ckpt_payload["state_dict"],
        config=config,
        seed=int(config.get("seed", 0)),
    )
    driver = make_driver_from_config(config, variational_state=vstate)
    return driver, vstate


def build_run_config(
    *,
    diag_shift: float,
    output_root,
    seed: int,
    length: int = TFIM_LENGTH,
    h_min: float = TFIM_H_MIN,
    h_max: float = TFIM_H_MAX,
    train_h_count: int = TFIM_TRAIN_H_COUNT,
    train_total_samples: int = TFIM_TRAIN_SAMPLES,
    validation_h_count: int = TFIM_DIAGNOSTIC_H_COUNT,
    validation_samples_per_h: int = TFIM_DIAGNOSTIC_SAMPLES_PER_H,
    learning_rate: float = TFIM_LEARNING_RATE,
    lr_schedule: str = "constant",
    use_ntk: bool = TFIM_USE_NTK,
    on_the_fly: bool = TFIM_ON_THE_FLY,
    linear_solver: str = TFIM_LINEAR_SOLVER,
    total_steps: int = TFIM_TOTAL_STEPS,
    checkpoint_every: int = TFIM_CHECKPOINT_EVERY,
    chunk_size: int | None = None,
    chunk_size_bwd: int | None = None,
    heldout_h_seed: int = TFIM_H_HELDOUT_SEED,
    sr_variant: str = "single",
    sr_weight_scheme: str = "uniform",
    sr_lambda_grid=None,
    sr_weight_fit_samples: int | None = None,
    sr_lambda_quantiles=None,
    sr_lambda_ema_rho: float | None = None,
    sr_weight_ema_rho: float | None = None,
    ngd_method: str = "sr",
    spring_momentum: float = 0.9,
    spring_proj_reg: float | None = None,
    pii_tau: float | None = None,
    pii_diag_shift: float = 0.0,
    init_from: str | None = None,
    init_from_step: int | None = None,
    model_config: dict | None = None,
):
    output_root = Path(output_root)
    if chunk_size is None:
        chunk_size = train_total_samples
    if chunk_size_bwd is None:
        chunk_size_bwd = train_total_samples

    merged_model_config = dict(TFIM_MODEL_CONFIG)
    if model_config is not None:
        merged_model_config.update(model_config)

    checkpoint_steps = list(range(checkpoint_every, total_steps + 1, checkpoint_every))
    diagnostic_steps = [step for step in checkpoint_steps if step >= 1100]
    parsed_sr_lambda_grid = [float(x) for x in parse_sr_lambda_grid(sr_lambda_grid)]

    return {
        "system": TFIM_SYSTEM,
        "model_type": TFIM_MODEL_TYPE,
        "diag_shift": float(diag_shift),
        "sr_variant": str(sr_variant),
        "sr_weight_scheme": str(sr_weight_scheme),
        "sr_lambda_grid": parsed_sr_lambda_grid,
        "sr_weight_fit_samples": (
            None if sr_weight_fit_samples is None else int(sr_weight_fit_samples)
        ),
        "sr_lambda_quantiles": (
            None
            if sr_lambda_quantiles is None
            else [float(x) for x in parse_sr_lambda_grid(sr_lambda_quantiles)]
        ),
        "sr_lambda_ema_rho": None if sr_lambda_ema_rho is None else float(sr_lambda_ema_rho),
        "sr_weight_ema_rho": None if sr_weight_ema_rho is None else float(sr_weight_ema_rho),
        "ngd_method": str(ngd_method),
        "spring_momentum": float(spring_momentum),
        "spring_proj_reg": None if spring_proj_reg is None else float(spring_proj_reg),
        "pii_tau": None if pii_tau is None else float(pii_tau),
        "pii_diag_shift": float(pii_diag_shift),
        "init_from": init_from,
        "init_from_step": init_from_step,
        "output_root": str(output_root),
        "length": int(length),
        "h_min": float(h_min),
        "h_max": float(h_max),
        "train_h_count": int(train_h_count),
        "train_total_samples": int(train_total_samples),
        "train_samples_per_h": float(train_total_samples) / float(train_h_count),
        "validation_h_count": int(validation_h_count),
        "validation_samples_per_h": int(validation_samples_per_h),
        "validation_n_samples": int(validation_h_count * validation_samples_per_h),
        "heldout_h_seed": int(heldout_h_seed),
        "learning_rate": float(learning_rate),
        "lr_schedule": str(lr_schedule),
        "use_ntk": bool(use_ntk),
        "on_the_fly": bool(on_the_fly),
        "linear_solver": str(linear_solver),
        "total_steps": int(total_steps),
        "checkpoint_every": int(checkpoint_every),
        "checkpoint_steps": checkpoint_steps,
        "diagnostic_steps": diagnostic_steps,
        "chunk_size": int(chunk_size),
        "validation_chunk_size": int(chunk_size),
        "chunk_size_bwd": int(chunk_size_bwd),
        "n_samples": int(train_total_samples),
        "n_chains": int(train_total_samples),
        "n_replicas": int(train_h_count),
        "seed": int(seed),
        "mode": "real",
        "phase_index": 0,
        "phase_name": "fnqs",
        "symmetry_index": 0,
        "model_config": merged_model_config,
    }
