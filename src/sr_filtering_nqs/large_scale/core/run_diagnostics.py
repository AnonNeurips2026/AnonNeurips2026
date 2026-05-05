"""
Checkpoint reconstruction helpers for offline large-scale diagnostics.

This module restores the J1J2 ViT path that the diagnostic scripts depend on.
It mirrors the active `large_scale/cli/train_sweep.py` configuration rather than
the archived training code.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import advanced_drivers as advd
import netket as nk
from nqs_nets.net.wrappers import ViTNd

from sr_filtering_nqs.large_scale.core.common import (
    LEARNING_RATE,
    VIT_D_MODEL,
    VIT_DEPTH,
    VIT_EXPANSION,
    VIT_HEADS,
    default_delta_bank_chunk_size_bwd,
    make_system,
    normalize_system_name,
)


def load_checkpoint(ckpt_path):
    with open(ckpt_path, "rb") as f:
        return pickle.load(f)


def resolve_n_samples_for_config(config, *, step: int | None = None) -> int:
    if "n_samples" in config and config["n_samples"] is not None:
        return int(config["n_samples"])

    phase_specs = config.get("phase_specs")
    if phase_specs:
        if step is None:
            step = int(config.get("step", 0))
        for spec in phase_specs:
            if step <= int(spec["step_end"]):
                if "n_samples" in spec:
                    return int(spec["n_samples"])
                break

    if "phase_samples" in config:
        phase_index = int(config.get("phase_index", 0))
        return int(config["phase_samples"][phase_index])

    raise ValueError(f"Unable to resolve n_samples from config keys={sorted(config.keys())}")


def _phase_spec_for_config(config, *, step: int | None = None) -> dict:
    phase_specs = config.get("phase_specs")
    if phase_specs:
        if step is None:
            step = int(config.get("step", 0))
        for spec in phase_specs:
            if step <= int(spec["step_end"]):
                return dict(spec)

    return {
        "phase_index": int(config.get("phase_index", 0)),
        "symmetry_index": int(config.get("symmetry_index", 0)),
        "name": config.get("phase_name", "phase"),
        "n_samples": resolve_n_samples_for_config(config, step=step),
    }


def make_vit_state(
    config,
    parameters,
    *,
    variables=None,
    step: int | None = None,
    n_samples_override: int | None = None,
    chunk_size_override: int | None = None,
    seed_override: int | None = None,
):
    system_name = normalize_system_name(config["system"])
    if system_name not in ("j1j2", "ssm"):
        raise ValueError(f"run_diagnostics.py only reconstructs ViT checkpoints, got {system_name}")

    system = make_system(system_name, L=int(config.get("lattice_size", 8)))
    phase_spec = _phase_spec_for_config(config, step=step)

    network = ViTNd(
        depth=int(config.get("vit_depth", VIT_DEPTH)),
        d_model=int(config.get("vit_d_model", VIT_D_MODEL)),
        heads=int(config.get("vit_heads", VIT_HEADS)),
        expansion_factor=int(config.get("vit_expansion", VIT_EXPANSION)),
        output_head=config.get("output_head", "Vanilla"),
        system=system,
    )
    base_model = network.network
    model = system.symmetrizing_functions[int(phase_spec["symmetry_index"])](base_model)

    n_samples = (
        int(n_samples_override)
        if n_samples_override is not None
        else resolve_n_samples_for_config(config, step=step)
    )
    sampler = nk.sampler.MetropolisExchange(
        system.hilbert_space,
        graph=system.graph,
        d_max=2,
        n_chains=n_samples,
    )

    chunk_size = chunk_size_override
    if chunk_size is None:
        chunk_size = config.get("chunk_size", 4096)
    if chunk_size is not None:
        chunk_size = min(int(chunk_size), n_samples)

    state_kwargs = {
        "n_samples": n_samples,
        "chunk_size": chunk_size,
    }
    if seed_override is not None:
        state_kwargs["seed"] = int(seed_override)

    vstate = nk.vqs.MCState(
        sampler,
        model=model,
        **state_kwargs,
    )
    if variables is not None:
        vstate.variables = variables
    else:
        vstate.parameters = parameters
    return system, phase_spec, vstate


def reconstruct_driver(
    config,
    parameters,
    *,
    variables=None,
    step: int | None = None,
    n_samples_override: int | None = None,
    chunk_size_override: int | None = None,
    chunk_size_bwd_override: int | None = None,
):
    system, _, vstate = make_vit_state(
        config,
        parameters,
        variables=variables,
        step=step,
        n_samples_override=n_samples_override,
        chunk_size_override=chunk_size_override,
    )
    system_name = normalize_system_name(config["system"])

    chunk_size_bwd = chunk_size_bwd_override
    if chunk_size_bwd is None:
        chunk_size_bwd = config.get("chunk_size_bwd")
    if chunk_size_bwd is None:
        chunk_size_bwd = default_delta_bank_chunk_size_bwd(
            system_name,
            n_samples=resolve_n_samples_for_config(config, step=step),
        )

    driver = advd.driver.VMC_NG(
        hamiltonian=system.hamiltonian,
        optimizer=nk.optimizer.Sgd(
            learning_rate=float(config.get("learning_rate", LEARNING_RATE))
        ),
        diag_shift=float(config["diag_shift"]),
        variational_state=vstate,
        chunk_size_bwd=chunk_size_bwd,
        use_ntk=True,
        linear_solver_fn={
            "cholesky": nk.optimizer.solver.cholesky,
            "pinv_smooth": nk.optimizer.solver.pinv_smooth,
            "pinv": nk.optimizer.solver.pinv,
            "LU": nk.optimizer.solver.LU,
        }[config.get("linear_solver", "cholesky")],
        mode=config.get("mode"),
    )
    driver.collect_residual_info = False
    return driver, vstate
