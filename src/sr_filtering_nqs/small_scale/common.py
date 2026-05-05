from __future__ import annotations

import copy
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = PROJECT_ROOT / "nqs_support_core" / "packages"
if str(LOCAL_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGES))

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.experimental import sparse as jsparse

import netket as nk
from netket import jax as nkjax


EPS = 1e-30
DEFAULT_PAIRED_NOISE_FRACTION_MIN = 0.30
SPARSE_OPERATOR_CACHE: dict[int, jsparse.BCOO] = {}
VIT_MODEL_RE = re.compile(
    r"^ViT(?:_d(?P<depth>\d+)_m(?P<d_model>\d+)_h(?P<heads>\d+)_e(?P<expansion>\d+))?$"
)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def copy_variables(variables: dict[str, Any]) -> dict[str, Any]:
    """Copy JAX/NumPy variable pytrees into plain NumPy-backed trees."""
    return jax.tree_util.tree_map(lambda x: np.array(x, copy=True), variables)


def pytree_isfinite(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)


def get_sparse_hamiltonian_bcoo(hamiltonian: Any) -> jsparse.BCOO:
    cache_key = id(hamiltonian)
    cached = SPARSE_OPERATOR_CACHE.get(cache_key)
    if cached is None:
        cached = jsparse.BCOO.from_scipy_sparse(hamiltonian.to_sparse().astype(np.complex128))
        SPARSE_OPERATOR_CACHE[cache_key] = cached
    return cached


def make_tfim_chain(
    L: int,
    h: float = 1.0,
    J: float = 1.0,
) -> tuple[nk.hilbert.Spin, Any, Any, dict[str, Any]]:
    graph = nk.graph.Chain(length=L, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
    hamiltonian = nk.operator.Ising(hilbert=hilbert, graph=graph, h=h, J=J)
    meta = {
        "name": "TFIM1D",
        "L": int(L),
        "n_spins": int(graph.n_nodes),
        "dim_h": int(hilbert.n_states),
        "h": float(h),
        "J": float(J),
    }
    return hilbert, hamiltonian, graph, meta


def make_square_j1j2(
    L: int = 4,
    J1: float = 1.0,
    J2: float = 0.5,
) -> tuple[nk.hilbert.Spin, Any, Any, dict[str, Any]]:
    graph = nk.graph.Square(L, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)

    edges_nn = list(graph.edges())
    edges_nnn: list[tuple[int, int]] = []
    for x in range(L):
        for y in range(L):
            i = x * L + y
            for dx, dy in ((1, 1), (1, -1)):
                nx_, ny_ = (x + dx) % L, (y + dy) % L
                j = nx_ * L + ny_
                if i < j:
                    edges_nnn.append((i, j))

    bond = np.array(
        [[1, 0, 0, 0], [0, -1, 2, 0], [0, 2, -1, 0], [0, 0, 0, 1]],
        dtype=float,
    ) / 4.0
    hamiltonian = nk.operator.GraphOperator(
        hilbert,
        graph=nk.graph.Graph(edges=edges_nn),
        bond_ops=[J1 * bond],
    )
    hamiltonian += nk.operator.GraphOperator(
        hilbert,
        graph=nk.graph.Graph(edges=edges_nnn),
        bond_ops=[J2 * bond],
    )
    meta = {
        "name": "J1J2Square",
        "L": int(L),
        "n_spins": int(graph.n_nodes),
        "dim_h": int(hilbert.n_states),
        "J1": float(J1),
        "J2": float(J2),
    }
    return hilbert, hamiltonian, graph, meta


def make_model(
    model_name: str,
    *,
    hilbert: Any,
    graph: Any,
    square_L: int | None = None,
    j_couplings: Iterable[float] | None = None,
) -> tuple[Any, str]:
    if model_name == "Jastrow":
        return nk.models.Jastrow(param_dtype=complex), "Jastrow"

    if model_name.startswith("RBM_a"):
        alpha = int(model_name.split("_a")[1])
        return nk.models.RBM(alpha=alpha, param_dtype=complex), f"RBM alpha={alpha}"

    vit_match = VIT_MODEL_RE.fullmatch(model_name)
    if vit_match is not None:
        from nqs_nets.net.wrappers import ViTNd
        from spin_vmc.system import SquareHeisenberg

        if square_L is None:
            raise ValueError("square_L is required for ViT models.")
        if j_couplings is None:
            j_couplings = (1.0, 0.5)
        if vit_match.group("depth") is None:
            depth, d_model, heads, expansion_factor = 2, 16, 4, 4
        else:
            depth = int(vit_match.group("depth"))
            d_model = int(vit_match.group("d_model"))
            heads = int(vit_match.group("heads"))
            expansion_factor = int(vit_match.group("expansion"))
        system = SquareHeisenberg(L=square_L, J=list(j_couplings))
        network = ViTNd(
            depth=depth,
            d_model=d_model,
            heads=heads,
            expansion_factor=expansion_factor,
            output_head="Vanilla",
            system=system,
        )
        return (
            network.network,
            f"ViT(d={depth},m={d_model},h={heads},e={expansion_factor})",
        )

    raise ValueError(f"Unknown model: {model_name}")


def count_real_params(params: Any) -> int:
    params_real, _ = nkjax.tree_to_real(params)
    flat, _ = ravel_pytree(params_real)
    return int(flat.size)


def build_delta_tree_converter(variables: dict[str, Any]):
    params_real, to_complex = nkjax.tree_to_real(variables["params"])
    _, unflatten = ravel_pytree(params_real)
    other_vars = {k: v for k, v in variables.items() if k != "params"}

    def delta_to_variables(delta_flat: np.ndarray | jnp.ndarray) -> dict[str, Any]:
        delta_real_tree = unflatten(jnp.asarray(delta_flat))
        delta_params = to_complex(delta_real_tree)
        return {"params": delta_params, **other_vars}

    return delta_to_variables


def compute_born_distribution(
    model: Any,
    variables: dict[str, Any],
    all_states: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> tuple[jax.Array, jax.Array]:
    log_psi_chunks = []
    for start in range(0, len(all_states), chunk_size):
        stop = min(start + chunk_size, len(all_states))
        states_chunk = jnp.asarray(all_states[start:stop])
        log_psi_chunks.append(model.apply(variables, states_chunk))

    log_psi = jnp.concatenate(log_psi_chunks, axis=0)
    log_prob = 2.0 * jnp.real(log_psi)
    log_prob -= jnp.max(log_prob)
    pi = jnp.exp(log_prob)
    pi /= jnp.sum(pi)
    return pi, log_psi


def compute_local_energies(
    hamiltonian: Any,
    log_psi: jax.Array,
) -> jax.Array:
    h_sparse = get_sparse_hamiltonian_bcoo(hamiltonian)
    psi = jnp.exp(log_psi - jnp.max(jnp.real(log_psi)))
    h_psi = h_sparse @ psi
    return jnp.real(h_psi / psi)


def compute_jvp_predictions(
    model: Any,
    variables: dict[str, Any],
    all_states: np.ndarray,
    delta_variables: dict[str, Any],
    *,
    chunk_size: int = 4096,
    return_numpy: bool = True,
) -> np.ndarray | jax.Array:
    base_params = variables["params"]
    other_vars = {k: v for k, v in variables.items() if k != "params"}
    delta_params = delta_variables["params"]

    @jax.jit
    def jvp_chunk(sample_chunk: jnp.ndarray) -> jnp.ndarray:
        def apply_fn(params_tree):
            return model.apply({"params": params_tree, **other_vars}, sample_chunk)

        _, pred = jax.jvp(apply_fn, (base_params,), (delta_params,))
        return jnp.real(pred)

    preds = []
    for start in range(0, len(all_states), chunk_size):
        stop = min(start + chunk_size, len(all_states))
        pred_chunk = jvp_chunk(jnp.asarray(all_states[start:stop]))
        preds.append(np.asarray(pred_chunk) if return_numpy else pred_chunk)
    if return_numpy:
        return np.concatenate(preds, axis=0)
    return jnp.concatenate(preds, axis=0)


def compute_jacobian_chunked(
    model: Any,
    variables: dict[str, Any],
    samples: np.ndarray,
    *,
    chunk_size: int = 128,
    return_numpy: bool = True,
) -> np.ndarray | jax.Array:
    params = variables["params"]
    other_vars = {k: v for k, v in variables.items() if k != "params"}
    params_real, to_complex = nkjax.tree_to_real(params)

    def log_psi_single(p_real_tree, x):
        p_complex = to_complex(p_real_tree)
        out = model.apply({"params": p_complex, **other_vars}, x[None])
        return jnp.real(out[0])

    grad_fn = jax.grad(log_psi_single)

    @jax.jit
    def jac_chunk(p_real_tree, sample_chunk):
        grads_tree = jax.vmap(grad_fn, in_axes=(None, 0))(p_real_tree, sample_chunk)
        return jax.vmap(lambda g: ravel_pytree(g)[0])(grads_tree)

    chunks = []
    for start in range(0, len(samples), chunk_size):
        stop = min(start + chunk_size, len(samples))
        sample_chunk = jnp.asarray(samples[start:stop])
        jacobian_chunk = jac_chunk(params_real, sample_chunk)
        chunks.append(np.asarray(jacobian_chunk) if return_numpy else jacobian_chunk)
    if return_numpy:
        return np.concatenate(chunks, axis=0)
    return jnp.concatenate(chunks, axis=0)


def safe_linear_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def solve_ridge_kernel(
    o_centered: np.ndarray,
    y_centered: np.ndarray,
    lam: float,
    n_samples: int,
) -> np.ndarray:
    kernel = (o_centered @ o_centered.T) / n_samples
    alpha = safe_linear_solve(
        kernel + lam * np.eye(n_samples, dtype=kernel.dtype),
        y_centered,
    )
    return (o_centered.T @ alpha) / n_samples


def exact_feature_moments(
    model: Any,
    variables: dict[str, Any],
    all_states: np.ndarray,
    pi: jax.Array,
    target_hloc: jax.Array,
    *,
    p_real: int,
    jac_chunk_size: int = 128,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    o_mean = jnp.zeros((p_real,), dtype=jnp.float64)
    second_moment = jnp.zeros((p_real, p_real), dtype=jnp.float64)
    g_raw = jnp.zeros((p_real,), dtype=jnp.float64)
    target_mean = jnp.array(0.0, dtype=jnp.float64)
    for start in range(0, len(all_states), jac_chunk_size):
        stop = min(start + jac_chunk_size, len(all_states))
        o_chunk = compute_jacobian_chunked(
            model,
            variables,
            all_states[start:stop],
            chunk_size=stop - start,
            return_numpy=False,
        )
        pi_chunk = pi[start:stop]
        target_chunk = target_hloc[start:stop]
        weighted_pi = pi_chunk[:, None]
        o_mean += jnp.sum(weighted_pi * o_chunk, axis=0)
        second_moment += o_chunk.T @ (weighted_pi * o_chunk)
        g_raw += o_chunk.T @ (pi_chunk * target_chunk)
        target_mean += jnp.sum(pi_chunk * target_chunk)

    s_exact = second_moment - jnp.outer(o_mean, o_mean)
    g_exact = g_raw - o_mean * target_mean
    return o_mean, s_exact, g_exact


def eigh_descending(matrix: np.ndarray | jax.Array) -> tuple[jax.Array, jax.Array]:
    eigenvalues, eigenvectors = jnp.linalg.eigh(jnp.asarray(matrix))
    order = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = jnp.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    return eigenvalues, eigenvectors


def pseudoinverse_solution(
    eigenvalues: np.ndarray | jax.Array,
    eigenvectors: np.ndarray | jax.Array,
    rhs: np.ndarray | jax.Array,
    *,
    rcond: float = 1e-10,
) -> jax.Array:
    if eigenvalues.size == 0:
        return jnp.zeros_like(rhs)
    eigenvalues = jnp.asarray(eigenvalues)
    eigenvectors = jnp.asarray(eigenvectors)
    rhs = jnp.asarray(rhs)
    threshold = jnp.maximum(jnp.max(eigenvalues), jnp.array(1.0, dtype=eigenvalues.dtype)) * rcond
    rhs_eig = eigenvectors.T @ rhs
    coeffs = jnp.zeros_like(rhs_eig)
    mask = eigenvalues > threshold
    safe_denominator = jnp.where(mask, eigenvalues, 1.0)
    coeffs = jnp.where(mask, rhs_eig / safe_denominator, 0.0)
    return eigenvectors @ coeffs


def compute_exact_regression_diagnostics(
    model: Any,
    variables: dict[str, Any],
    hamiltonian: Any,
    all_states: np.ndarray,
    *,
    jac_chunk_size: int = 128,
    apply_chunk_size: int = 4096,
    store_arrays: bool = False,
    store_matrix: bool = False,
) -> dict[str, Any]:
    p_real = count_real_params(variables["params"])
    pi, log_psi = compute_born_distribution(
        model,
        variables,
        all_states,
        chunk_size=apply_chunk_size,
    )
    eloc = compute_local_energies(hamiltonian, log_psi)
    energy = float(jnp.sum(pi * eloc))
    target_hloc = eloc - energy
    target_var = float(jnp.sum(pi * target_hloc**2))

    feature_mean, s_exact, g_exact = exact_feature_moments(
        model,
        variables,
        all_states,
        pi,
        target_hloc,
        p_real=p_real,
        jac_chunk_size=jac_chunk_size,
    )
    eigenvalues, eigenvectors = eigh_descending(s_exact)
    delta_star = pseudoinverse_solution(eigenvalues, eigenvectors, g_exact)
    explained = float(jnp.dot(g_exact, delta_star))
    sigma_gap_sq = max(0.0, target_var - explained)
    noise_fraction = sigma_gap_sq / max(target_var, EPS)

    result = {
        "energy": energy,
        "target_var": target_var,
        "sigma_gap_sq": sigma_gap_sq,
        "noise_fraction": noise_fraction,
        "eigenvalues": np.asarray(eigenvalues, dtype=np.float64),
        "delta_star_flat": np.asarray(delta_star, dtype=np.float64),
        "feature_mean": np.asarray(feature_mean, dtype=np.float64),
        "p_real": p_real,
        "timing_seconds": None,
    }
    if store_matrix:
        result["S_exact"] = np.asarray(s_exact, dtype=np.float64)

    if store_arrays:
        target_ideal = compute_jvp_predictions(
            model,
            variables,
            all_states,
            build_delta_tree_converter(variables)(np.asarray(delta_star)),
            chunk_size=apply_chunk_size,
            return_numpy=False,
        )
        target_ideal -= jnp.sum(pi * target_ideal)
        expressivity_gap = target_hloc - target_ideal
        result.update(
            {
                "pi": np.asarray(pi, dtype=np.float32),
                "log_psi_real": np.asarray(jnp.real(log_psi), dtype=np.float32),
                "target_hloc": np.asarray(target_hloc, dtype=np.float32),
                "target_ideal": np.asarray(target_ideal, dtype=np.float32),
                "expressivity_gap": np.asarray(expressivity_gap, dtype=np.float32),
                "explicit_sigma_gap_sq": float(jnp.sum(pi * expressivity_gap**2)),
            }
        )
    return result


def precompute_centered_feature_matrix(
    model: Any,
    variables: dict[str, Any],
    all_states: np.ndarray,
    feature_mean: np.ndarray,
    *,
    jac_chunk_size: int,
) -> jax.Array:
    feature_mean_jax = jnp.asarray(feature_mean)
    chunks = []
    for start in range(0, len(all_states), jac_chunk_size):
        stop = min(start + jac_chunk_size, len(all_states))
        o_chunk = compute_jacobian_chunked(
            model,
            variables,
            all_states[start:stop],
            chunk_size=stop - start,
            return_numpy=False,
        )
        chunks.append(o_chunk - feature_mean_jax[None, :])
    return jnp.concatenate(chunks, axis=0)


def checkpoint_metric_value(
    checkpoint_summary: dict[str, Any],
    metric_key: str,
) -> float:
    if metric_key not in ("noise_fraction", "sigma_gap_sq"):
        raise ValueError(f"Unsupported checkpoint metric: {metric_key}")
    return float(checkpoint_summary[metric_key])


def resolve_match_min_value(
    *,
    selection_mode: str,
    selection_metric: str,
    match_min_value: float | None,
) -> float | None:
    if match_min_value is not None:
        return float(match_min_value)
    if selection_mode == "paired_match" and selection_metric == "noise_fraction":
        return DEFAULT_PAIRED_NOISE_FRACTION_MIN
    return None


def select_checkpoint_band(
    checkpoint_summaries: list[dict[str, Any]],
    *,
    band_low: float,
    band_high: float,
    target_noise: float,
) -> tuple[int, str]:
    in_band = [
        idx
        for idx, item in enumerate(checkpoint_summaries)
        if band_low <= item["noise_fraction"] <= band_high
    ]
    if in_band:
        return in_band[0], "in_band"

    best_idx = min(
        range(len(checkpoint_summaries)),
        key=lambda idx: abs(checkpoint_summaries[idx]["noise_fraction"] - target_noise),
    )
    return best_idx, "closest_to_target"


def select_checkpoint_target(
    checkpoint_summaries: list[dict[str, Any]],
    *,
    metric_key: str,
    target_value: float | None,
) -> tuple[int, str]:
    if not checkpoint_summaries:
        raise ValueError("No checkpoint summaries available for selection.")

    if target_value is None:
        best_idx = max(
            range(len(checkpoint_summaries)),
            key=lambda idx: checkpoint_metric_value(checkpoint_summaries[idx], metric_key),
        )
        return best_idx, "largest_metric"

    best_idx = min(
        range(len(checkpoint_summaries)),
        key=lambda idx: (
            abs(checkpoint_metric_value(checkpoint_summaries[idx], metric_key) - target_value),
            -checkpoint_metric_value(checkpoint_summaries[idx], metric_key),
        ),
    )
    return best_idx, "closest_to_target"


def select_paired_checkpoints(
    checkpoint_groups: list[list[dict[str, Any]]],
    *,
    metric_key: str,
    match_min_value: float | None,
    match_target_value: float | None,
) -> tuple[list[int], dict[str, Any]]:
    if len(checkpoint_groups) != 2:
        raise ValueError("Paired checkpoint matching requires exactly two models.")

    candidates = []
    for idx_a, item_a in enumerate(checkpoint_groups[0]):
        value_a = checkpoint_metric_value(item_a, metric_key)
        for idx_b, item_b in enumerate(checkpoint_groups[1]):
            value_b = checkpoint_metric_value(item_b, metric_key)
            mean_value = 0.5 * (value_a + value_b)
            abs_diff = abs(value_a - value_b)
            rel_diff = abs_diff / max(abs(mean_value), EPS)
            min_value = min(value_a, value_b)
            candidates.append(
                {
                    "indices": [int(idx_a), int(idx_b)],
                    "metric_values": [float(value_a), float(value_b)],
                    "mean_value": float(mean_value),
                    "min_value": float(min_value),
                    "abs_diff": float(abs_diff),
                    "rel_diff": float(rel_diff),
                    "passes_min_value": (
                        True if match_min_value is None else min_value >= match_min_value
                    ),
                    "target_distance": (
                        None
                        if match_target_value is None
                        else abs(mean_value - match_target_value)
                    ),
                }
            )

    if not candidates:
        raise ValueError("No candidate checkpoint pairs were generated.")

    qualified = [item for item in candidates if item["passes_min_value"]]
    pool = qualified if qualified else candidates

    if match_target_value is not None:
        best = min(
            pool,
            key=lambda item: (
                item["target_distance"],
                item["rel_diff"],
                -item["min_value"],
                -item["mean_value"],
                item["abs_diff"],
            ),
        )
        reason = "paired_target_match"
    else:
        best = min(
            pool,
            key=lambda item: (
                item["rel_diff"],
                -item["min_value"],
                -item["mean_value"],
                item["abs_diff"],
            ),
        )
        reason = "paired_closest_high_metric"

    if qualified is not pool:
        reason += "_fallback_no_pair_above_min"

    metadata = {
        "reason": reason,
        "metric": metric_key,
        "match_min_value": None if match_min_value is None else float(match_min_value),
        "match_target_value": None if match_target_value is None else float(match_target_value),
        "selected_indices": list(best["indices"]),
        "selected_metric_values": list(best["metric_values"]),
        "selected_mean_value": float(best["mean_value"]),
        "selected_min_value": float(best["min_value"]),
        "selected_abs_diff": float(best["abs_diff"]),
        "selected_rel_diff": float(best["rel_diff"]),
    }
    return list(best["indices"]), metadata


def pair_metric_summary(
    item_a: dict[str, Any],
    item_b: dict[str, Any],
    *,
    metric_key: str,
) -> dict[str, float | list[float]]:
    value_a = checkpoint_metric_value(item_a, metric_key)
    value_b = checkpoint_metric_value(item_b, metric_key)
    mean_value = 0.5 * (value_a + value_b)
    abs_diff = abs(value_a - value_b)
    rel_diff = abs_diff / max(abs(mean_value), EPS)
    return {
        "values": [float(value_a), float(value_b)],
        "mean_value": float(mean_value),
        "min_value": float(min(value_a, value_b)),
        "abs_diff": float(abs_diff),
        "rel_diff": float(rel_diff),
    }


def select_paired_checkpoints_sigma_priority(
    checkpoint_groups: list[list[dict[str, Any]]],
    *,
    match_min_value: float | None,
    match_target_value: float | None,
) -> tuple[list[int], dict[str, Any]]:
    if len(checkpoint_groups) != 2:
        raise ValueError("Paired checkpoint matching requires exactly two models.")

    candidates = []
    for idx_a, item_a in enumerate(checkpoint_groups[0]):
        for idx_b, item_b in enumerate(checkpoint_groups[1]):
            sigma = pair_metric_summary(item_a, item_b, metric_key="sigma_gap_sq")
            noise = pair_metric_summary(item_a, item_b, metric_key="noise_fraction")
            candidates.append(
                {
                    "indices": [int(idx_a), int(idx_b)],
                    "sigma_gap_sq_values": sigma["values"],
                    "sigma_gap_sq_mean": sigma["mean_value"],
                    "sigma_gap_sq_min": sigma["min_value"],
                    "sigma_gap_sq_abs_diff": sigma["abs_diff"],
                    "sigma_gap_sq_rel_diff": sigma["rel_diff"],
                    "noise_fraction_values": noise["values"],
                    "noise_fraction_mean": noise["mean_value"],
                    "noise_fraction_min": noise["min_value"],
                    "noise_fraction_abs_diff": noise["abs_diff"],
                    "noise_fraction_rel_diff": noise["rel_diff"],
                    "passes_min_value": (
                        True
                        if match_min_value is None
                        else sigma["min_value"] >= match_min_value
                    ),
                    "target_distance": (
                        None
                        if match_target_value is None
                        else abs(sigma["mean_value"] - match_target_value)
                    ),
                }
            )

    if not candidates:
        raise ValueError("No candidate checkpoint pairs were generated.")

    qualified = [item for item in candidates if item["passes_min_value"]]
    pool = qualified if qualified else candidates

    if match_target_value is not None:
        sort_key = lambda item: (
            item["target_distance"],
            item["sigma_gap_sq_rel_diff"],
            item["sigma_gap_sq_abs_diff"],
            item["noise_fraction_rel_diff"],
            item["noise_fraction_abs_diff"],
            -item["sigma_gap_sq_mean"],
        )
        reason = "paired_sigma_priority_target_match"
    else:
        sort_key = lambda item: (
            item["sigma_gap_sq_rel_diff"],
            item["sigma_gap_sq_abs_diff"],
            item["noise_fraction_rel_diff"],
            item["noise_fraction_abs_diff"],
            -item["sigma_gap_sq_mean"],
        )
        reason = "paired_sigma_priority_closest"

    ranked_pool = sorted(pool, key=sort_key)
    best = ranked_pool[0]
    if qualified is not pool:
        reason += "_fallback_no_pair_above_min"

    metadata = {
        "reason": reason,
        "metric": "sigma_gap_sq",
        "secondary_metric": "noise_fraction",
        "match_min_value": None if match_min_value is None else float(match_min_value),
        "match_target_value": None if match_target_value is None else float(match_target_value),
        "candidate_pairs_evaluated": len(candidates),
        "qualified_pairs": len(qualified),
        "selected_indices": list(best["indices"]),
        "selected_sigma_gap_sq_values": list(best["sigma_gap_sq_values"]),
        "selected_sigma_gap_sq_mean": float(best["sigma_gap_sq_mean"]),
        "selected_sigma_gap_sq_abs_diff": float(best["sigma_gap_sq_abs_diff"]),
        "selected_sigma_gap_sq_rel_diff": float(best["sigma_gap_sq_rel_diff"]),
        "selected_noise_fraction_values": list(best["noise_fraction_values"]),
        "selected_noise_fraction_mean": float(best["noise_fraction_mean"]),
        "selected_noise_fraction_abs_diff": float(best["noise_fraction_abs_diff"]),
        "selected_noise_fraction_rel_diff": float(best["noise_fraction_rel_diff"]),
        "top_candidates": ranked_pool[:10],
    }
    return list(best["indices"]), metadata


def summarize_trials(
    lambda_grid: np.ndarray,
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_names = ("r_train_hloc", "r_pop_hloc", "r_pop_ideal", "gap_ratio")
    n_trials = len(trials)
    summary: dict[str, Any] = {
        "lambda_grid": np.asarray(lambda_grid, dtype=np.float64),
        "n_trials": n_trials,
    }
    for metric in metric_names:
        values = np.asarray([trial["metrics"][metric] for trial in trials], dtype=np.float64)
        summary[f"{metric}_mean"] = np.mean(values, axis=0)
        summary[f"{metric}_std"] = np.std(values, axis=0)
        if n_trials > 1:
            summary[f"{metric}_stderr"] = np.std(values, axis=0, ddof=1) / np.sqrt(n_trials)
        else:
            summary[f"{metric}_stderr"] = np.zeros_like(summary[f"{metric}_std"])

    best_idx = int(np.argmin(summary["r_pop_ideal_mean"]))
    summary["lambda_best_idx"] = best_idx
    summary["lambda_best"] = float(lambda_grid[best_idx])
    summary["lambda_small_idx"] = 0
    summary["lambda_small"] = float(lambda_grid[0])
    summary["lambda_last_idx"] = len(lambda_grid) - 1
    summary["lambda_last"] = float(lambda_grid[-1])
    return summary


def representative_predictions(
    model: Any,
    variables: dict[str, Any],
    all_states: np.ndarray,
    pi: np.ndarray,
    delta_to_variables,
    delta_small: np.ndarray,
    delta_best: np.ndarray,
    delta_last: np.ndarray,
    *,
    apply_chunk_size: int = 4096,
    feature_matrix_centered: jax.Array | None = None,
) -> dict[str, Any]:
    if feature_matrix_centered is not None:
        pi_jax = jnp.asarray(pi, dtype=jnp.float64)
        pred_small = feature_matrix_centered @ jnp.asarray(delta_small)
        pred_best = feature_matrix_centered @ jnp.asarray(delta_best)
        pred_last = feature_matrix_centered @ jnp.asarray(delta_last)
        pred_small -= jnp.sum(pi_jax * pred_small)
        pred_best -= jnp.sum(pi_jax * pred_best)
        pred_last -= jnp.sum(pi_jax * pred_last)
        pred_small = np.asarray(pred_small, dtype=np.float32)
        pred_best = np.asarray(pred_best, dtype=np.float32)
        pred_last = np.asarray(pred_last, dtype=np.float32)
    else:
        pred_small = compute_jvp_predictions(
            model,
            variables,
            all_states,
            delta_to_variables(delta_small),
            chunk_size=apply_chunk_size,
        )
        pred_best = compute_jvp_predictions(
            model,
            variables,
            all_states,
            delta_to_variables(delta_best),
            chunk_size=apply_chunk_size,
        )
        pred_last = compute_jvp_predictions(
            model,
            variables,
            all_states,
            delta_to_variables(delta_last),
            chunk_size=apply_chunk_size,
        )
        pred_small -= np.sum(pi * pred_small)
        pred_best -= np.sum(pi * pred_best)
        pred_last -= np.sum(pi * pred_last)
    return {
        "pred_small": pred_small.astype(np.float32),
        "pred_best": pred_best.astype(np.float32),
        "pred_last": pred_last.astype(np.float32),
    }


def solve_ridge_lambda_sweep(
    o_centered: np.ndarray | jax.Array,
    y_centered: np.ndarray | jax.Array,
    lambda_grid: np.ndarray,
    n_samples: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    o_centered = jnp.asarray(o_centered, dtype=jnp.float64)
    y_centered = jnp.asarray(y_centered, dtype=jnp.float64)
    lambda_grid_jax = jnp.asarray(lambda_grid, dtype=jnp.float64)
    kernel = (o_centered @ o_centered.T) / n_samples
    eigenvalues, eigenvectors = jnp.linalg.eigh(kernel)
    y_proj = eigenvectors.T @ y_centered
    denom = eigenvalues[:, None] + lambda_grid_jax[None, :]

    # lambda=0 denotes the ridgeless minimum-norm solution. The empirical
    # kernel usually has numerical zero modes, so handle those by a pseudoinverse
    # cutoff rather than dividing by zero.
    threshold = jnp.maximum(jnp.max(eigenvalues), jnp.array(1.0, dtype=eigenvalues.dtype)) * 1e-10
    positive_mode = eigenvalues > threshold
    ridge_coeffs = y_proj[:, None] / denom
    ridgeless_coeffs = jnp.where(
        positive_mode[:, None],
        y_proj[:, None] / jnp.where(positive_mode, eigenvalues, 1.0)[:, None],
        0.0,
    )
    coeffs = jnp.where(lambda_grid_jax[None, :] == 0.0, ridgeless_coeffs, ridge_coeffs)
    alpha_all = eigenvectors @ coeffs
    delta_all = (o_centered.T @ alpha_all) / n_samples
    pred_train_all = kernel @ alpha_all
    return kernel, delta_all, pred_train_all


def trial_metrics_from_deltas(
    *,
    o_train_centered: np.ndarray,
    y_train_centered: np.ndarray,
    delta_flat: np.ndarray,
    delta_star: np.ndarray,
    s_exact: np.ndarray,
    sigma_gap_sq: float,
) -> dict[str, float]:
    pred_train = o_train_centered @ delta_flat
    r_train_hloc = float(np.mean((pred_train - y_train_centered) ** 2))
    diff = delta_flat - delta_star
    r_pop_ideal = float(diff @ (s_exact @ diff))
    r_pop_hloc = float(r_pop_ideal + sigma_gap_sq)
    return {
        "r_train_hloc": r_train_hloc,
        "r_pop_hloc": r_pop_hloc,
        "r_pop_ideal": r_pop_ideal,
        "gap_ratio": r_pop_hloc / max(r_train_hloc, EPS),
    }


def run_trials_for_checkpoint(
    *,
    model: Any,
    variables: dict[str, Any],
    all_states: np.ndarray,
    exact_diag: dict[str, Any],
    lambda_grid: np.ndarray,
    n_samples: int,
    n_trials: int,
    seed: int,
    jac_chunk_size: int,
    apply_chunk_size: int,
    feature_matrix_centered: jax.Array | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    pi = np.asarray(exact_diag["pi"], dtype=np.float64)
    pi = np.maximum(pi, 0.0)
    pi_sum = float(np.sum(pi))
    if not np.isfinite(pi_sum) or pi_sum <= 0.0:
        raise ValueError("Invalid Born probabilities for exact sampling.")
    pi = pi / pi_sum
    target_hloc = np.asarray(exact_diag["target_hloc"], dtype=np.float64)
    target_ideal = np.asarray(exact_diag["target_ideal"], dtype=np.float64)
    sigma_gap_sq = float(exact_diag["sigma_gap_sq"])
    lambda_grid_jax = jnp.asarray(lambda_grid, dtype=jnp.float64)
    pi_jax = jnp.asarray(pi, dtype=jnp.float64)
    target_hloc_jax = jnp.asarray(target_hloc, dtype=jnp.float64)
    target_ideal_jax = jnp.asarray(target_ideal, dtype=jnp.float64)
    feature_matrix_centered_jax = (
        None if feature_matrix_centered is None else jnp.asarray(feature_matrix_centered, dtype=jnp.float64)
    )
    delta_star = None if feature_matrix_centered is not None else np.asarray(exact_diag["delta_star_flat"], dtype=np.float64)
    s_exact = None if feature_matrix_centered is not None else np.asarray(exact_diag["S_exact"], dtype=np.float64)

    representative_delta_map: dict[int, np.ndarray] = {}
    representative_train_indices = None
    representative_seed = None
    representative_metric_arrays: dict[str, np.ndarray] | None = None

    trials = []
    for trial_idx in range(n_trials):
        trial_seed = seed + trial_idx
        rng = np.random.default_rng(trial_seed)
        train_indices = rng.choice(len(all_states), size=n_samples, replace=True, p=pi)
        train_indices_jax = jnp.asarray(train_indices, dtype=jnp.int32)
        y_train = jnp.take(target_hloc_jax, train_indices_jax, axis=0)
        if feature_matrix_centered_jax is not None:
            o_train = jnp.take(feature_matrix_centered_jax, train_indices_jax, axis=0)
        else:
            train_states = all_states[train_indices]
            o_train = jnp.asarray(
                compute_jacobian_chunked(
                    model,
                    variables,
                    train_states,
                    chunk_size=jac_chunk_size,
                ),
                dtype=jnp.float64,
            )
        o_train_centered = o_train - jnp.mean(o_train, axis=0, keepdims=True)
        y_train_centered = y_train - jnp.mean(y_train)

        _, delta_all, pred_train_all = solve_ridge_lambda_sweep(
            o_train_centered,
            y_train_centered,
            lambda_grid_jax,
            n_samples,
        )
        r_train_hloc = jnp.mean((pred_train_all - y_train_centered[:, None]) ** 2, axis=0)

        if feature_matrix_centered_jax is not None:
            pred_pop_all = feature_matrix_centered_jax @ delta_all
            r_pop_ideal = jnp.sum(
                pi_jax[:, None] * (pred_pop_all - target_ideal_jax[:, None]) ** 2,
                axis=0,
            )
        else:
            delta_all_np = np.asarray(delta_all, dtype=np.float64)
            diff_all = delta_all_np - delta_star[:, None]
            r_pop_ideal = jnp.asarray(
                np.einsum("pl,pq,ql->l", diff_all, s_exact, diff_all, optimize=True),
                dtype=jnp.float64,
            )
        r_pop_hloc = r_pop_ideal + sigma_gap_sq
        gap_ratio = r_pop_hloc / jnp.maximum(r_train_hloc, EPS)

        trial_metric_arrays = {
            "r_train_hloc": np.asarray(r_train_hloc, dtype=np.float64),
            "r_pop_hloc": np.asarray(r_pop_hloc, dtype=np.float64),
            "r_pop_ideal": np.asarray(r_pop_ideal, dtype=np.float64),
            "gap_ratio": np.asarray(gap_ratio, dtype=np.float64),
        }
        if trial_idx == 0:
            delta_all_np = np.asarray(delta_all, dtype=np.float64)
            for lam_idx in range(len(lambda_grid)):
                representative_delta_map[lam_idx] = delta_all_np[:, lam_idx]

        trial_result = {
            "seed": int(trial_seed),
            "train_indices": train_indices.astype(np.int32),
            "metrics": trial_metric_arrays,
        }
        trials.append(trial_result)

        if trial_idx == 0:
            representative_train_indices = train_indices.astype(np.int32)
            representative_seed = int(trial_seed)
            representative_metric_arrays = trial_metric_arrays

    summary = summarize_trials(lambda_grid, trials)
    delta_to_variables = build_delta_tree_converter(variables)
    rep_predictions = representative_predictions(
        model,
        variables,
        all_states,
        pi,
        delta_to_variables,
        representative_delta_map[summary["lambda_small_idx"]],
        representative_delta_map[summary["lambda_best_idx"]],
        representative_delta_map[summary["lambda_last_idx"]],
        apply_chunk_size=apply_chunk_size,
        feature_matrix_centered=feature_matrix_centered_jax,
    )

    explicit_checks = {}
    for label, pred_key, lam_idx in (
        ("small", "pred_small", summary["lambda_small_idx"]),
        ("best", "pred_best", summary["lambda_best_idx"]),
        ("last", "pred_last", summary["lambda_last_idx"]),
    ):
        pred = np.asarray(rep_predictions[pred_key], dtype=np.float64)
        explicit_checks[label] = {
            "lambda": float(lambda_grid[lam_idx]),
            "r_pop_hloc_explicit": float(np.sum(pi * (pred - target_hloc) ** 2)),
            "r_pop_ideal_explicit": float(np.sum(pi * (pred - target_ideal) ** 2)),
            "identity_error": float(
                np.sum(pi * (pred - target_hloc) ** 2)
                - (np.sum(pi * (pred - target_ideal) ** 2) + sigma_gap_sq)
            ),
        }

    representative = {
        "seed": representative_seed,
        "train_indices": representative_train_indices,
        "metrics": representative_metric_arrays,
        "predictions": rep_predictions,
        "explicit_checks": explicit_checks,
    }
    return trials, summary, representative


def training_checkpoint_path(
    checkpoint_dir: Path,
    *,
    step: int,
) -> Path:
    return checkpoint_dir / f"step_{step:04d}.pkl"


def build_training_checkpoint_payload(
    *,
    system_meta: dict[str, Any],
    model_name: str,
    model_desc: str,
    step: int,
    variables: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "system": dict(system_meta),
        "model_name": model_name,
        "model_desc": model_desc,
        "state_origin": "figure1_training_checkpoint",
        "step": int(step),
        "variables": copy_variables(variables),
        "summary": dict(summary),
    }


def save_training_progress(
    path: Path,
    *,
    system_meta: dict[str, Any],
    model_name: str,
    model_desc: str,
    train_config: dict[str, Any],
    completed_step: int,
    checkpoint_summaries: list[dict[str, Any]],
    checkpoint_records: list[dict[str, Any]],
    status: str,
) -> None:
    payload = {
        "format_version": 1,
        "system": dict(system_meta),
        "model_name": model_name,
        "model_desc": model_desc,
        "train_config": dict(train_config),
        "completed_step": int(completed_step),
        "checkpoint_summaries": list(checkpoint_summaries),
        "checkpoint_records": list(checkpoint_records),
        "status": status,
        "updated_at_utc": now(),
    }
    save_pickle(path, payload)


def restore_training_progress(
    progress_path: Path,
) -> dict[str, Any]:
    progress = load_pickle(progress_path)
    checkpoint_runtime = []
    for record in progress.get("checkpoint_records", []):
        payload = load_pickle(Path(record["path"]))
        checkpoint_runtime.append(
            {
                "step": int(payload["step"]),
                "variables": payload["variables"],
                "path": str(record["path"]),
            }
        )
    progress["checkpoint_runtime"] = checkpoint_runtime
    return progress


def train_model_checkpoints(
    *,
    model: Any,
    hilbert: Any,
    hamiltonian: Any,
    all_states: np.ndarray,
    train_samples: int,
    train_steps: int,
    diag_every: int,
    learning_rate: float,
    train_diag_shift: float,
    seed: int,
    jac_chunk_size: int,
    apply_chunk_size: int,
    store_checkpoint_arrays: bool,
    system_meta: dict[str, Any] | None = None,
    model_name: str = "unknown_model",
    model_desc: str = "unknown_model",
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    initial_variables: dict[str, Any] | None = None,
    initial_step: int | None = None,
    stop_energy: float | None = None,
) -> dict[str, Any]:
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=16)
    vstate = nk.vqs.MCState(sampler, model=model, n_samples=train_samples, seed=seed)
    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    driver = nk.driver.VMC_SR(
        hamiltonian,
        optimizer,
        diag_shift=train_diag_shift,
        variational_state=vstate,
    )

    checkpoint_summaries: list[dict[str, Any]] = []
    checkpoint_runtime: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    start_step = 0
    progress_path = None if checkpoint_dir is None else checkpoint_dir / "progress.pkl"
    train_config = {
        "train_samples": int(train_samples),
        "train_steps": int(train_steps),
        "diag_every": int(diag_every),
        "learning_rate": float(learning_rate),
        "train_diag_shift": float(train_diag_shift),
        "seed": int(seed),
        "jac_chunk_size": int(jac_chunk_size),
        "apply_chunk_size": int(apply_chunk_size),
        "store_checkpoint_arrays": bool(store_checkpoint_arrays),
        "initial_step": None if initial_step is None else int(initial_step),
        "stop_energy": None if stop_energy is None else float(stop_energy),
    }

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if resume and progress_path is not None and progress_path.exists():
        progress = restore_training_progress(progress_path)
        checkpoint_summaries = list(progress.get("checkpoint_summaries", []))
        checkpoint_runtime = list(progress.get("checkpoint_runtime", []))
        checkpoint_records = list(progress.get("checkpoint_records", []))
        if checkpoint_runtime:
            start_step = int(checkpoint_runtime[-1]["step"])
            vstate.variables = checkpoint_runtime[-1]["variables"]
            print(f"Resuming {model_name} from step {start_step}")
        if start_step >= train_steps:
            return {
                "checkpoint_summaries": checkpoint_summaries,
                "checkpoint_runtime": checkpoint_runtime,
                "checkpoint_records": checkpoint_records,
                "progress_path": None if progress_path is None else str(progress_path),
                "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
            }
    elif initial_variables is not None:
        start_step = 0 if initial_step is None else int(initial_step)
        if start_step < 0:
            raise ValueError(f"initial_step must be non-negative, got {initial_step}.")
        vstate.variables = copy_variables(initial_variables)

    def persist_checkpoint(step: int, summary: dict[str, Any], variables: dict[str, Any]) -> None:
        if checkpoint_dir is None or progress_path is None:
            return
        checkpoint_path = training_checkpoint_path(checkpoint_dir, step=step)
        checkpoint_payload = build_training_checkpoint_payload(
            system_meta={} if system_meta is None else system_meta,
            model_name=model_name,
            model_desc=model_desc,
            step=step,
            variables=variables,
            summary=summary,
        )
        save_pickle(checkpoint_path, checkpoint_payload)
        checkpoint_records.append({"step": int(step), "path": str(checkpoint_path)})
        save_training_progress(
            progress_path,
            system_meta={} if system_meta is None else system_meta,
            model_name=model_name,
            model_desc=model_desc,
            train_config=train_config,
            completed_step=step,
            checkpoint_summaries=checkpoint_summaries,
            checkpoint_records=checkpoint_records,
            status="running" if step < train_steps else "complete",
        )

    def compute_and_store_checkpoint(step: int) -> dict[str, Any]:
        variables = copy_variables(vstate.variables)
        if not pytree_isfinite(variables):
            raise FloatingPointError(f"Non-finite parameters detected for {model_name} at step {step}.")

        t0 = time.time()
        diag = compute_exact_regression_diagnostics(
            model,
            variables,
            hamiltonian,
            all_states,
            jac_chunk_size=jac_chunk_size,
            apply_chunk_size=apply_chunk_size,
            store_arrays=store_checkpoint_arrays,
            store_matrix=False,
        )
        diag["timing_seconds"] = float(time.time() - t0)
        summary = {
            "step": int(step),
            "energy": float(diag["energy"]),
            "target_var": float(diag["target_var"]),
            "sigma_gap_sq": float(diag["sigma_gap_sq"]),
            "noise_fraction": float(diag["noise_fraction"]),
            "eigenvalues": np.asarray(diag["eigenvalues"], dtype=np.float32),
            "delta_star_flat": np.asarray(diag["delta_star_flat"], dtype=np.float32),
            "timing_seconds": float(diag["timing_seconds"]),
        }
        if store_checkpoint_arrays:
            for key in ("pi", "log_psi_real", "target_hloc", "target_ideal", "expressivity_gap"):
                summary[key] = diag[key]
            summary["explicit_sigma_gap_sq"] = float(diag["explicit_sigma_gap_sq"])
        checkpoint_summaries.append(summary)
        checkpoint_runtime.append({"step": int(step), "variables": variables})
        persist_checkpoint(step, summary, variables)
        return summary

    if not checkpoint_summaries:
        compute_and_store_checkpoint(start_step)

    for step in range(start_step + diag_every, train_steps + 1, diag_every):
        driver.run(n_iter=diag_every)
        summary = compute_and_store_checkpoint(step)
        if stop_energy is not None and float(summary["energy"]) <= float(stop_energy):
            break

    return {
        "checkpoint_summaries": checkpoint_summaries,
        "checkpoint_runtime": checkpoint_runtime,
        "checkpoint_records": checkpoint_records,
        "progress_path": None if progress_path is None else str(progress_path),
        "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
    }


def run_family_experiment(
    *,
    experiment_name: str,
    output_path: Path,
    system_meta: dict[str, Any],
    hilbert: Any,
    hamiltonian: Any,
    graph: Any,
    model_specs: list[str],
    train_samples: int,
    train_steps: int,
    diag_every: int,
    learning_rate: float,
    train_diag_shift: float,
    n_samples: int,
    n_trials: int,
    lambda_grid: np.ndarray,
    seed: int,
    selection_mode: str,
    selection_metric: str,
    match_target_value: float | None,
    match_min_value: float | None,
    band_low: float,
    band_high: float,
    target_noise: float,
    jac_chunk_size: int,
    apply_chunk_size: int,
    store_checkpoint_arrays: bool,
    training_checkpoint_root: Path | None = None,
    resume_training: bool = False,
    square_L: int | None = None,
    j_couplings: Iterable[float] | None = None,
) -> dict[str, Any]:
    if selection_mode in ("paired_match", "paired_match_sigma_priority") and len(model_specs) != 2:
        raise ValueError("Paired checkpoint matching requires exactly two model specs.")

    all_states = np.asarray(hilbert.all_states())
    results = {
        "format_version": 1,
        "experiment": experiment_name,
        "created_at_utc": now(),
        "system": dict(system_meta),
        "ns": int(n_samples),
        "train_config": {
            "train_samples": int(train_samples),
            "train_steps": int(train_steps),
            "diag_every": int(diag_every),
            "learning_rate": float(learning_rate),
            "train_diag_shift": float(train_diag_shift),
            "seed": int(seed),
            "band_low": float(band_low),
            "band_high": float(band_high),
            "target_noise": float(target_noise),
            "jac_chunk_size": int(jac_chunk_size),
            "apply_chunk_size": int(apply_chunk_size),
            "store_checkpoint_arrays": bool(store_checkpoint_arrays),
            "training_checkpoint_root": (
                None if training_checkpoint_root is None else str(training_checkpoint_root)
            ),
            "resume_training": bool(resume_training),
        },
        "selection": {
            "mode": selection_mode,
            "metric": selection_metric,
            "match_target_value": None if match_target_value is None else float(match_target_value),
            "match_min_value": None if match_min_value is None else float(match_min_value),
            "band_low": float(band_low),
            "band_high": float(band_high),
            "target_noise": float(target_noise),
        },
        "lambda_grid": np.asarray(lambda_grid, dtype=np.float64),
        "models": [],
    }

    trained_models: list[dict[str, Any]] = []
    for model_idx, model_name in enumerate(model_specs):
        model, model_desc = make_model(
            model_name,
            hilbert=hilbert,
            graph=graph,
            square_L=square_L,
            j_couplings=j_couplings,
        )
        print(f"\n{'=' * 72}")
        print(f"Model {model_name} ({model_desc})")
        print(f"{'=' * 72}")

        model_checkpoint_dir = (
            None
            if training_checkpoint_root is None
            else training_checkpoint_root / sanitize_name(model_name)
        )

        training = train_model_checkpoints(
            model=model,
            hilbert=hilbert,
            hamiltonian=hamiltonian,
            all_states=all_states,
            train_samples=train_samples,
            train_steps=train_steps,
            diag_every=diag_every,
            learning_rate=learning_rate,
            train_diag_shift=train_diag_shift,
            seed=seed + 10_000 * model_idx,
            jac_chunk_size=jac_chunk_size,
            apply_chunk_size=apply_chunk_size,
            store_checkpoint_arrays=store_checkpoint_arrays,
            system_meta=system_meta,
            model_name=model_name,
            model_desc=model_desc,
            checkpoint_dir=model_checkpoint_dir,
            resume=resume_training,
        )
        trained_models.append(
            {
                "model_name": model_name,
                "model_desc": model_desc,
                "model": model,
                "training": training,
                "trial_seed": seed + 100_000 * (model_idx + 1),
            }
        )

    resolved_match_min_value = resolve_match_min_value(
        selection_mode=selection_mode,
        selection_metric=selection_metric,
        match_min_value=match_min_value,
    )
    results["selection"]["resolved_match_min_value"] = (
        None if resolved_match_min_value is None else float(resolved_match_min_value)
    )

    if selection_mode == "independent_band":
        selected_indices = []
        selected_reasons = []
        for item in trained_models:
            selected_idx, selected_reason = select_checkpoint_band(
                item["training"]["checkpoint_summaries"],
                band_low=band_low,
                band_high=band_high,
                target_noise=target_noise,
            )
            selected_indices.append(selected_idx)
            selected_reasons.append(selected_reason)
        results["selection"]["resolved_target_value"] = float(target_noise)
    elif selection_mode == "independent_target":
        default_target_value = target_noise if selection_metric == "noise_fraction" else None
        resolved_target_value = (
            default_target_value if match_target_value is None else float(match_target_value)
        )
        selected_indices = []
        selected_reasons = []
        for item in trained_models:
            selected_idx, selected_reason = select_checkpoint_target(
                item["training"]["checkpoint_summaries"],
                metric_key=selection_metric,
                target_value=resolved_target_value,
            )
            selected_indices.append(selected_idx)
            selected_reasons.append(selected_reason)
        results["selection"]["resolved_target_value"] = (
            None if resolved_target_value is None else float(resolved_target_value)
        )
    elif selection_mode == "paired_match":
        selected_indices, pair_metadata = select_paired_checkpoints(
            [item["training"]["checkpoint_summaries"] for item in trained_models],
            metric_key=selection_metric,
            match_min_value=resolved_match_min_value,
            match_target_value=match_target_value,
        )
        selected_reasons = [pair_metadata["reason"]] * len(trained_models)
        results["selection"]["resolved_target_value"] = (
            None if match_target_value is None else float(match_target_value)
        )
        results["selection"]["pair_metadata"] = pair_metadata
    elif selection_mode == "paired_match_sigma_priority":
        selected_indices, pair_metadata = select_paired_checkpoints_sigma_priority(
            [item["training"]["checkpoint_summaries"] for item in trained_models],
            match_min_value=resolved_match_min_value,
            match_target_value=match_target_value,
        )
        selected_reasons = [pair_metadata["reason"]] * len(trained_models)
        results["selection"]["resolved_target_value"] = (
            None if match_target_value is None else float(match_target_value)
        )
        results["selection"]["pair_metadata"] = pair_metadata
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    for model_idx, item in enumerate(trained_models):
        training = item["training"]
        selected_idx = int(selected_indices[model_idx])
        selected_reason = selected_reasons[model_idx]
        selected_vars = training["checkpoint_runtime"][selected_idx]["variables"]
        selected_step = training["checkpoint_summaries"][selected_idx]["step"]
        selected = compute_exact_regression_diagnostics(
            item["model"],
            selected_vars,
            hamiltonian,
            all_states,
            jac_chunk_size=jac_chunk_size,
            apply_chunk_size=apply_chunk_size,
            store_arrays=True,
            store_matrix=False,
        )
        selected["step"] = int(selected_step)
        selected["selected_reason"] = selected_reason
        selected["selection_metric"] = selection_metric
        selected["selection_metric_value"] = checkpoint_metric_value(selected, selection_metric)
        feature_matrix_centered = precompute_centered_feature_matrix(
            item["model"],
            selected_vars,
            all_states,
            np.asarray(selected["feature_mean"], dtype=np.float64),
            jac_chunk_size=jac_chunk_size,
        )

        trials, summary, representative = run_trials_for_checkpoint(
            model=item["model"],
            variables=selected_vars,
            all_states=all_states,
            exact_diag=selected,
            lambda_grid=lambda_grid,
            n_samples=n_samples,
            n_trials=n_trials,
            seed=item["trial_seed"],
            jac_chunk_size=jac_chunk_size,
            apply_chunk_size=apply_chunk_size,
            feature_matrix_centered=feature_matrix_centered,
        )

        model_result = {
            "model_name": item["model_name"],
            "model_desc": item["model_desc"],
            "P_real": int(selected["p_real"]),
            "P_over_Ns": float(selected["p_real"] / n_samples),
            "checkpoint_step": int(selected["step"]),
            "checkpoint_energy": float(selected["energy"]),
            "noise_fraction": float(selected["noise_fraction"]),
            "sigma_gap_sq": float(selected["sigma_gap_sq"]),
            "target_var": float(selected["target_var"]),
            "selected_index": selected_idx,
            "selected_reason": selected_reason,
            "selection_metric": selection_metric,
            "selection_metric_value": float(selected["selection_metric_value"]),
            "checkpoint_summaries": training["checkpoint_summaries"],
            "training_checkpoint_dir": training.get("checkpoint_dir"),
            "training_progress_path": training.get("progress_path"),
            "training_checkpoint_records": training.get("checkpoint_records", []),
            "selected_checkpoint": selected,
            "trials": trials,
            "summary": summary,
            "representative": representative,
        }
        results["models"].append(model_result)
        save_pickle(output_path, results)

    return results
