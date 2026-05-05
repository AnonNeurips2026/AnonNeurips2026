from typing import Callable, Optional, Union
from functools import partial

from einops import rearrange

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from netket import jax as nkjax
from netket.jax._jacobian.default_mode import JacobianMode
from netket.utils.types import Array
from netket.utils.version_check import module_version

from nqs_support_core._src import distributed as distributed
from nqs_support_core._src.external import neural_tangents as nt

from advanced_drivers._src.driver.ngd.srt_onthefly import (
    _compute_quadratic_model_srt_onthefly,
)


def _replica_centering_matrix(
    n_replicas: int,
    samples_per_replica: int,
    *,
    dtype,
):
    block = jnp.eye(samples_per_replica, dtype=dtype)
    block = block - jnp.full(
        (samples_per_replica, samples_per_replica),
        1.0 / samples_per_replica,
        dtype=dtype,
    )
    return jnp.kron(jnp.eye(n_replicas, dtype=dtype), block)


@partial(
    jax.jit,
    static_argnames=(
        "log_psi",
        "solver_fn",
        "n_replicas",
        "chunk_size",
        "mode",
        "collect_quadratic_model",
        "collect_gradient_statistics",
        "collect_residual_info",
    ),
)
def srt_onthefly(
    log_psi,
    local_grad,
    parameters,
    model_state,
    samples,
    *,
    n_replicas: int,
    diag_shift: Union[float, Array],
    solver_fn: Callable[[Array, Array], Array],
    mode: JacobianMode,
    proj_reg: Optional[Union[float, Array]] = None,
    momentum: Optional[Union[float, Array]] = None,
    old_updates: Optional[Array] = None,
    chunk_size: Optional[int] = None,
    collect_quadratic_model: bool = False,
    collect_gradient_statistics: bool = False,
    collect_residual_info: bool = False,
):
    del collect_gradient_statistics

    N_mc = local_grad.size
    if N_mc % n_replicas != 0:
        raise ValueError(
            f"N_mc={N_mc} must be divisible by n_replicas={n_replicas}"
        )
    samples_per_replica = N_mc // n_replicas

    parameters_real, rss = nkjax.tree_to_real(parameters)

    def _apply_fn(parameters_real, samples):
        variables = {"params": rss(parameters_real), **model_state}
        log_amp = log_psi(variables, samples)

        if mode == "complex":
            re, im = log_amp.real, log_amp.imag
            return jnp.concatenate((re[:, None], im[:, None]), axis=-1)
        return log_amp.real

    def jvp_f_chunk(parameters, vector, samples):
        f = lambda params: _apply_fn(params, samples)
        _, acc = jax.jvp(f, (parameters,), (vector,))
        return acc

    local_grad = local_grad.reshape(n_replicas, samples_per_replica)
    de = local_grad - jnp.mean(local_grad, axis=1, keepdims=True)
    dv = 2.0 * de.reshape(-1) / jnp.sqrt(N_mc)

    token = None
    if momentum is not None:
        if old_updates is None:
            old_updates = tree_map(jnp.zeros_like, parameters_real)
        else:
            acc = nkjax.apply_chunked(
                jvp_f_chunk, in_axes=(None, None, 0), chunk_size=chunk_size
            )(parameters_real, old_updates, samples)
            acc = acc.reshape(n_replicas, samples_per_replica, *acc.shape[1:])
            avg = jnp.mean(acc, axis=1, keepdims=True)
            acc = (acc - avg).reshape(N_mc, *acc.shape[2:]) / jnp.sqrt(N_mc)
            dv -= momentum * acc

    if mode == "complex":
        dv = jnp.stack([jnp.real(dv), jnp.imag(dv)], axis=-1)
        dv = jax.lax.collapse(dv, 0, 2)
    else:
        dv = jnp.real(dv)
    dv, token = distributed.allgather(dv, token=token)

    all_samples, token = distributed.allgather(samples, token=token)

    _jacobian_contraction = nt.empirical_ntk_fn(
        f=_apply_fn,
        trace_axes=(),
        vmap_axes=0,
        implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION,
    )

    def jacobian_contraction(samples, all_samples, parameters_real):
        if chunk_size is None:
            return _jacobian_contraction(samples, all_samples, parameters_real).real

        _all_samples, _ = nkjax.chunk(all_samples, chunk_size=chunk_size)
        ntk_local = jax.lax.map(
            lambda batch_lattice: _jacobian_contraction(
                samples, batch_lattice, parameters_real
            ).real,
            _all_samples,
        )
        if mode == "complex":
            return rearrange(ntk_local, "nbatches i j z w -> i (nbatches j) z w")
        return rearrange(ntk_local, "nbatches i j -> i (nbatches j)")

    if distributed.mode() == "sharding":
        mesh = jax.make_mesh(
            (distributed.device_count(),), ("S",), devices=jax.devices()
        )
        in_specs = (P("S", None), P(), P())
        out_specs = P("S", None, None, None) if mode == "complex" else P("S", None)
        check_rep = module_version("jax") < (0, 4, 38)
        jacobian_contraction = shard_map(
            jacobian_contraction,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=check_rep,
        )

    with nkjax.sharding._increase_SHARD_MAP_STACK_LEVEL():
        ntk_local = jacobian_contraction(samples, all_samples, parameters_real).real

    ntk, token = distributed.allgather(ntk_local, token=token)
    if mode == "complex":
        ntk = rearrange(ntk, "i j z w -> (i z) (j w)")

    delta = _replica_centering_matrix(
        n_replicas,
        samples_per_replica,
        dtype=ntk.dtype,
    )
    if mode == "complex":
        delta_conc = jnp.zeros((2 * N_mc, 2 * N_mc), dtype=ntk.dtype)
        delta_conc = delta_conc.at[0::2, 0::2].set(delta)
        delta_conc = delta_conc.at[1::2, 1::2].set(delta)
    else:
        delta_conc = delta

    ntk = (delta_conc @ (ntk @ delta_conc)) / N_mc
    ntk_shifted = ntk + diag_shift * jnp.eye(ntk.shape[0], dtype=ntk.dtype)
    if proj_reg is not None:
        ntk_shifted = ntk_shifted + proj_reg / N_mc

    aus_vector = solver_fn(ntk_shifted, dv)
    if isinstance(aus_vector, tuple):
        aus_vector, info = aus_vector
    else:
        info = {}
    if info is None:
        info = {}

    if collect_quadratic_model:
        info.update(_compute_quadratic_model_srt_onthefly(ntk, aus_vector, dv))

    aus_vector = aus_vector / jnp.sqrt(N_mc)
    aus_vector = delta_conc @ aus_vector

    if collect_residual_info:
        info["aux_vector"] = aus_vector
        info["ntk_matrix"] = ntk
        info["dv"] = dv
        info["N_mc"] = N_mc

    if mode == "complex":
        aus_vector = aus_vector.reshape(-1, 2)
    aus_vector = distributed.shard_replicated(aus_vector, axis=0)

    vjp_fun = nkjax.vjp_chunked(
        _apply_fn,
        parameters_real,
        samples,
        chunk_size=chunk_size,
        chunk_argnums=1,
        nondiff_argnums=1,
    )
    updates = vjp_fun(aus_vector)[0]

    if momentum is not None:
        updates = tree_map(lambda x, y: x + momentum * y, updates, old_updates)
        old_updates = updates

    return rss(updates), old_updates, info
