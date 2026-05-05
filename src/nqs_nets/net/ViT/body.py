# Embedding and Encoder classes for ViT

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from jax import random
from jax._src import dtypes
from einops import rearrange
from typing import Callable, Optional


def custom_uniform(scale=1e-2, dtype=jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        return (2.0 * random.uniform(key, shape, dtype) - 1.0) * scale

    return init


def roll(J, shift, axis=-1):
    return jnp.roll(J, shift, axis=axis)


@partial(jax.vmap, in_axes=(None, 0, None, None, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0, None, None), out_axes=1)
def roll2d(mat, i, j, Lx, Ly):
    """
    Construct the (h, Np, Np) matrix where the (h,i,j) elements are equivalent if the relative positions
    of the patches i,j are the same. Assumes that the paches are indexed as s = y + xLx.
    """
    mat = mat.reshape(mat.shape[0], Lx, Ly)
    mat = jnp.roll(jnp.roll(mat, i, axis=-2), j, axis=-1)
    return mat.reshape(mat.shape[0], -1)


@partial(jax.vmap, in_axes=(None, 0, None, None, None, None, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0, None, None, None, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, None, 0, None, None, None), out_axes=1)
def roll3d(mat, i, j, k, Lx, Ly, Lz):
    """ """
    mat = mat.reshape(mat.shape[0], Lx, Ly, Lz)
    mat = jnp.roll(jnp.roll(jnp.roll(mat, i, axis=-3), j, axis=-2), k, axis=-1)
    return mat.reshape(mat.shape[0], -1)


@partial(jax.vmap, in_axes=(None, 0, None))
def _compute_attn(J, x, h):
    x = rearrange(x, " L_eff (h d_eff) ->  L_eff h d_eff", h=h)

    x = rearrange(x, " L_eff h d_eff ->  h L_eff d_eff")
    x = jnp.matmul(J, x)
    x = rearrange(x, " h L_eff d_eff  ->  L_eff h d_eff")

    x = rearrange(x, " L_eff h d_eff ->  L_eff (h d_eff)")

    return x


class FMHA(nn.Module):
    d_model: int
    h: int
    plattice_shape: tuple  # shape of patched lattice (Lx,Ly,Lz)
    kernel_shape: Optional[tuple] = (
        None  # (kx,ky,kz), masks the attention if kernel_shape smaller than plattice_shape and transl_invariant == True
    )
    transl_invariant: bool = True

    def setup(self):
        kernel_shape = (
            self.kernel_shape or self.plattice_shape
        )  # use a local variable for kernel shape to avoid flax.errors.SetAttributeInModuleSetupError
        assert np.all(
            [k <= p for k, p in zip(kernel_shape, self.plattice_shape)]
        ), "kernel_shape must be smaller than or equal to plattice_shape"
        self.Np = np.prod(self.plattice_shape)
        self.n_dim = len(self.plattice_shape)
        self.v = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )
        # Make the (h, Np, Np) matrix, with parameters the same according to relative positions
        if self.transl_invariant:
            self.J = self.param(
                "J",
                custom_uniform(scale=(3.0 / self.Np) ** 0.5),
                (self.h, np.prod(kernel_shape)),  # h \times Nk parameters
                jnp.float64,
            )
            if self.n_dim == 1:
                self.J = jnp.pad(
                    self.J,
                    pad_width=(
                        (0, 0),
                        (0, abs(kernel_shape[0] - self.plattice_shape[0])),
                    ),
                    mode="constant",
                    constant_values=0,
                )  # fill in remaining with zeros
                # to make an (h, Np) matrix
                self.J = jax.vmap(roll, (None, 0), out_axes=1)(
                    self.J, jnp.arange(self.Np) - (kernel_shape[0] // 2)
                )  # -> (h, Np, Np), -kernel_shape[0]//2 centers the kernel
            elif self.n_dim == 2:
                # Pad J matrix with zeros for masked attention
                self.J = self.J.reshape(
                    self.J.shape[0], kernel_shape[0], kernel_shape[1]
                )  # (h, kx, ky) allows correct padding of zeros
                self.J = jnp.pad(
                    self.J,
                    pad_width=(
                        (0, 0),
                        (0, abs(kernel_shape[0] - self.plattice_shape[0])),
                        (0, abs(kernel_shape[1] - self.plattice_shape[1])),
                    ),
                    mode="constant",
                    constant_values=0,
                )  # to (h, Lx, Ly)
                self.J = self.J.reshape(self.J.shape[0], -1)  # flatten back to (h, Np)
                self.J = roll2d(
                    self.J,
                    jnp.arange(self.plattice_shape[0]) - (kernel_shape[0] // 2),
                    jnp.arange(self.plattice_shape[1]) - (kernel_shape[1] // 2),
                    self.plattice_shape[0],
                    self.plattice_shape[1],
                )  # -kernel_shape[i]//2 centers the kernel
                self.J = self.J.reshape(
                    self.J.shape[0], -1, self.J.shape[-1]
                )  # -> (h, Np, Np)
            elif self.n_dim == 3:
                # Pad J matrix with zeros for masked attention
                self.J = self.J.reshape(
                    self.J.shape[0], kernel_shape[0], kernel_shape[1], kernel_shape[2]
                )  # (h, kx, ky, kz)
                self.J = jnp.pad(
                    self.J,
                    pad_width=(
                        (0, 0),
                        (0, abs(kernel_shape[0] - self.plattice_shape[0])),
                        (0, abs(kernel_shape[1] - self.plattice_shape[1])),
                        (0, abs(kernel_shape[2] - self.plattice_shape[2])),
                    ),
                    mode="constant",
                    constant_values=0,
                )
                self.J = self.J.reshape(self.J.shape[0], -1)  # flatten back to (h, Np)
                self.J = roll3d(
                    self.J,
                    jnp.arange(self.plattice_shape[0]) - (kernel_shape[0] // 2),
                    jnp.arange(self.plattice_shape[1]) - (kernel_shape[1] // 1),
                    jnp.arange(self.plattice_shape[2]) - (kernel_shape[2] // 2),
                    self.plattice_shape[0],
                    self.plattice_shape[1],
                    self.plattice_shape[2],
                )
                self.J = self.J.reshape(
                    self.J.shape[0], -1, self.J.shape[-1]
                )  # -> (h, Np, Np)
            else:
                raise ValueError("Invalid n_dim: must be 1,2 or 3")
        else:
            self.J = self.param(
                "J",
                custom_uniform(scale=(3.0 / self.Np) ** 0.5),
                (self.h, self.Np, self.Np),
                jnp.float64,
            )

        self.W = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )

    def __call__(self, x):
        x = self.v(x)

        x = _compute_attn(self.J, x, self.h)

        x = self.W(x)

        return x


class Embed(nn.Module):
    d_model: int
    extract_patches: Callable

    def setup(self):
        self.embed = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )

    def __call__(self, x):
        x = self.extract_patches(x)
        x = self.embed(x)

        return x


class EncoderBlock(nn.Module):
    d_model: int
    h: int
    plattice_shape: tuple
    kernel_shape: tuple = None
    expansion_factor: int = 4
    transl_invariant: bool = True

    def setup(self):
        self.attn = FMHA(
            d_model=self.d_model,
            h=self.h,
            plattice_shape=self.plattice_shape,
            kernel_shape=self.kernel_shape,
            transl_invariant=self.transl_invariant,
        )

        self.layer_norm_1 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        self.layer_norm_2 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.ff = nn.Sequential(
            [
                nn.Dense(
                    self.expansion_factor * self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=jnp.float64,
                    dtype=jnp.float64,
                ),
                nn.gelu,
                nn.Dense(
                    self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=jnp.float64,
                    dtype=jnp.float64,
                ),
            ]
        )

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))

        x = x + self.ff(self.layer_norm_2(x))
        return x


class Encoder(nn.Module):
    num_layers: int
    d_model: int
    h: int
    plattice_shape: tuple
    kernel_shape: tuple = None  # kernel_shape for masked attention, if None, no masking
    expansion_factor: int = 4
    transl_invariant: bool = True

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                h=self.h,
                plattice_shape=self.plattice_shape,
                kernel_shape=self.kernel_shape,
                expansion_factor=self.expansion_factor,
                transl_invariant=self.transl_invariant,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)

        return x
