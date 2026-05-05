import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import logsumexp
from typing import Callable


def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


class OutputHead(nn.Module):
    d_model: int
    output_depth: int = 1

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.norms_real = [
            nn.LayerNorm(
                use_scale=True,
                use_bias=True,
                dtype=jnp.float64,
                param_dtype=jnp.float64,
            )
            for _ in range(self.output_depth-1)
        ]

        self.norms_imag = [
            nn.LayerNorm(
                use_scale=True,
                use_bias=True,
                dtype=jnp.float64,
                param_dtype=jnp.float64,
            )
            for _ in range(self.output_depth-1)
        ]

        self.output_layers_real = [
            nn.Dense(
                self.d_model,
                param_dtype=jnp.float64,
                dtype=jnp.float64,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=jax.nn.initializers.zeros,
            )
            for _ in range(self.output_depth)
        ]
        self.output_layers_imag = [
            nn.Dense(
                self.d_model,
                param_dtype=jnp.float64,
                dtype=jnp.float64,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=jax.nn.initializers.zeros,
            )
            for _ in range(self.output_depth)
        ]

    def __call__(self, x):
        # Shape is (samples, patches, d_model)
        x = self.out_layer_norm(x.sum(axis=-2))
        x_real = x
        x_imag = x
        for i in range(self.output_depth - 1):
            x_real = self.norms_real[i](
                nn.activation.gelu(self.output_layers_real[i](x_real))
            )
            x_imag = self.norms_imag[i](
                nn.activation.gelu(self.output_layers_imag[i](x_imag))
            )

        x_real = self.output_layers_real[-1](x_real)
        x_imag = self.output_layers_imag[-1](x_imag)

        z = x_real + 1j * x_imag
        return jnp.sum(log_cosh(z), axis=-1)


class RealPositiveHead(nn.Module):
    r"""
    Single Layer OutputHead with real FFNN
    Returns log\Psi = \sum_i log(cosh(x_i)), such that \Psi is real and positive
    """

    d_model: int

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.output_layer0 = nn.Dense(
            self.d_model,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):
        # Shape is (samples, patches, d_model)
        x = self.out_layer_norm(x.sum(axis=-2))
        x = self.output_layer0(x)
        return jnp.sum(log_cosh(x), axis=-1)

class FTHead(nn.Module):
    """
    Output head performing quantum number projection to a specific momentum sector
    """

    d_model: int  # number of features
    q: tuple  # (qx,qy,..)
    compute_positions: (
        Callable  # function to compute the (npatches,ndim) positions of the patches
    )

    def setup(self):

        self.norm_real = nn.LayerNorm(
            use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64
        )
        self.norm_imag = nn.LayerNorm(
            use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64
        )

        self.dense_real = nn.Dense(
            self.d_model,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )
        self.dense_imag = nn.Dense(
            self.d_model,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):
        # Input shape is (...,patches, d_model)
        # Complex MLP over d_model
        # print("Head input shape", x.shape)
        x_real = self.norm_real(self.dense_real(x)) #not sure whether we need these layer norms 
        x_imag = self.norm_imag(self.dense_imag(x))
        z = x_real + 1j * x_imag
        z = jnp.sum(log_cosh(z), axis=-1)
        # print("Before FT shape", z.shape)
        positions = self.compute_positions()  # (Np, ndim)
        q = jnp.pi * jnp.array(self.q)
        b = jnp.exp(-1j * positions @ q) / z.shape[-1]  # (Np,) vector
        z = logsumexp(z, axis=-1, b=b)
        return z


class FTHeadReal(nn.Module):
    """
    Output head performing quantum number projection to a specific momentum sector with real FFNN
    """

    b: int  # patch size
    d_model: int  # number of features
    q: tuple  # (qx,qy,..)
    compute_positions: (
        Callable  # function to compute the (npatches,ndim) positions of the patches
    )

    def setup(self):

        self.norm_real = nn.LayerNorm(
            use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64
        )

        self.dense_real = nn.Dense(
            self.d_model,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):
        # Input shape is (...,patches, d_model)
        # Complex MLP over d_model
        # print("Head input shape", x.shape)
        x = self.norm_real(self.dense_real(x)) #not sure whether we need this layer norm
        x = jnp.sum(log_cosh(x), axis=-1)
        # print("Before FT shape", z.shape)
        positions = self.compute_positions(self.b)  # (Np, ndim)
        q = jnp.pi * jnp.array(self.q)
        b = jnp.exp(-1j * positions @ q)  # (Np,) vector
        z = logsumexp(z, axis=-1, b=b)
        return z
