from flax import linen as nn
from netket.utils.types import Array
from netket.graph import AbstractGraph
from typing import Callable, Type, Optional, Sequence
import jax.numpy as jnp


class SignHelper:
    def __init__(self, graph: AbstractGraph, x_sublattices: Optional[Sequence[int]] = None, x_sites: Optional[Sequence[int]] = None):
        """
            Computes the number of up spins on the x sublattice.
            The sites on the sublattice are deduced from the argument x_sublattices or x_sites.
            x_sublattices: A sequence of indices of the sublattices within the unit cell belonging to the x sublattice
            x_sites: A sequence of site indices that belong to the x sublattice
        """
        self.graph = graph
        if x_sublattices is None and x_sites is None:
            raise ValueError("Either x_sublattices or x_sites must be provided")
        if x_sublattices is not None and x_sites is not None:
            raise ValueError("Only one of x_sublattices or x_sites can be provided")
        if x_sublattices is not None: 
            self.x_sublattices = x_sublattices
            self.all_x_sites = self.extract_basis_ids()
            self.Nx = len(self.all_x_sites)
        else:
            self.all_x_sites = jnp.array(x_sites)
            self.Nx = len(self.all_x_sites)

    def extract_basis_ids(self):
        basis_ids = []
        for site in self.graph.sites:
            if site.basis_coord[-1] in self.x_sublattices:
                basis_ids.append(site.id)

        return jnp.array(basis_ids)

    def compute_nx(self, x):
        mx = jnp.sum(x[..., self.all_x_sites], axis=-1)
        nx = (mx + self.Nx) / 2
        return nx


class SignRule(nn.Module):
    """
    Computes a Marshall-type sign rule (-1)^n_x where n_x is the number of up spins
    on the x sublattice.
    """

    compute_nx: Callable
    dtype: Type = (
        jnp.complex128
    )  # float64 + float64 to make complex number makes complex128, so use this unless float32 in real part of net

    @nn.compact
    def __call__(self, x: Array):
        return self.dtype((-1) ** self.compute_nx(x))


class SignNet(nn.Module):
    logpsi: nn.Module
    sign_type: Type
    compute_nx: Callable
    dtype: Type = (
        jnp.complex128
    )  # float64 + float64 to make complex number makes complex128, so use this unless float32 in real part of net

    def setup(self):
        self.sign = self.sign_type(compute_nx=self.compute_nx, dtype=self.dtype)

    def __call__(self, x: Array):
        return self.logpsi(x) + jnp.log(self.sign(x))
    
class DoubleSignNet(nn.Module):
    logpsi: nn.Module
    sign_type: Type
    compute_nx1: Callable
    compute_nx2: Callable
    dtype: Type = (
        jnp.complex128
    )  # float64 + float64 to make complex number makes complex128, so use this unless float32 in real part of net

    def setup(self):
        self.sign1 = self.sign_type(compute_nx=self.compute_nx1, dtype=self.dtype)
        self.sign2 = self.sign_type(compute_nx=self.compute_nx2, dtype=self.dtype)

    def __call__(self, x: Array):
        return self.logpsi(x) + jnp.log(self.sign1(x)) + jnp.log(self.sign2(x))
