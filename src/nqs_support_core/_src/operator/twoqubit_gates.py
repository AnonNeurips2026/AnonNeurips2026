from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.operator import DiscreteJaxOperator, spin
from netket.hilbert import AbstractHilbert


@register_pytree_node_class
class Rxx(DiscreteJaxOperator):
    def __init__(self, hi: AbstractHilbert, idx: tuple, angle: float):
        super().__init__(hi)
        self._local_states = jnp.asarray(hi.local_states)
        self._idx = tuple(int(i) for i in idx)
        self._angle = angle

    @property
    def angle(self):
        """
        The angle of this rotation.
        """
        return self._angle

    @property
    def idx(self):
        """
        The tuple with qubit pair on which the rotation acts
        """
        return self._idx

    @property
    def dtype(self):
        return complex

    @property
    def H(self):
        return Rxx(self.hilbert, self.idx, -self.angle)

    def __eq__(self, o):
        if isinstance(o, Rxx):
            return o.idx == self.idx and o.angle == self.angle
        return False

    def tree_flatten(self):
        children = (self.angle,)
        aux_data = (
            self.hilbert,
            self.idx,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (angle,) = children
        return cls(*aux_data, angle)

    @property
    def max_conn_size(self) -> int:
        return 4

    @jax.jit
    def get_conn_padded(self, x):
        xr = x.reshape(-1, x.shape[-1])
        xp, mels = get_conns_and_mels_Rxx(xr, self.idx, self.angle, self._local_states)
        xp = xp.reshape(x.shape[:-1] + xp.shape[-2:])
        mels = mels.reshape(x.shape[:-1] + mels.shape[-1:])
        return xp, mels

    def get_conn_flattened(self, x, sections):
        xp, mels = self.get_conn_padded(x)
        sections[:] = np.arange(2, mels.size + 2, 2)

        xp = xp.reshape(-1, self.hilbert.size)
        mels = mels.reshape(
            -1,
        )
        return xp, mels

    def to_local_operator(self):  # RXX = cos(θ/2) I  – i sin(θ/2) X_i X_j
        c = np.cos(self.angle / 2)
        s = np.sin(self.angle / 2)
        idx1, idx2 = self.idx
        return c - 1j * s * spin.sigmax(self.hilbert, idx1) * spin.sigmax(
            self.hilbert, idx2
        )


@partial(jax.vmap, in_axes=(0, None, None, None), out_axes=(0, 0))
def get_conns_and_mels_Rxx(sigma, idx, angle, local_states):
    assert sigma.ndim == 1

    state_0 = jnp.asarray(local_states[0], dtype=sigma.dtype)
    state_1 = jnp.asarray(local_states[1], dtype=sigma.dtype)

    conns = jnp.tile(sigma, (2, 1))

    idx1, idx2 = idx
    curr1, curr2 = sigma[idx1], sigma[idx2]

    flip1 = jnp.where(curr1 == state_0, state_1, state_0)
    flip2 = jnp.where(curr2 == state_0, state_1, state_0)
    conns = conns.at[1, idx1].set(flip1)
    conns = conns.at[1, idx2].set(flip2)

    mels = jnp.zeros(2, dtype=complex)
    mels = mels.at[0].set(jnp.cos(angle / 2))
    mels = mels.at[1].set(-1j * jnp.sin(angle / 2))

    return conns, mels
