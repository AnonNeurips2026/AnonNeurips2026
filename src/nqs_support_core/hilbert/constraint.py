import netket as nk
import jax.numpy as jnp
from netket.utils import dispatch
import jax


class SpinHalfConstraint(nk.hilbert.constraint.DiscreteHilbertConstraint):
    """
    Constraint on fermionic hilbert space to have one fermion per site.
    Assumes Nup = N//2 and Ndown = N//2.
    Used in nk.hilbert.SpinOrbitalFermions(..., constraint = SpinHalfConstraint())
    """

    def __call__(self, x):
        return jnp.all(
            (x[..., : x.shape[-1] // 2] + x[..., x.shape[-1] // 2 :]) == 1, axis=-1
        )


@dispatch.dispatch
def random_state(
    hilb: nk.hilbert.SpinOrbitalFermions,
    constraint: nk.hilbert.constraint.ExtraConstraint[
        nk.hilbert.constraint.SumOnPartitionConstraint, SpinHalfConstraint
    ],
    key,
    batches: int,
    *,
    dtype=None,
):

    @jax.vmap
    def gen_sample(seed):
        up_positions = jax.random.choice(
            jax.random.PRNGKey(seed),
            hilb.n_orbitals,
            shape=(hilb.n_orbitals // 2,),
            replace=False,
        )
        up_occupation = jnp.zeros((hilb.n_orbitals,), dtype=jnp.int8)
        up_occupation = up_occupation.at[up_positions].set(1)
        down_occupation = (
            ~up_occupation + 2
        )  # where there is no up spin there is a down spin
        return jnp.concatenate((up_occupation, down_occupation), axis=-1)

    seeds = jax.random.randint(key, shape=(batches,), minval=0, maxval=int(1e6))
    samples = gen_sample(seeds)
    return samples
