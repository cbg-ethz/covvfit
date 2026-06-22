"""Stable variant-to-mutation mapping utilities."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float


def make_log_A_mask(
    A: Float[Array, "variants loci"], dtype: jnp.dtype | None = None
) -> Float[Array, "variants loci"]:
    """Build a log-space mask for variant definitions matrix A."""
    A = jnp.asarray(A)
    if dtype is None:
        dtype = A.dtype

    if A.ndim != 2:
        raise ValueError("A must have shape (variants, loci).")

    neg = jnp.finfo(dtype).min
    return jnp.where(A == 1, jnp.array(0.0, dtype=dtype), jnp.array(neg, dtype=dtype))


def compute_log_mu(
    log_pi: Float[Array, "samples variants"],
    A: Float[Array, "variants loci"],
) -> Float[Array, "samples loci"]:
    """Compute log expected mutation frequencies in a stable way."""
    log_pi = jnp.asarray(log_pi)
    if log_pi.ndim != 2:
        raise ValueError("log_pi must have shape (samples, variants).")

    A = jnp.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must have shape (variants, loci).")
    if log_pi.shape[1] != A.shape[0]:
        raise ValueError("log_pi and A shapes are incompatible on variants axis.")

    log_pi = jax.nn.log_softmax(log_pi, axis=1)
    log_A_mask = make_log_A_mask(A, dtype=log_pi.dtype)
    tmp = log_pi[:, :, None] + log_A_mask[None, :, :]
    return logsumexp(tmp, axis=1)


def compute_mu(
    log_pi: Float[Array, "samples variants"],
    A: Float[Array, "variants loci"],
) -> Float[Array, "samples loci"]:
    """Compute expected mutation frequencies."""
    return jnp.exp(compute_log_mu(log_pi=log_pi, A=A))
