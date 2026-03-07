import jax
import jax.numpy as jnp
import numpy.testing as npt
from covvfit._variant_beta_binomial.growth import LinearGrowthParams, compute_log_pi


def test_compute_log_pi_shape() -> None:
    theta = LinearGrowthParams(
        intercepts=jnp.array([[0.0, 1.0, -1.0], [0.5, -0.2, 0.3]]),
        slopes=jnp.array([0.1, -0.2, 0.05]),
    )
    city = jnp.array([0, 1, 1, 0], dtype=jnp.int32)
    time = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=float)

    out = compute_log_pi(theta=theta, city=city, time=time)

    assert out.shape == (4, 3)


def test_compute_log_pi_reproducible() -> None:
    theta = LinearGrowthParams(
        intercepts=jnp.array([[0.1, 0.2], [0.3, 0.4]]),
        slopes=jnp.array([0.05, -0.03]),
    )
    city = jnp.array([0, 1], dtype=jnp.int32)
    time = jnp.array([2.0, 5.0], dtype=float)

    out1 = compute_log_pi(theta=theta, city=city, time=time)
    out2 = compute_log_pi(theta=theta, city=city, time=time)
    npt.assert_allclose(out1, out2)


def test_softmax_rows_sum_to_one() -> None:
    theta = LinearGrowthParams(
        intercepts=jnp.array([[0.0, 1.0, -2.0], [1.0, -1.0, 0.5]]),
        slopes=jnp.array([0.2, 0.1, -0.1]),
    )
    city = jnp.array([0, 0, 1], dtype=jnp.int32)
    time = jnp.array([0.0, 1.0, 2.0], dtype=float)

    log_pi = compute_log_pi(theta=theta, city=city, time=time)
    probs = jax.nn.softmax(log_pi, axis=-1)

    npt.assert_allclose(probs.sum(axis=-1), jnp.ones((3,)), atol=1e-6)


def test_city_changes_output() -> None:
    theta = LinearGrowthParams(
        intercepts=jnp.array([[0.0, 0.0], [2.0, -2.0]]),
        slopes=jnp.array([0.0, 0.0]),
    )
    time = jnp.array([1.0, 1.0], dtype=float)
    city = jnp.array([0, 1], dtype=jnp.int32)

    out = compute_log_pi(theta=theta, city=city, time=time)

    assert not jnp.allclose(out[0], out[1])
