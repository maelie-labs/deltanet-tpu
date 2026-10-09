"""Causality tests for Pallas TPU GDN-2 kernel.

Verifies that modifying inputs at t > t_cutoff produces strictly zero change
in outputs at t <= t_cutoff, and that gradients of a loss defined on t <= t_cutoff
are strictly zero for all input tokens at t > t_cutoff.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from deltanet_tpu.gdn2 import chunk_gdn2


def make_test_inputs(key, B, L, Hk, Hv, K, V):
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    b = jax.nn.sigmoid(jax.random.normal(k3, (B, L, Hk, K), jnp.float32) * 0.39)
    w = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hv, V), jnp.float32) * 0.39)
    g = -jnp.exp(jax.random.normal(k5, (B, L, Hk, K), jnp.float32) * 0.39 - 1.0)
    return q, k, v, g, b, w


@pytest.mark.parametrize("t_cutoff", [100, 128])
def test_causality_forward_and_backward(t_cutoff):
    """Test strictly causal forward propagation and backward gradient isolation."""
    chunk_size = 128
    B, L, Hk, Hv, K, V = 1, 256, 2, 4, 128, 128

    key = jax.random.PRNGKey(101)
    q1, k1, v1, g1, b1, w1 = make_test_inputs(key, B, L, Hk, Hv, K, V)

    # Create perturbed inputs where only tokens at t >= t_cutoff are modified
    pert_key = jax.random.PRNGKey(202)
    qp, kp, vp, gp, bp, wp = make_test_inputs(pert_key, B, L - t_cutoff, Hk, Hv, K, V)

    q2 = q1.at[:, t_cutoff:].set(qp)
    k2 = k1.at[:, t_cutoff:].set(kp)
    v2 = v1.at[:, t_cutoff:].set(vp)
    g2 = g1.at[:, t_cutoff:].set(gp)
    b2 = b1.at[:, t_cutoff:].set(bp)
    w2 = w1.at[:, t_cutoff:].set(wp)

    # 1. Forward causality: outputs at t < t_cutoff must be bit-identical
    o1 = chunk_gdn2(q1, k1, v1, g1, b1, w1, chunk_size=chunk_size, precision="fp32")
    o2 = chunk_gdn2(q2, k2, v2, g2, b2, w2, chunk_size=chunk_size, precision="fp32")

    diff_fwd = np.max(np.abs(np.asarray(o1[:, :t_cutoff] - o2[:, :t_cutoff])))
    assert diff_fwd == 0.0, f"Forward causality violation: max diff = {diff_fwd}"

    # 2. Backward causality: loss depends only on outputs at t < t_cutoff
    def loss_fn(q, k, v, g, b, w):
        o = chunk_gdn2(q, k, v, g, b, w, chunk_size=chunk_size, precision="fp32")
        return jnp.sum(o[:, :t_cutoff] ** 2)

    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))
    grads1 = grad_fn(q1, k1, v1, g1, b1, w1)
    grads2 = grad_fn(q2, k2, v2, g2, b2, w2)

    grad_names = ["dq", "dk", "dv", "dg", "db", "dw"]
    for name, g_1, g_2 in zip(grad_names, grads1, grads2):
        g_1_np = np.asarray(g_1)
        g_2_np = np.asarray(g_2)

        # Gradients at t >= t_cutoff must be strictly 0
        max_future_grad = np.max(np.abs(g_1_np[:, t_cutoff:]))
        assert max_future_grad == 0.0, f"{name} non-zero future gradient: {max_future_grad}"

        # Gradients at t < t_cutoff must match identically
        diff_grad = np.max(np.abs(g_1_np[:, :t_cutoff] - g_2_np[:, :t_cutoff]))
        assert diff_grad == 0.0, f"{name} backward causality mismatch: {diff_grad}"
