"""Strong decay tests for Pallas TPU GDN-2 kernel.

Verifies that extreme negative decay (g <= -150) underflows cleanly to zero
without producing NaNs or Infs in either forward outputs or backward gradients.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from deltanet_tpu.gdn2 import chunk_gdn2
from deltanet_tpu.reference import token_serial_gdn2


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def make_test_inputs(key, B, L, Hk, Hv, K, V):
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    b = jax.nn.sigmoid(jax.random.normal(k3, (B, L, Hk, K), jnp.float32) * 0.39)
    w = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hv, V), jnp.float32) * 0.39)
    # Strong negative decay <= -150
    g = -150.0 - jnp.abs(jax.random.normal(k5, (B, L, Hk, K), jnp.float32) * 50.0)
    return q, k, v, g, b, w


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_strong_decay_finiteness_and_parity(precision):
    """Verify forward and backward numerical stability under extreme decay (g <= -150)."""
    B, L, Hk, Hv, K, V = 1, 256, 2, 4, 128, 128
    chunk_size = 128
    key = jax.random.PRNGKey(501)
    q, k, v, g, b, w = make_test_inputs(key, B, L, Hk, Hv, K, V)
    do = jax.random.normal(jax.random.PRNGKey(502), (B, L, Hv, V), jnp.float32)

    def run_ref(q, k, v, g, b, w):
        return token_serial_gdn2(q, k, v, g, b, w)

    def run_kernel(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, chunk_size=chunk_size, precision=precision)

    o_ref, vjp_ref = jax.vjp(run_ref, q, k, v, g, b, w)
    o_kernel, vjp_kernel = jax.vjp(run_kernel, q, k, v, g, b, w)

    # 1. Output finiteness
    assert bool(jnp.all(jnp.isfinite(o_kernel))), "Kernel forward output contains NaNs or Infs"
    assert bool(jnp.all(jnp.isfinite(o_ref))), "Reference forward output contains NaNs or Infs"

    tol = 1e-3 if precision == "fp32" else 1e-2
    err_o = rel_l2(o_kernel, o_ref)
    assert err_o < tol, f"Forward rel L2 error {err_o:.2e} exceeds {tol:.1e}"

    # 2. Gradient finiteness and parity
    grads_ref = vjp_ref(do)
    grads_kernel = vjp_kernel(do)

    grad_names = ["dq", "dk", "dv", "dg", "db", "dw"]
    for name, g_k, g_r in zip(grad_names, grads_kernel, grads_ref):
        assert bool(jnp.all(jnp.isfinite(g_k))), f"Kernel gradient {name} contains NaNs or Infs"
        assert bool(jnp.all(jnp.isfinite(g_r))), f"Reference gradient {name} contains NaNs or Infs"
        err_g = rel_l2(g_k, g_r)
        assert err_g < tol, f"Gradient {name} rel L2 error {err_g:.2e} exceeds {tol:.1e}"


def test_positive_decay_clamped():
    """Verify that positive log-decay (g > 0) is clamped to zero, matching reference and decode step."""
    B, L, Hk, Hv, K, V = 1, 128, 2, 4, 128, 128
    key = jax.random.PRNGKey(503)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    b = jax.nn.sigmoid(jax.random.normal(k3, (B, L, Hk, K), jnp.float32) * 0.39)
    w = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hv, V), jnp.float32) * 0.39)
    # Positive log-decay
    g = jnp.abs(jax.random.normal(k5, (B, L, Hk, K), jnp.float32) * 0.05) + 0.01

    out_kernel = chunk_gdn2(q, k, v, g, b, w, precision="fp32")
    out_ref = token_serial_gdn2(q, k, v, g, b, w)
    err = rel_l2(out_kernel, out_ref)
    assert err < 1e-3, f"Positive decay clamp rel L2 error {err:.2e} exceeds 1e-3"


def test_gate_straight_through_at_zero():
    """Verify that _nonpositive_gate preserves full gradient (not halved) at g=0."""
    from deltanet_tpu.gdn2 import _nonpositive_gate

    grad_fn = jax.grad(lambda x: jnp.sum(_nonpositive_gate(x)))
    g_zero = jnp.zeros(4, dtype=jnp.float32)
    # Derivative should be exactly 1.0 (straight-through), not 0.5 as jnp.minimum would give
    d = grad_fn(g_zero)
    assert jnp.allclose(d, 1.0), f"Expected gradient 1.0 at zero, got {d}"

    # For g > 0, gradient should be 0.0
    g_pos = jnp.ones(4, dtype=jnp.float32)
    d_pos = grad_fn(g_pos)
    assert jnp.allclose(d_pos, 0.0), f"Expected gradient 0.0 for positive, got {d_pos}"

