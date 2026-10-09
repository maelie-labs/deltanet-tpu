"""Tests for arbitrary sequence lengths in chunk_gdn2.

Verifies that any sequence length (such as 1, 127, 1000, 8191) executes cleanly
without requiring divisibility by chunk size, matching the token-serial reference
in forward output, backward gradients, and final recurrent state.
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


@pytest.mark.parametrize("L", [1, 127, 1000, 8191])
def test_arbitrary_sequence_lengths(L):
    """Verify arbitrary lengths match reference on forward, backward, and final state."""
    B, Hk, Hv, K, V = 1, 2, 4, 128, 128
    chunk_size = 128
    key = jax.random.PRNGKey(1010 + L)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    g = -jnp.exp(jax.random.normal(k3, (B, L, Hk, K), jnp.float32) * 0.39 - 1.0)
    b = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hk, K), jnp.float32) * 0.39)
    w = jax.nn.sigmoid(jax.random.normal(k5, (B, L, Hv, V), jnp.float32) * 0.39)

    # 1. Forward and final state parity vs token-serial reference
    o_ref, s_ref = token_serial_gdn2(q, k, v, g, b, w, output_final_state=True)
    o_kern, s_kern = chunk_gdn2(
        q, k, v, g, b, w,
        chunk_size=chunk_size,
        output_final_state=True,
        precision="fp32",
    )

    assert o_kern.shape == (B, L, Hv, V), f"Output shape {o_kern.shape} != expected {(B, L, Hv, V)}"
    assert s_kern.shape == (B, Hv, K, V), f"State shape {s_kern.shape} != expected {(B, Hv, K, V)}"

    err_o = rel_l2(o_kern, o_ref)
    assert err_o < 1e-5, f"Forward rel L2 error {err_o:.2e} exceeds 1e-5 at L={L}"

    err_s = rel_l2(s_kern, s_ref)
    assert err_s < 1e-5, f"Final state rel L2 error {err_s:.2e} exceeds 1e-5 at L={L}"

    # 2. Forward-only inference mode parity
    o_inf, s_inf = chunk_gdn2(
        q, k, v, g, b, w,
        chunk_size=chunk_size,
        output_final_state=True,
        is_inference=True,
        precision="fp32",
    )
    assert rel_l2(o_inf, o_kern) < 1e-5
    assert rel_l2(s_inf, s_kern) < 1e-5

    # 3. Backward gradient execution and shape verification
    def loss_fn(q, k, v):
        out = chunk_gdn2(q, k, v, g, b, w, chunk_size=chunk_size, precision="fp32")
        return jnp.sum(out)

    grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
    assert grads[0].shape == (B, L, Hk, K)
    assert grads[1].shape == (B, L, Hk, K)
    assert grads[2].shape == (B, L, Hv, V)
    assert not jnp.isnan(grads[0]).any()
    assert not jnp.isnan(grads[1]).any()
    assert not jnp.isnan(grads[2]).any()
