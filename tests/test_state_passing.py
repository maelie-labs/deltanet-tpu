"""State passing and chunk splitting continuity tests for Pallas TPU GDN-2 kernel.

Verifies that splitting a sequence into multiple chunks and passing recurrent state
across them produces output identically matching a single un-split execution.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from deltanet_tpu.gdn2 import chunk_gdn2


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
    g = -jnp.exp(jax.random.normal(k5, (B, L, Hk, K), jnp.float32) * 0.39 - 1.0)
    return q, k, v, g, b, w


@pytest.mark.parametrize("Hk,Hv", [(2, 2), (2, 4)])
@pytest.mark.parametrize("split_idx", [64, 128, 192])
def test_state_passing_continuity(Hk, Hv, split_idx):
    """Verify that state passing across arbitrary chunk boundaries matches single-call output."""
    B, L, K, V = 1, 256, 128, 128
    chunk_size = 128
    key = jax.random.PRNGKey(601 + split_idx)
    q, k, v, g, b, w = make_test_inputs(key, B, L, Hk, Hv, K, V)

    # 1. Single call with final state output
    o_full, final_state_full = chunk_gdn2(
        q, k, v, g, b, w,
        chunk_size=chunk_size,
        precision="fp32",
        output_final_state=True,
    )

    # 2. Split call into [0:split_idx] and [split_idx:L]
    q1, k1, v1, g1, b1, w1 = (
        q[:, :split_idx], k[:, :split_idx], v[:, :split_idx],
        g[:, :split_idx], b[:, :split_idx], w[:, :split_idx],
    )
    q2, k2, v2, g2, b2, w2 = (
        q[:, split_idx:], k[:, split_idx:], v[:, split_idx:],
        g[:, split_idx:], b[:, split_idx:], w[:, split_idx:],
    )

    o1, state_mid = chunk_gdn2(
        q1, k1, v1, g1, b1, w1,
        chunk_size=chunk_size,
        precision="fp32",
        output_final_state=True,
    )

    o2, final_state_split = chunk_gdn2(
        q2, k2, v2, g2, b2, w2,
        initial_state=state_mid,
        chunk_size=chunk_size,
        precision="fp32",
        output_final_state=True,
    )

    o_split = jnp.concatenate([o1, o2], axis=1)

    # Compare full vs split output
    err_o = rel_l2(o_split, o_full)
    assert err_o < 1e-4, f"Split output rel L2 error {err_o:.2e} exceeds 1e-4"

    # Compare final recurrent state
    err_s = rel_l2(final_state_split, final_state_full)
    assert err_s < 1e-4, f"Final state rel L2 error {err_s:.2e} exceeds 1e-4"
