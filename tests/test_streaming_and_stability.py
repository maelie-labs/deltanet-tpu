"""Tests for streaming prefill and stability guard features."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from deltanet_tpu.gdn2 import chunk_gdn2, gdn2_decode_step


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


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("L, block_size", [(512, 256), (1024, 256), (1024, 512)])
def test_streaming_prefill_matches_monolithic(precision, L, block_size):
    """Verify streaming prefill produces bitwise/near-bitwise identical output and state."""
    B, Hk, Hv, K, V = 1, 2, 4, 128, 128
    key = jax.random.PRNGKey(99 + L)
    q, k, v, g, b, w = make_test_inputs(key, B, L, Hk, Hv, K, V)

    # Monolithic prefill
    out_mono, state_mono = chunk_gdn2(
        q, k, v, g, b, w,
        chunk_size=128,
        precision=precision,
        is_inference=True,
        output_final_state=True,
        streaming_prefill=False,
    )

    # Streaming prefill
    out_stream, state_stream = chunk_gdn2(
        q, k, v, g, b, w,
        chunk_size=128,
        precision=precision,
        is_inference=True,
        output_final_state=True,
        streaming_prefill=True,
        prefill_block_size=block_size,
    )

    tol = 1e-5 if precision == "fp32" else 1e-2
    err_out = rel_l2(out_stream, out_mono)
    err_state = rel_l2(state_stream, state_mono)

    assert err_out < tol, f"Streaming prefill output mismatch: rel_l2={err_out:.2e}"
    assert err_state < tol, f"Streaming prefill final state mismatch: rel_l2={err_state:.2e}"


def test_stability_guard_decode_step():
    """Verify stability_guard caps the decode state norm at stability_tau."""
    B, Hk, Hv, K, V = 1, 2, 2, 128, 128
    key = jax.random.PRNGKey(1234)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    q = jax.random.normal(k0, (B, Hk, K), jnp.float32)
    k = jax.random.normal(k1, (B, Hk, K), jnp.float32)
    v = jax.random.normal(k2, (B, Hv, V), jnp.float32)
    # Erase gate with negative values and decay with positive values
    b = jnp.full((B, Hk, K), -0.5, dtype=jnp.float32)
    g = jnp.full((B, Hk, K), 0.5, dtype=jnp.float32)
    # Massive initial state
    state = jnp.full((B, Hv, K, V), 100.0, dtype=jnp.float32)
    w = jnp.ones((B, Hv, V), dtype=jnp.float32)

    # Run with stability guard enabled
    tau = 50.0
    _, state_guarded = gdn2_decode_step(
        q, k, v, g, b, w, state,
        stability_guard=True,
        stability_tau=tau,
    )

    max_head_norm = float(np.linalg.norm(state_guarded, axis=(-2, -1)).max())
    assert max_head_norm <= tau + 1e-2, f"Head state norm {max_head_norm} exceeded tau {tau}"


def test_stability_guard_chunk_prefill():
    """Verify stability guard bounds state during prefill."""
    B, L, Hk, Hv, K, V = 1, 256, 2, 2, 128, 128
    key = jax.random.PRNGKey(5678)
    q, k, v, _, _, w = make_test_inputs(key, B, L, Hk, Hv, K, V)
    # Unstable gates
    b = jnp.full((B, L, Hk, K), -0.1, dtype=jnp.float32)
    g = jnp.full((B, L, Hk, K), 0.2, dtype=jnp.float32)

    tau = 20.0
    _, state_guarded = chunk_gdn2(
        q, k, v, g, b, w,
        is_inference=True,
        output_final_state=True,
        stability_guard=True,
        stability_tau=tau,
    )

    max_head_norm = float(np.linalg.norm(state_guarded, axis=(-2, -1)).max())
    assert max_head_norm <= tau + 1e-2, f"Prefill head state norm {max_head_norm} exceeded tau {tau}"
