"""Tests for forward-only prefill and single-token decode inference."""

import math
import jax
import jax.numpy as jnp
import pytest

from deltanet_tpu import chunk_gdn2, gdn2_decode_step


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_chunked_prefill_pieces_equal_single_call(dtype):
    """Verify that chunked prefill evaluated in pieces matches a single call."""
    B, L, Hk, Hv, K, V = 1, 512, 2, 4, 128, 128
    key = jax.random.PRNGKey(42)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    jnp_dtype = jnp.float32 if dtype == "fp32" else jnp.bfloat16
    q = jax.random.normal(k0, (B, L, Hk, K), dtype=jnp_dtype) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), dtype=jnp_dtype) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), dtype=jnp_dtype) * 0.1
    g = -jnp.exp(jax.random.normal(k3, (B, L, Hk, K), dtype=jnp.float32) - 1.0)
    b = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hk, K), dtype=jnp.float32))
    w = jax.nn.sigmoid(jax.random.normal(k5, (B, L, Hv, V), dtype=jnp.float32))

    out_full, state_full = chunk_gdn2(
        q, k, v, g, b, w,
        is_inference=True,
        output_final_state=True,
        chunk_size=128,
        precision=dtype,
    )

    split_idx = 256
    out1, state1 = chunk_gdn2(
        q[:, :split_idx], k[:, :split_idx], v[:, :split_idx],
        g[:, :split_idx], b[:, :split_idx], w[:, :split_idx],
        is_inference=True,
        output_final_state=True,
        chunk_size=128,
        precision=dtype,
    )

    out2, state2 = chunk_gdn2(
        q[:, split_idx:], k[:, split_idx:], v[:, split_idx:],
        g[:, split_idx:], b[:, split_idx:], w[:, split_idx:],
        initial_state=state1,
        is_inference=True,
        output_final_state=True,
        chunk_size=128,
        precision=dtype,
    )

    out_pieces = jnp.concatenate([out1, out2], axis=1)

    tol = 1e-4 if dtype == "fp32" else 1e-2
    rel_out = jnp.linalg.norm(out_full - out_pieces) / jnp.linalg.norm(out_full)
    rel_state = jnp.linalg.norm(state_full - state2) / jnp.linalg.norm(state_full)

    assert rel_out < tol, f"Prefill pieces output mismatch: {rel_out}"
    assert rel_state < tol, f"Prefill pieces state mismatch: {rel_state}"


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_decode_steps_equal_chunk_gdn2_with_reset(dtype):
    """Verify that N decode steps match chunk_gdn2 on the same N tokens with document reset."""
    B, N, Hk, Hv, K, V = 1, 128, 2, 4, 128, 128
    key = jax.random.PRNGKey(101)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    jnp_dtype = jnp.float32 if dtype == "fp32" else jnp.bfloat16
    q = jax.random.normal(k0, (B, N, Hk, K), dtype=jnp_dtype) * 0.1
    k = jax.random.normal(k1, (B, N, Hk, K), dtype=jnp_dtype) * 0.1
    v = jax.random.normal(k2, (B, N, Hv, V), dtype=jnp_dtype) * 0.1
    g = -jnp.exp(jax.random.normal(k3, (B, N, Hk, K), dtype=jnp.float32) - 1.0)
    b = jax.nn.sigmoid(jax.random.normal(k4, (B, N, Hk, K), dtype=jnp.float32))
    w = jax.nn.sigmoid(jax.random.normal(k5, (B, N, Hv, V), dtype=jnp.float32))

    # Document reset at N // 2
    reset_idx = N // 2
    segment_ids = (jnp.arange(N) >= reset_idx).astype(jnp.int32)[None]

    out_chunk, state_chunk = chunk_gdn2(
        q, k, v, g, b, w,
        segment_ids=segment_ids,
        is_inference=True,
        output_final_state=True,
        chunk_size=128,
        precision=dtype,
    )

    # Step by step decode
    state = jnp.zeros((B, Hv, K, V), dtype=jnp.float32)
    outs = []
    for t in range(N):
        reset = (t == reset_idx)
        ot, state = gdn2_decode_step(
            q[:, t:t+1], k[:, t:t+1], v[:, t:t+1],
            g[:, t:t+1], b[:, t:t+1], w[:, t:t+1],
            state, reset=reset,
        )
        outs.append(ot)

    out_decode = jnp.concatenate(outs, axis=1)

    tol = 1e-4 if dtype == "fp32" else 1e-2
    rel_out = jnp.linalg.norm(out_chunk - out_decode) / jnp.linalg.norm(out_chunk)
    rel_state = jnp.linalg.norm(state_chunk - state) / jnp.linalg.norm(state_chunk)

    assert rel_out < tol, f"Decode steps output mismatch: {rel_out}"
    assert rel_state < tol, f"Decode steps state mismatch: {rel_state}"


def test_decode_step_3d_inputs():
    """Verify gdn2_decode_step handles 3D inputs without sequence dimension."""
    B, Hk, Hv, K, V = 2, 2, 4, 128, 128
    key = jax.random.PRNGKey(7)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    q = jax.random.normal(k0, (B, Hk, K), dtype=jnp.float32)
    k = jax.random.normal(k1, (B, Hk, K), dtype=jnp.float32)
    v = jax.random.normal(k2, (B, Hv, V), dtype=jnp.float32)
    g = -jnp.ones((B, Hk, K), dtype=jnp.float32)
    b = jnp.ones((B, Hk, K), dtype=jnp.float32) * 0.5
    w = jnp.ones((B, Hv, V), dtype=jnp.float32) * 0.5
    state = jnp.zeros((B, Hv, K, V), dtype=jnp.float32)

    o, new_state = gdn2_decode_step(q, k, v, g, b, w, state)
    assert o.shape == (B, Hv, V)
    assert new_state.shape == (B, Hv, K, V)
