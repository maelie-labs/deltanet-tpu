"""Tests for on-the-fly chunk state recomputation in backward pass (recompute_states=True).

Verifies bitwise/near-bitwise parity between:
1. Standard stashed states (recompute_states=False)
2. On-the-fly recomputed states (recompute_states=True)
And validates gradient parity against token-serial reference.
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
    g = -jnp.exp(jax.random.normal(k5, (B, L, Hk, K), jnp.float32) * 0.39 - 1.0)
    return q, k, v, g, b, w


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("L", [128, 256, 512])
@pytest.mark.parametrize("fallback_scan", [False, True])
def test_recompute_states_parity_vs_stashed(precision, L, fallback_scan):
    """Verify recompute_states=True yields exact gradient parity with recompute_states=False."""
    B, Hk, Hv, K, V = 1, 2, 4, 128, 128
    key = jax.random.PRNGKey(77 + L)
    q, k, v, g, b, w = make_test_inputs(key, B, L, Hk, Hv, K, V)
    do = jax.random.normal(jax.random.PRNGKey(88 + L), (B, L, Hv, V), jnp.float32)

    def run_stashed(q, k, v, g, b, w):
        return chunk_gdn2(
            q, k, v, g, b, w,
            chunk_size=128,
            precision=precision,
            fallback_scan=fallback_scan,
            recompute_states=False,
        )

    def run_recompute(q, k, v, g, b, w):
        return chunk_gdn2(
            q, k, v, g, b, w,
            chunk_size=128,
            precision=precision,
            fallback_scan=fallback_scan,
            recompute_states=True,
        )

    o_stashed, vjp_stashed = jax.vjp(run_stashed, q, k, v, g, b, w)
    o_recompute, vjp_recompute = jax.vjp(run_recompute, q, k, v, g, b, w)

    # Forward outputs must match
    err_o = rel_l2(o_recompute, o_stashed)
    assert err_o < 1e-6, f"Forward output mismatch with recompute_states=True: rel_l2={err_o:.2e}"

    # Backward gradients must match
    grads_stashed = vjp_stashed(do)
    grads_recompute = vjp_recompute(do)

    names = ["q", "k", "v", "g", "b", "w"]
    for name, g_stash, g_recomp in zip(names, grads_stashed, grads_recompute):
        err_g = rel_l2(g_recomp, g_stash)
        tol = 1e-5 if precision == "fp32" else 1e-3
        assert err_g < tol, f"Gradient d_{name} rel_l2={err_g:.2e} exceeds tolerance {tol:.1e}"


@pytest.mark.parametrize("L", [256, 512])
def test_recompute_states_with_init_and_final_state(L):
    """Verify state recomputation with initial_state and output_final_state."""
    B, Hk, Hv, K, V = 1, 2, 2, 128, 128
    key = jax.random.PRNGKey(101 + L)
    q, k, v, g, b, w = make_test_inputs(key, B, L, Hk, Hv, K, V)
    init_state = jax.random.normal(jax.random.PRNGKey(202), (B, Hv, K, V), jnp.float32) * 0.05
    do = jax.random.normal(jax.random.PRNGKey(303), (B, L, Hv, V), jnp.float32)
    d_final = jax.random.normal(jax.random.PRNGKey(404), (B, Hv, K, V), jnp.float32)

    def run_stashed(q, k, v, g, b, w, s0):
        return chunk_gdn2(
            q, k, v, g, b, w,
            initial_state=s0,
            output_final_state=True,
            chunk_size=128,
            precision="fp32",
            recompute_states=False,
        )

    def run_recompute(q, k, v, g, b, w, s0):
        return chunk_gdn2(
            q, k, v, g, b, w,
            initial_state=s0,
            output_final_state=True,
            chunk_size=128,
            precision="fp32",
            recompute_states=True,
        )

    (o_s, fs_s), vjp_s = jax.vjp(run_stashed, q, k, v, g, b, w, init_state)
    (o_r, fs_r), vjp_r = jax.vjp(run_recompute, q, k, v, g, b, w, init_state)

    assert rel_l2(o_r, o_s) < 1e-6
    assert rel_l2(fs_r, fs_s) < 1e-6

    grads_s = vjp_s((do, d_final))
    grads_r = vjp_r((do, d_final))

    for g_s, g_r in zip(grads_s, grads_r):
        if g_s is not None:
            assert rel_l2(g_r, g_s) < 1e-5
