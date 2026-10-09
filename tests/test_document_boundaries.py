"""Document boundary tests for Pallas TPU GDN-2 kernel.

Verifies that intra-chunk document resets and length-1 documents
produce outputs and gradients matching token-serial reference.
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


def test_document_boundaries_cu_seqlens():
    """Verify document resets specified via cu_seqlens, including length-1 documents."""
    B, L, Hk, Hv, K, V = 1, 256, 2, 4, 128, 128
    chunk_size = 128
    key = jax.random.PRNGKey(301)
    q, k, v, g, b, w = make_test_inputs(key, B, L, Hk, Hv, K, V)
    do = jax.random.normal(jax.random.PRNGKey(302), (B, L, Hv, V), jnp.float32)

    # Document boundaries: doc0=50 tokens, doc1=1 token (length-1), doc2=100 tokens, doc3=105 tokens
    cu_seqlens = jnp.array([0, 50, 51, 151, 256], dtype=jnp.int32)

    def run_ref(q, k, v, g, b, w):
        return token_serial_gdn2(q, k, v, g, b, w, cu_seqlens=cu_seqlens)

    def run_kernel(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, cu_seqlens=cu_seqlens, chunk_size=chunk_size, precision="fp32")

    o_ref, vjp_ref = jax.vjp(run_ref, q, k, v, g, b, w)
    o_kernel, vjp_kernel = jax.vjp(run_kernel, q, k, v, g, b, w)

    err_o = rel_l2(o_kernel, o_ref)
    assert err_o < 1e-5, f"Output rel L2 error {err_o:.2e} exceeds tolerance 1e-5"

    grads_ref = vjp_ref(do)
    grads_kernel = vjp_kernel(do)

    grad_names = ["dq", "dk", "dv", "dg", "db", "dw"]
    for name, g_k, g_r in zip(grad_names, grads_kernel, grads_ref):
        err_g = rel_l2(g_k, g_r)
        assert err_g < 1e-5, f"Gradient {name} rel L2 error {err_g:.2e} exceeds tolerance 1e-5"


def test_document_boundaries_segment_ids():
    """Verify document resets specified via explicit segment_ids per token."""
    B, L, Hk, Hv, K, V = 1, 256, 2, 4, 128, 128
    chunk_size = 128
    key = jax.random.PRNGKey(401)
    q, k, v, g, b, w = make_test_inputs(key, B, L, Hk, Hv, K, V)
    do = jax.random.normal(jax.random.PRNGKey(402), (B, L, Hv, V), jnp.float32)

    # Segment IDs: frequent changes creating many small documents
    # 0..20: 0; 21: 1; 22..127: 2; 128..129: 3; 130..255: 4
    seg_ids = np.zeros((B, L), dtype=np.int32)
    seg_ids[:, 21] = 1
    seg_ids[:, 22:128] = 2
    seg_ids[:, 128:130] = 3
    seg_ids[:, 130:] = 4
    seg_ids = jnp.asarray(seg_ids)

    def run_ref(q, k, v, g, b, w):
        return token_serial_gdn2(q, k, v, g, b, w, segment_ids=seg_ids)

    def run_kernel(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, segment_ids=seg_ids, chunk_size=chunk_size, precision="fp32")

    o_ref, vjp_ref = jax.vjp(run_ref, q, k, v, g, b, w)
    o_kernel, vjp_kernel = jax.vjp(run_kernel, q, k, v, g, b, w)

    err_o = rel_l2(o_kernel, o_ref)
    assert err_o < 1e-5, f"Output rel L2 error {err_o:.2e} exceeds tolerance 1e-5"

    grads_ref = vjp_ref(do)
    grads_kernel = vjp_kernel(do)

    grad_names = ["dq", "dk", "dv", "dg", "db", "dw"]
    for name, g_k, g_r in zip(grad_names, grads_kernel, grads_ref):
        err_g = rel_l2(g_k, g_r)
        assert err_g < 1e-5, f"Gradient {name} rel L2 error {err_g:.2e} exceeds tolerance 1e-5"


def test_cu_seqlens_under_jit():
    """Verify that cu_seqlens works under jax.jit without ConcretizationTypeError."""
    B, L, Hk, Hv, K, V = 1, 128, 2, 4, 128, 128
    key = jax.random.PRNGKey(501)
    q, k, v, g, b, w = make_test_inputs(key, B, L, Hk, Hv, K, V)

    cu_list = [0, 48, 128]

    @jax.jit
    def jitted_fn(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, cu_seqlens=cu_list, precision="fp32")

    out_jit = jitted_fn(q, k, v, g, b, w)
    assert out_jit.shape == (B, L, Hv, V)

    out_ref = token_serial_gdn2(q, k, v, g, b, w, cu_seqlens=cu_list)
    assert rel_l2(out_jit, out_ref) < 1e-5


def test_cu_seqlens_validation():
    """Verify cu_seqlens validation errors."""
    B, L, Hk, Hv, K, V = 1, 128, 2, 4, 128, 128
    key = jax.random.PRNGKey(502)
    q, k, v, g, b, w = make_test_inputs(key, B, L, Hk, Hv, K, V)

    with pytest.raises(ValueError, match="cu_seqlens\\[0\\] must be 0"):
        chunk_gdn2(q, k, v, g, b, w, cu_seqlens=[10, 64, 128])

    seg_ids = jnp.zeros((B, L), dtype=jnp.int32)
    with pytest.raises(ValueError, match="Cannot pass both segment_ids and cu_seqlens"):
        chunk_gdn2(q, k, v, g, b, w, segment_ids=seg_ids, cu_seqlens=[0, 64, 128])


@pytest.mark.parametrize("fallback_scan", [False, True])
@pytest.mark.parametrize("seg_pattern", [
    [0] * 40 + [1] * 40 + [0] * 176,
    [1] * 30 + [0] * 10 + [2] * 30 + [0] * 10 + [3] * 176,
])
def test_reused_segment_ids_weak_decay(fallback_scan, seg_pattern):
    """Verify document resets with reused IDs, pad gaps, and weak decay (g ~ -0.01)."""
    B, L, Hk, Hv, K, V = 1, 256, 2, 4, 128, 128
    chunk_size = 128
    key = jax.random.PRNGKey(601)
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    b = jax.nn.sigmoid(jax.random.normal(k3, (B, L, Hk, K), jnp.float32) * 0.39)
    w = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hv, V), jnp.float32) * 0.39)
    # Weak decay regime: g ~ -0.01
    g = jnp.full((B, L, Hk, K), -0.01, dtype=jnp.float32)
    do = jax.random.normal(jax.random.PRNGKey(602), (B, L, Hv, V), jnp.float32)

    seg_ids = jnp.array([seg_pattern], dtype=jnp.int32)

    def run_ref(q, k, v, g, b, w):
        return token_serial_gdn2(q, k, v, g, b, w, segment_ids=seg_ids)

    def run_kernel(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, segment_ids=seg_ids, chunk_size=chunk_size, precision="fp32", fallback_scan=fallback_scan)

    o_ref, vjp_ref = jax.vjp(run_ref, q, k, v, g, b, w)
    o_kernel, vjp_kernel = jax.vjp(run_kernel, q, k, v, g, b, w)

    err_o = rel_l2(o_kernel, o_ref)
    assert err_o < 1e-4, f"Output rel L2 error {err_o:.2e} exceeds tolerance 1e-4 (fallback_scan={fallback_scan})"

    grads_ref = vjp_ref(do)
    grads_kernel = vjp_kernel(do)

    grad_names = ["dq", "dk", "dv", "dg", "db", "dw"]
    for name, g_k, g_r in zip(grad_names, grads_kernel, grads_ref):
        err_g = rel_l2(g_k, g_r)
        assert err_g < 1e-4, f"Gradient {name} rel L2 error {err_g:.2e} exceeds tolerance 1e-4 (fallback_scan={fallback_scan})"
