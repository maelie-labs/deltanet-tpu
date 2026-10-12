"""Adversarial numerical robustness tests for GDN-2 forward and backward passes.

Tests synthetic edge cases comparing chunked GDN-2 kernels against the
canonical token-serial reference scan and direct triangular solves:
1. Intra-chunk unit lower-triangular doubling inverter parity against solve_triangular
2. Identical keys across multiple chunks (L=128 to 1024) with saturated erase gates (beta -> 1)
3. Near-identical key perturbations (eps=1e-4, 1e-2, 1e-1)
4. Periodic cycling keys with periods 2, 4, 8, 16, 32, 64
5. Precision parity under bf16 and fp32
6. Boundedness (|T| <= 1.0) under alternating and random sign patterns at C=64 and C=128
7. Full GDN-2 forward and backward passes on anti-aligned keys with vanishing decay (g = -10)
8. Guard validation asserting chunk size C <= 8 * B for degree-7 outer polynomial nilpotency.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from deltanet_tpu.gdn2 import (
    chunk_gdn2,
    _invert_unit_triangular_core,
)
from deltanet_tpu.reference import token_serial_gdn2


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    norm_b = np.linalg.norm(b)
    if norm_b == 0.0:
        return float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / (norm_b + 1e-12))


def make_adversarial_batch(
    case_type: str,
    B: int = 1,
    L: int = 256,
    Hk: int = 1,
    Hv: int = 1,
    K: int = 128,
    V: int = 128,
    seed: int = 42,
    period: int = 2,
    eps: float = 1e-4,
    g_val: float = -10.0,
    b_val: float = 5.0,
    k_scale: float = 1.0,
    num_segments: int = 1,
):
    key = jax.random.PRNGKey(seed)
    k_q, k_k, k_v, k_do = jax.random.split(key, 4)

    q = jax.random.normal(k_q, (B, L, Hk, K), dtype=jnp.float32)
    v = jax.random.normal(k_v, (B, L, Hv, V), dtype=jnp.float32)
    w = jnp.zeros((B, L, Hv, V), dtype=jnp.float32)  # sigmoid(0) = 0.5
    do = jax.random.normal(k_do, (B, L, Hv, V), dtype=jnp.float32)

    # Base decay and erase gate logits for fused prologue
    g = jnp.full((B, L, Hk, K), g_val, dtype=jnp.float32)
    b = jnp.full((B, L, Hk, K), b_val, dtype=jnp.float32)

    if case_type == "identical_keys":
        # Exactly one key vector repeated for all tokens
        k_single = jax.random.normal(k_k, (B, 1, Hk, K), dtype=jnp.float32) * k_scale
        k = jnp.repeat(k_single, L, axis=1)

    elif case_type == "near_identical_keys":
        k_single = jax.random.normal(k_k, (B, 1, Hk, K), dtype=jnp.float32) * k_scale
        noise = jax.random.normal(key, (B, L, Hk, K), dtype=jnp.float32) * eps
        k = jnp.repeat(k_single, L, axis=1) + noise

    elif case_type == "periodic_keys":
        # Cycle through 'period' distinct key vectors
        k_pat = jax.random.normal(k_k, (B, period, Hk, K), dtype=jnp.float32) * k_scale
        repeats = (L + period - 1) // period
        k = jnp.tile(k_pat, (1, repeats, 1, 1))[:, :L, :, :]

    elif case_type == "random_uniform":
        k = jax.random.normal(k_k, (B, L, Hk, K), dtype=jnp.float32) * k_scale

    else:
        raise ValueError(f"Unknown case_type: {case_type}")

    if num_segments > 1:
        # Document boundaries inside chunk: e.g. segments [0, 0, 1, 1, 2, 2, ...]
        seg_len = L // num_segments
        seg_ids = jnp.repeat(jnp.arange(num_segments, dtype=jnp.int32), seg_len)
        if seg_ids.shape[0] < L:
            seg_ids = jnp.pad(seg_ids, (0, L - seg_ids.shape[0]), constant_values=num_segments - 1)
        seg_ids = jnp.broadcast_to(seg_ids[None, :], (B, L))
    else:
        seg_ids = None

    return q, k, v, g, b, w, do, seg_ids


def test_inverter_instrumentation_all_ones():
    """Verify that _invert_unit_triangular_core produces exact inverse on all-ones A without error."""
    for C in [64, 128]:
        A = jnp.tril(jnp.ones((C, C), dtype=jnp.float32), -1)
        T = _invert_unit_triangular_core(A)
        L = jnp.eye(C, dtype=jnp.float32) + A
        res = jnp.matmul(L, T) - jnp.eye(C, dtype=jnp.float32)
        err = float(jnp.max(jnp.abs(res)))
        assert not bool(jnp.isnan(T).any()), f"T contains NaN for C={C}"
        assert not bool(jnp.isinf(T).any()), f"T contains Inf for C={C}"
        assert err < 1e-4, f"Inversion residual {err:.2e} exceeds tolerance 1e-4 for C={C}"
        assert float(jnp.max(jnp.abs(T))) <= 1.0 + 1e-5, f"T max abs {float(jnp.max(jnp.abs(T)))} exceeds 1.0"


@pytest.mark.parametrize("L", [128, 256, 512, 1024])
def test_adversarial_identical_keys(L):
    """Verify forward and backward finiteness and parity on identical keys across chunks."""
    q, k, v, g, b, w, do, seg_ids = make_adversarial_batch("identical_keys", L=L)

    def ref_fn(q, k, v, g, b, w):
        return token_serial_gdn2(q, k, v, g, b, w, segment_ids=seg_ids, fuse_prologue=True, qk_l2norm=True)

    def ker_fn(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, segment_ids=seg_ids, chunk_size=128, precision="bf16", fuse_prologue=True, qk_l2norm=True)

    o_ref, vjp_ref = jax.vjp(ref_fn, q, k, v, g, b, w)
    grads_ref = vjp_ref(do)

    o_k, vjp_k = jax.vjp(ker_fn, q, k, v, g, b, w)
    grads_k = vjp_k(do)

    assert bool(jnp.all(jnp.isfinite(o_k))), f"Forward output has non-finite values for L={L}"
    for idx, gk in enumerate(grads_k):
        assert bool(jnp.all(jnp.isfinite(gk))), f"Grad #{idx} has non-finite values for L={L}"

    err_o = rel_l2(o_k, o_ref)
    assert err_o < 0.10, f"Forward rel error {err_o:.2e} exceeds 0.10 for L={L}"

    def ker_fp32_fn(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, segment_ids=seg_ids, chunk_size=128, precision="fp32", fuse_prologue=True, qk_l2norm=True)

    _, vjp_k_fp32 = jax.vjp(ker_fp32_fn, q, k, v, g, b, w)
    grads_k_fp32 = vjp_k_fp32(do)
    for idx, (gk, gr) in enumerate(zip(grads_k_fp32, grads_ref)):
        err_g = rel_l2(gk, gr)
        assert err_g < 1e-4, f"fp32 Grad #{idx} rel error {err_g:.2e} exceeds 1e-4 for L={L}"


@pytest.mark.parametrize("eps", [1e-4, 1e-2, 1e-1])
def test_adversarial_near_identical_keys(eps):
    """Verify stability on near-identical keys with tiny perturbation."""
    q, k, v, g, b, w, do, seg_ids = make_adversarial_batch("near_identical_keys", L=512, eps=eps)

    def ref_fn(q, k, v, g, b, w):
        return token_serial_gdn2(q, k, v, g, b, w, segment_ids=seg_ids, fuse_prologue=True, qk_l2norm=True)

    def ker_fn(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, segment_ids=seg_ids, chunk_size=128, precision="bf16", fuse_prologue=True, qk_l2norm=True)

    o_ref, vjp_ref = jax.vjp(ref_fn, q, k, v, g, b, w)
    grads_ref = vjp_ref(do)

    o_k, vjp_k = jax.vjp(ker_fn, q, k, v, g, b, w)
    grads_k = vjp_k(do)

    assert bool(jnp.all(jnp.isfinite(o_k))), f"Forward non-finite for eps={eps}"
    for idx, gk in enumerate(grads_k):
        assert bool(jnp.all(jnp.isfinite(gk))), f"Grad #{idx} non-finite for eps={eps}"

    err_o = rel_l2(o_k, o_ref)
    assert err_o < 0.10, f"Forward rel error {err_o:.2e} exceeds 0.10 for eps={eps}"

    def ker_fp32_fn(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, segment_ids=seg_ids, chunk_size=128, precision="fp32", fuse_prologue=True, qk_l2norm=True)

    _, vjp_k_fp32 = jax.vjp(ker_fp32_fn, q, k, v, g, b, w)
    grads_k_fp32 = vjp_k_fp32(do)
    for idx, (gk, gr) in enumerate(zip(grads_k_fp32, grads_ref)):
        err_g = rel_l2(gk, gr)
        assert err_g < 1e-4, f"fp32 Grad #{idx} rel error {err_g:.2e} exceeds 1e-4 for eps={eps}"


@pytest.mark.parametrize("period", [2, 4, 8, 16, 32, 64])
def test_adversarial_periodic_keys(period):
    """Verify stability on periodic cycling keys."""
    q, k, v, g, b, w, do, seg_ids = make_adversarial_batch("periodic_keys", L=512, period=period)

    def ref_fn(q, k, v, g, b, w):
        return token_serial_gdn2(q, k, v, g, b, w, segment_ids=seg_ids, fuse_prologue=True, qk_l2norm=True)

    def ker_fn(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, segment_ids=seg_ids, chunk_size=128, precision="bf16", fuse_prologue=True, qk_l2norm=True)

    o_ref, vjp_ref = jax.vjp(ref_fn, q, k, v, g, b, w)
    grads_ref = vjp_ref(do)

    o_k, vjp_k = jax.vjp(ker_fn, q, k, v, g, b, w)
    grads_k = vjp_k(do)

    assert bool(jnp.all(jnp.isfinite(o_k))), f"Forward non-finite for period={period}"
    for idx, gk in enumerate(grads_k):
        assert bool(jnp.all(jnp.isfinite(gk))), f"Grad #{idx} non-finite for period={period}"

    err_o = rel_l2(o_k, o_ref)
    assert err_o < 0.05, f"Forward rel error {err_o:.2e} exceeds 0.05 for period={period}"


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_adversarial_fp32_and_bf16_parity(precision):
    """Verify identical keys parity in both fp32 and bf16 modes."""
    q, k, v, g, b, w, do, seg_ids = make_adversarial_batch("identical_keys", L=512)

    def ref_fn(q, k, v, g, b, w):
        return token_serial_gdn2(q, k, v, g, b, w, segment_ids=seg_ids, fuse_prologue=True, qk_l2norm=True)

    def ker_fn(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, segment_ids=seg_ids, chunk_size=128, precision=precision, fuse_prologue=True, qk_l2norm=True)

    o_ref, vjp_ref = jax.vjp(ref_fn, q, k, v, g, b, w)
    grads_ref = vjp_ref(do)

    o_k, vjp_k = jax.vjp(ker_fn, q, k, v, g, b, w)
    grads_k = vjp_k(do)

    assert bool(jnp.all(jnp.isfinite(o_k))), f"Forward non-finite for precision={precision}"
    for idx, gk in enumerate(grads_k):
        assert bool(jnp.all(jnp.isfinite(gk))), f"Grad #{idx} non-finite for precision={precision}"

    err_o = rel_l2(o_k, o_ref)
    tol = 1e-4 if precision == "fp32" else 0.10
    assert err_o < tol, f"Forward rel error {err_o:.2e} exceeds {tol:.1e} for precision={precision}"


@pytest.mark.parametrize("C", [64, 128])
def test_adversarial_opposite_signs_inverse(C):
    """Verify that _invert_unit_triangular_core maintains contractive bound |T| <= 1.0 on alternating and random signs."""
    eye = jnp.eye(C, dtype=jnp.float32)

    # 1. Alternating signs (+1, -1, +1, -1, ...)
    signs_alt = jnp.array([(-1.0) ** i for i in range(C)], dtype=jnp.float32)
    K_alt = signs_alt[:, None] * signs_alt[None, :]
    A_alt = jnp.tril(K_alt, -1)

    T_alt = _invert_unit_triangular_core(A_alt)
    assert not bool(jnp.isnan(T_alt).any()), f"T contains NaN for alternating signs at C={C}"
    assert not bool(jnp.isinf(T_alt).any()), f"T contains Inf for alternating signs at C={C}"

    res_alt = jnp.matmul(eye + A_alt, T_alt) - eye
    err_alt = float(jnp.max(jnp.abs(res_alt)))
    assert err_alt < 1e-4, f"Alternating signs residual {err_alt:.2e} exceeds 1e-4 for C={C}"
    max_entry_alt = float(jnp.max(jnp.abs(T_alt)))
    assert max_entry_alt <= 1.0 + 1e-5, f"Alternating signs max |T| {max_entry_alt:.2f} exceeds 1.0 for C={C}"

    # 2. Search across randomized sign patterns
    rng = np.random.RandomState(42)
    for trial in range(20):
        s = rng.choice([-1.0, 1.0], size=C)
        A_rand = jnp.array(np.tril(s[:, None] * s[None, :], -1), dtype=jnp.float32)
        T_rand = _invert_unit_triangular_core(A_rand)
        assert not bool(jnp.isnan(T_rand).any()), f"Trial {trial} NaN for C={C}"
        res_rand = jnp.matmul(eye + A_rand, T_rand) - eye
        assert float(jnp.max(jnp.abs(res_rand))) < 1e-4, f"Trial {trial} residual failed for C={C}"
        assert float(jnp.max(jnp.abs(T_rand))) <= 1.0 + 1e-5, f"Trial {trial} max |T| failed for C={C}"


def test_adversarial_opposite_signs_gdn2():
    """Verify forward and backward finiteness and parity under alternating keys with vanishing decay (g = -10) and beta -> 1."""
    C = 128
    B, L = 1, 512
    Hk, Hv = 2, 4
    K, V = 128, 128

    k_base = np.zeros((B, L, Hk, K), dtype=np.float32)
    signs = np.array([(-1.0) ** t for t in range(L)], dtype=np.float32)
    k_base[:, :, :, 0] = signs[None, :, None]
    k = jnp.array(k_base)

    q = jax.random.normal(jax.random.PRNGKey(1), (B, L, Hk, K), dtype=jnp.float32) * 0.1
    v = jax.random.normal(jax.random.PRNGKey(2), (B, L, Hv, V), dtype=jnp.float32) * 0.1
    # g = -10.0 (decay rate ~ -exp(-10) ~ -4.5e-5 ~ 0, vanishing decay / expansive), beta -> 1 (b = 10.0 -> sigmoid(10) ~ 1.0)
    g = jnp.full((B, L, Hk, K), -10.0, dtype=jnp.float32)
    b = jnp.full((B, L, Hk, K), 10.0, dtype=jnp.float32)
    w = jnp.zeros((B, L, Hv, V), dtype=jnp.float32)
    do = jax.random.normal(jax.random.PRNGKey(3), (B, L, Hv, V), dtype=jnp.float32)

    def ref_fn(q, k, v, g, b, w):
        return token_serial_gdn2(q, k, v, g, b, w, fuse_prologue=True, qk_l2norm=True)

    def ker_fn(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, chunk_size=C, precision="fp32", fuse_prologue=True, qk_l2norm=True)

    o_ref, vjp_ref = jax.vjp(ref_fn, q, k, v, g, b, w)
    grads_ref = vjp_ref(do)

    o_k, vjp_k = jax.vjp(ker_fn, q, k, v, g, b, w)
    grads_k = vjp_k(do)

    assert bool(jnp.all(jnp.isfinite(o_k))), f"Forward output non-finite for alternating keys (C={C})"
    for idx, gk in enumerate(grads_k):
        assert bool(jnp.all(jnp.isfinite(gk))), f"Grad #{idx} non-finite for alternating keys (C={C})"

    err_o = rel_l2(o_k, o_ref)
    assert err_o < 1e-4, f"Forward rel error {err_o:.2e} exceeds 1e-4 for C={C}"
    for idx, (gk, gr) in enumerate(zip(grads_k, grads_ref)):
        norm_gr = float(np.linalg.norm(np.asarray(gr)))
        if norm_gr > 1e-6:
            err_g = rel_l2(gk, gr)
            assert err_g < 1e-4, f"Grad #{idx} rel error {err_g:.2e} exceeds 1e-4 for C={C}"
        else:
            diff_g = float(np.max(np.abs(np.asarray(gk) - np.asarray(gr))))
            assert diff_g < 1e-6, f"Grad #{idx} abs diff {diff_g:.2e} exceeds 1e-6 for C={C}"


@pytest.mark.parametrize("C", [64, 128])
def test_inverter_direct_triangular_solve(C):
    """Verify that _invert_unit_triangular_core matches a direct triangular solve."""
    key = jax.random.PRNGKey(C)
    for _ in range(5):
        key, subkey = jax.random.split(key)
        A_raw = jax.random.uniform(subkey, (C, C), minval=-0.5, maxval=0.5)
        A = jnp.tril(A_raw, -1)
        eye = jnp.eye(C, dtype=jnp.float32)
        L = eye + A

        T_core = _invert_unit_triangular_core(A)
        T_direct = jax.scipy.linalg.solve_triangular(L, eye, lower=True, unit_diagonal=True)

        err = float(jnp.max(jnp.abs(T_core - T_direct)))
        rel_err = float(jnp.linalg.norm(T_core - T_direct) / jnp.linalg.norm(T_direct))
        assert err < 1e-4, f"Direct solve diff max {err:.2e} exceeds 1e-4 for C={C}"
        assert rel_err < 1e-4, f"Direct solve rel err {rel_err:.2e} exceeds 1e-4 for C={C}"


def test_inverter_guard_exceeds_max_block_rows():
    """Verify that _invert_unit_triangular_core raises ValueError when C > 8 * B."""
    C = 256
    A = jnp.tril(jnp.ones((C, C), dtype=jnp.float32), -1)
    with pytest.raises(ValueError, match="exceeds maximum supported"):
        _invert_unit_triangular_core(A)

