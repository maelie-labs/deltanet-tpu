"""Parity tests comparing Pallas TPU kernel output and gradients against token-serial reference."""

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


PARITY_LENGTHS = [128, 256, 512] + ([8192] if jax.default_backend() == "tpu" else [])


@pytest.mark.parametrize("precision, tol_out, tol_grad", [
    ("fp32", 1e-5, 1e-5),
    ("bf16", 1e-2, 1e-2),
])
@pytest.mark.parametrize("L", PARITY_LENGTHS)
def test_parity_forward_and_backward(precision, tol_out, tol_grad, L):
    """Verify forward and backward gradients against token-serial reference."""
    B, Hk, Hv, K, V = 1, 2, 4, 128, 128
    key = jax.random.PRNGKey(42 + L)
    q, k, v, g, b, w = make_test_inputs(key, B, L, Hk, Hv, K, V)
    do = jax.random.normal(jax.random.PRNGKey(100 + L), (B, L, Hv, V), jnp.float32)

    def run_ref(q, k, v, g, b, w):
        return token_serial_gdn2(q, k, v, g, b, w)

    def run_kernel(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, chunk_size=128, precision=precision)

    # 1. Forward output
    o_ref, vjp_ref = jax.vjp(run_ref, q, k, v, g, b, w)
    o_kernel, vjp_kernel = jax.vjp(run_kernel, q, k, v, g, b, w)

    err_o = rel_l2(o_kernel, o_ref)
    assert err_o < tol_out, f"Forward rel L2 error {err_o:.2e} exceeds tolerance {tol_out:.1e}"

    # 2. Backward gradients
    grads_ref = vjp_ref(do)
    grads_kernel = vjp_kernel(do)

    grad_names = ["dq", "dk", "dv", "dg", "db", "dw"]
    errs_g = {}
    for name, g_k, g_r in zip(grad_names, grads_kernel, grads_ref):
        err_g = rel_l2(g_k, g_r)
        errs_g[name] = err_g
        assert err_g < tol_grad, f"Gradient {name} rel L2 {err_g:.2e} exceeds tolerance {tol_grad:.1e}"
    max_g = max(errs_g.values())
    print(f"[measured] Parity L={L} {precision}: out rel L2 = {err_o:.4e}, max grad rel L2 = {max_g:.4e}", flush=True)


def test_parity_qk_l2norm_and_scale():
    """Verify forward and backward gradients with qk_l2norm=True, scale=0.5, and G=2."""
    B, L, Hk, Hv, K, V = 1, 128, 2, 4, 128, 128
    key = jax.random.PRNGKey(777)
    q, k, v, g, b, w = make_test_inputs(key, B, L, Hk, Hv, K, V)
    do = jax.random.normal(jax.random.PRNGKey(888), (B, L, Hv, V), jnp.float32)

    def run_ref(q, k, v, g, b, w):
        return token_serial_gdn2(q, k, v, g, b, w, scale=0.5, qk_l2norm=True)

    def run_pallas(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, chunk_size=128, scale=0.5, qk_l2norm=True, precision="fp32")

    def run_fallback_bf16(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, chunk_size=128, scale=0.5, qk_l2norm=True, precision="bf16", fallback_scan=True)

    # 1. Pallas kernel in fp32 vs reference (tolerance 1e-5)
    o_ref, vjp_ref = jax.vjp(run_ref, q, k, v, g, b, w)
    o_pallas, vjp_pallas = jax.vjp(run_pallas, q, k, v, g, b, w)

    err_o = rel_l2(o_pallas, o_ref)
    assert err_o < 1e-5, f"Pallas forward rel L2 error {err_o:.2e} exceeds 1e-5"

    grads_ref = vjp_ref(do)
    grads_pallas = vjp_pallas(do)
    grad_names = ["dq", "dk", "dv", "dg", "db", "dw"]
    for name, g_k, g_r in zip(grad_names, grads_pallas, grads_ref):
        err_g = rel_l2(g_k, g_r)
        assert err_g < 1e-5, f"Pallas gradient {name} rel L2 {err_g:.2e} exceeds 1e-5"

    # 2. fallback_scan in bf16 vs reference (tolerance 1e-2)
    o_fb, vjp_fb = jax.vjp(run_fallback_bf16, q, k, v, g, b, w)
    err_o_fb = rel_l2(o_fb, o_ref)
    assert err_o_fb < 1e-2, f"fallback_scan bf16 forward rel L2 error {err_o_fb:.2e} exceeds 1e-2"

    grads_fb = vjp_fb(do)
    for name, g_k, g_r in zip(grad_names, grads_fb, grads_ref):
        err_g = rel_l2(g_k, g_r)
        assert err_g < 1e-2, f"fallback_scan bf16 gradient {name} rel L2 {err_g:.2e} exceeds 1e-2"


def test_parity_fuse_prologue():
    """Verify forward output and backward cotangents with fuse_prologue=True against token-serial reference."""
    B, L, Hk, Hv, K, V = 1, 128, 2, 4, 128, 128
    key = jax.random.PRNGKey(999)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    # Raw un-activated logits for g, b, w
    b = jax.random.normal(k3, (B, L, Hk, K), jnp.float32) * 0.39
    w = jax.random.normal(k4, (B, L, Hv, V), jnp.float32) * 0.39
    g = jax.random.normal(k5, (B, L, Hk, K), jnp.float32) * 0.39 - 1.0
    do = jax.random.normal(jax.random.PRNGKey(1001), (B, L, Hv, V), jnp.float32)

    def run_ref(q, k, v, g, b, w):
        return token_serial_gdn2(q, k, v, g, b, w, fuse_prologue=True, qk_l2norm=True)

    def run_pallas(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, chunk_size=128, precision="fp32", fuse_prologue=True, qk_l2norm=True)

    def run_fallback_bf16(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, chunk_size=128, precision="bf16", fallback_scan=True, fuse_prologue=True, qk_l2norm=True)

    # 1. Pallas kernel in fp32 vs reference (tolerance 1e-5)
    o_ref, vjp_ref = jax.vjp(run_ref, q, k, v, g, b, w)
    o_pallas, vjp_pallas = jax.vjp(run_pallas, q, k, v, g, b, w)

    err_o = rel_l2(o_pallas, o_ref)
    assert err_o < 1e-5, f"Pallas fuse_prologue forward rel L2 error {err_o:.2e} exceeds 1e-5"

    grads_ref = vjp_ref(do)
    grads_pallas = vjp_pallas(do)
    grad_names = ["dq", "dk", "dv", "dg", "db", "dw"]
    for name, g_k, g_r in zip(grad_names, grads_pallas, grads_ref):
        err_g = rel_l2(g_k, g_r)
        assert err_g < 1e-5, f"Pallas fuse_prologue gradient {name} rel L2 {err_g:.2e} exceeds 1e-5"

    # 2. fallback_scan in bf16 vs reference (tolerance 1e-2)
    o_fb, vjp_fb = jax.vjp(run_fallback_bf16, q, k, v, g, b, w)
    err_o_fb = rel_l2(o_fb, o_ref)
    assert err_o_fb < 1e-2, f"fallback_scan bf16 fuse_prologue forward rel L2 error {err_o_fb:.2e} exceeds 1e-2"

    grads_fb = vjp_fb(do)
    for name, g_k, g_r in zip(grad_names, grads_fb, grads_ref):
        err_g = rel_l2(g_k, g_r)
        assert err_g < 1e-2, f"fallback_scan bf16 fuse_prologue gradient {name} rel L2 {err_g:.2e} exceeds 1e-2"


@pytest.mark.parametrize("L", [300, 384])
def test_parity_fuse_prologue_unaligned(L):
    """Verify unaligned sequence lengths with fuse_prologue=True, final state, cotangents, and segment IDs."""
    from deltanet_tpu import chunk_gdn, chunk_kda

    B, Hk, Hv, K, V = 1, 2, 4, 128, 128
    key = jax.random.PRNGKey(2026 + L)
    k0, k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 9)

    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    g = jax.random.normal(k3, (B, L, Hk, K), jnp.float32) * 0.39 - 1.0
    b = jax.random.normal(k4, (B, L, Hk, K), jnp.float32) * 0.39
    w = jax.random.normal(k5, (B, L, Hv, V), jnp.float32) * 0.39
    init_state = jax.random.normal(k6, (B, Hv, K, V), jnp.float32) * 0.05
    seg_ids = jnp.array([[0] * (L // 3) + [1] * (L // 3) + [2] * (L - 2 * (L // 3))], dtype=jnp.int32)

    do = jax.random.normal(k7, (B, L, Hv, V), jnp.float32)
    ds = jax.random.normal(k8, (B, Hv, K, V), jnp.float32)

    # 1. GDN-2: forward, final state, and backward cotangents with initial state and segment IDs
    def loss_ref(q, k, v, g, b, w, init_s):
        o, s = token_serial_gdn2(
            q, k, v, g, b, w,
            segment_ids=seg_ids, initial_state=init_s,
            fuse_prologue=True, qk_l2norm=True, output_final_state=True,
        )
        return jnp.sum(o * do) + jnp.sum(s * ds)

    def loss_pallas(q, k, v, g, b, w, init_s):
        o, s = chunk_gdn2(
            q, k, v, g, b, w,
            segment_ids=seg_ids, initial_state=init_s,
            chunk_size=128, precision="fp32",
            fuse_prologue=True, qk_l2norm=True, output_final_state=True,
        )
        return jnp.sum(o * do) + jnp.sum(s * ds)

    o_ref, s_ref = token_serial_gdn2(
        q, k, v, g, b, w,
        segment_ids=seg_ids, initial_state=init_state,
        fuse_prologue=True, qk_l2norm=True, output_final_state=True,
    )
    o_pallas, s_pallas = chunk_gdn2(
        q, k, v, g, b, w,
        segment_ids=seg_ids, initial_state=init_state,
        chunk_size=128, precision="fp32",
        fuse_prologue=True, qk_l2norm=True, output_final_state=True,
    )

    err_o = rel_l2(o_pallas, o_ref)
    err_s = rel_l2(s_pallas, s_ref)
    assert err_o < 1e-5, f"GDN-2 L={L} forward rel L2 error {err_o:.2e} exceeds 1e-5"
    assert err_s < 1e-5, f"GDN-2 L={L} final state rel L2 error {err_s:.2e} exceeds 1e-5"

    g_ref = jax.grad(loss_ref, argnums=(0, 1, 2, 3, 4, 5, 6))(q, k, v, g, b, w, init_state)
    g_pallas = jax.grad(loss_pallas, argnums=(0, 1, 2, 3, 4, 5, 6))(q, k, v, g, b, w, init_state)
    grad_names = ["dq", "dk", "dv", "dg", "db", "dw", "d_init"]
    for name, gk, gr in zip(grad_names, g_pallas, g_ref):
        err_g = rel_l2(gk, gr)
        assert err_g < 1e-5, f"GDN-2 L={L} gradient {name} rel L2 {err_g:.2e} exceeds 1e-5"

    # 2. GDN (scalar decay, shared beta)
    g_scal = jax.random.normal(k3, (B, L, Hk, 1), jnp.float32) * 0.39 - 1.0
    beta = jax.random.normal(k4, (B, L, Hk, 1), jnp.float32) * 0.39
    o_gdn, s_gdn = chunk_gdn(
        q, k, v, g_scal, beta,
        segment_ids=seg_ids, initial_state=init_state,
        fuse_prologue=True, qk_l2norm=True, output_final_state=True,
        precision="fp32",
    )
    b_broad = jnp.broadcast_to(beta, (B, L, Hk, K))
    w_broad = jnp.broadcast_to(jnp.repeat(beta, Hv // Hk, axis=2), (B, L, Hv, V))
    g_broad = jnp.broadcast_to(g_scal, (B, L, Hk, K))
    o_ref_gdn, s_ref_gdn = token_serial_gdn2(
        q, k, v, g_broad, b_broad, w_broad,
        segment_ids=seg_ids, initial_state=init_state,
        fuse_prologue=True, qk_l2norm=True, output_final_state=True,
    )
    assert rel_l2(o_gdn, o_ref_gdn) < 1e-5, f"GDN L={L} forward rel error exceeds 1e-5"
    assert rel_l2(s_gdn, s_ref_gdn) < 1e-5, f"GDN L={L} state rel error exceeds 1e-5"

    # 3. KDA (channel decay, shared beta)
    o_kda, s_kda = chunk_kda(
        q, k, v, g, beta,
        segment_ids=seg_ids, initial_state=init_state,
        fuse_prologue=True, qk_l2norm=True, output_final_state=True,
        precision="fp32",
    )
    o_ref_kda, s_ref_kda = token_serial_gdn2(
        q, k, v, g, b_broad, w_broad,
        segment_ids=seg_ids, initial_state=init_state,
        fuse_prologue=True, qk_l2norm=True, output_final_state=True,
    )
    assert rel_l2(o_kda, o_ref_kda) < 1e-5, f"KDA L={L} forward rel error exceeds 1e-5"
    assert rel_l2(s_kda, s_ref_kda) < 1e-5, f"KDA L={L} state rel error exceeds 1e-5"


def test_parity_fuse_prologue_scale():
    """Verify scale plumbing on the fused path."""
    B, L, Hk, Hv, K, V = 1, 128, 2, 4, 128, 128
    key = jax.random.PRNGKey(333)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    g = jax.random.normal(k3, (B, L, Hk, K), jnp.float32) * 0.39 - 1.0
    b = jax.random.normal(k4, (B, L, Hk, K), jnp.float32) * 0.39
    w = jax.random.normal(k5, (B, L, Hv, V), jnp.float32) * 0.39

    o_ref = token_serial_gdn2(q, k, v, g, b, w, scale=0.2, fuse_prologue=True, qk_l2norm=True)
    o_pallas = chunk_gdn2(q, k, v, g, b, w, scale=0.2, fuse_prologue=True, qk_l2norm=True, precision="fp32")
    err = rel_l2(o_pallas, o_ref)
    assert err < 1e-5, f"scale=0.2 fused path error {err:.2e} exceeds 1e-5"


def test_fuse_prologue_requires_qk_l2norm():
    """Verify ValueError is raised when fuse_prologue=True and qk_l2norm=False."""
    B, L, Hk, Hv, K, V = 1, 128, 2, 4, 128, 128
    key = jax.random.PRNGKey(444)
    q, k, v, g, b, w = make_test_inputs(key, B, L, Hk, Hv, K, V)
    with pytest.raises(ValueError, match="fuse_prologue=True requires qk_l2norm=True"):
        chunk_gdn2(q, k, v, g, b, w, fuse_prologue=True, qk_l2norm=False)


def test_parity_scale_as_array():
    """Verify scale passed as a JAX scalar array on both default and fused paths."""
    B, L, Hk, Hv, K, V = 1, 128, 2, 4, 128, 128
    key = jax.random.PRNGKey(555)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    g = jax.random.normal(k3, (B, L, Hk, K), jnp.float32) * 0.39 - 1.0
    b = jax.random.normal(k4, (B, L, Hk, K), jnp.float32) * 0.39
    w = jax.random.normal(k5, (B, L, Hv, V), jnp.float32) * 0.39

    scale_arr = jnp.float32(0.2)

    # 1. Default path (fuse_prologue=False)
    o_default_arr = chunk_gdn2(q, k, v, g, b, w, scale=scale_arr, fuse_prologue=False, precision="fp32")
    o_default_flt = chunk_gdn2(q, k, v, g, b, w, scale=0.2, fuse_prologue=False, precision="fp32")
    assert rel_l2(o_default_arr, o_default_flt) == 0.0, "default path scale=jnp.float32(0.2) must match float scale"

    # 2. Fused path (fuse_prologue=True)
    o_fused_arr = chunk_gdn2(q, k, v, g, b, w, scale=scale_arr, fuse_prologue=True, qk_l2norm=True, precision="fp32")
    o_fused_flt = chunk_gdn2(q, k, v, g, b, w, scale=0.2, fuse_prologue=True, qk_l2norm=True, precision="fp32")
    assert rel_l2(o_fused_arr, o_fused_flt) == 0.0, "fused path scale=jnp.float32(0.2) must match float scale"

    # 3. Tracer raises clear ValueError
    with pytest.raises(ValueError, match="scale must be a concrete Python float or scalar, not a JAX tracer"):
        @jax.jit
        def f(s):
            return chunk_gdn2(q, k, v, g, b, w, scale=s)
        f(0.2)


