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

