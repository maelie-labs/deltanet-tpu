"""Tests for specialized delta-rule and linear attention variants on TPU.

Verifies that chunk_gdn, chunk_kda, chunk_gla, and chunk_delta_rule match their
respective reference implementations in forward output, backward gradients,
inference prefill mode, and single-token decode steps.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from deltanet_tpu.gdn2 import chunk_gdn2, gdn2_decode_step
from deltanet_tpu.variants import (
    chunk_delta_rule,
    chunk_gdn,
    chunk_gla,
    chunk_kda,
    delta_rule_decode_step,
    gdn_decode_step,
    gla_decode_step,
    kda_decode_step,
)
from deltanet_tpu.reference import (
    reference_delta_rule,
    reference_gdn,
    reference_gla,
    reference_kda,
    token_serial_gdn2,
)


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


@pytest.mark.parametrize("Hv,Hk", [(2, 2), (4, 2)])
def test_chunk_gdn_parity(Hv, Hk):
    """Verify Gated DeltaNet (scalar decay per head, shared gate) matches reference."""
    B, L, K, V = 1, 256, 128, 128
    chunk_size = 128
    key = jax.random.PRNGKey(701)
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    g = -jnp.exp(jax.random.normal(k3, (B, L, Hk, 1), jnp.float32) * 0.39 - 1.0)
    beta = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hk, 1), jnp.float32) * 0.39)
    do = jax.random.normal(jax.random.PRNGKey(702), (B, L, Hv, V), jnp.float32)

    def run_ref(q, k, v, g, beta):
        return reference_gdn(q, k, v, g, beta)

    def run_kernel(q, k, v, g, beta):
        return chunk_gdn(q, k, v, g, beta, chunk_size=chunk_size, precision="fp32")

    o_ref, vjp_ref = jax.vjp(run_ref, q, k, v, g, beta)
    o_kernel, vjp_kernel = jax.vjp(run_kernel, q, k, v, g, beta)

    err_o = rel_l2(o_kernel, o_ref)
    assert err_o < 1e-5, f"GDN forward rel L2 error {err_o:.2e} exceeds 1e-5"

    grads_ref = vjp_ref(do)
    grads_kernel = vjp_kernel(do)

    for name, g_k, g_r in zip(["dq", "dk", "dv", "dg", "dbeta"], grads_kernel, grads_ref):
        err_g = rel_l2(g_k, g_r)
        assert err_g < 1e-5, f"GDN {name} rel L2 error {err_g:.2e} exceeds 1e-5"


def test_chunk_gdn_qwen3_5_4b_layout():
    """Verify Gated DeltaNet with Qwen3.5-4B layout (gates per value head Hv > Hk)."""
    B, L, Hk, Hv, K, V = 1, 128, 2, 4, 128, 128
    chunk_size = 128
    key = jax.random.PRNGKey(705)
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    # Decay and beta generated per value head (Hv=4)
    g = -jnp.exp(jax.random.normal(k3, (B, L, Hv, 1), jnp.float32) * 0.39 - 1.0)
    beta = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hv, 1), jnp.float32) * 0.39)
    do = jax.random.normal(jax.random.PRNGKey(706), (B, L, Hv, V), jnp.float32)

    def run_ref(q, k, v, g, beta):
        return reference_gdn(q, k, v, g, beta)

    def run_kernel(q, k, v, g, beta):
        return chunk_gdn(q, k, v, g, beta, chunk_size=chunk_size, precision="fp32")

    o_ref, vjp_ref = jax.vjp(run_ref, q, k, v, g, beta)
    o_kernel, vjp_kernel = jax.vjp(run_kernel, q, k, v, g, beta)

    err_o = rel_l2(o_kernel, o_ref)
    assert err_o < 1e-5, f"Qwen3.5-4B layout forward rel L2 error {err_o:.2e} exceeds 1e-5"

    grads_ref = vjp_ref(do)
    grads_kernel = vjp_kernel(do)

    for name, g_k, g_r in zip(["dq", "dk", "dv", "dg", "dbeta"], grads_kernel, grads_ref):
        err_g = rel_l2(g_k, g_r)
        assert err_g < 1e-5, f"Qwen3.5-4B layout {name} rel L2 error {err_g:.2e} exceeds 1e-5"

    # Also verify decode step with Hv-head gates
    state = jnp.zeros((B, Hv, K, V), dtype=jnp.float32)
    q_tok = q[:, 0]
    k_tok = k[:, 0]
    v_tok = v[:, 0]
    g_tok = g[:, 0]
    beta_tok = beta[:, 0]
    out_dec, state_dec = gdn_decode_step(q_tok, k_tok, v_tok, g_tok, beta_tok, state)
    assert out_dec.shape == (B, Hv, V)
    assert state_dec.shape == (B, Hv, K, V)

    out_ref_dec, state_ref_dec = reference_gdn(
        q_tok[:, None], k_tok[:, None], v_tok[:, None], g_tok[:, None], beta_tok[:, None],
        initial_state=state, output_final_state=True
    )
    assert rel_l2(out_dec[:, None], out_ref_dec) < 1e-5
    assert rel_l2(state_dec, state_ref_dec) < 1e-5


def test_chunk_gdn2_hv_decay_hk_erase():
    """Verify chunk_gdn2 with g at Hv heads and b at Hk heads (per-value-head decay, per-key-head erase)."""
    B, L, Hk, Hv, K, V = 1, 128, 2, 4, 128, 128
    chunk_size = 128
    key = jax.random.PRNGKey(710)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    g = -jnp.exp(jax.random.normal(k3, (B, L, Hv, 1), jnp.float32) * 0.39 - 1.0)
    b = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hk, K), jnp.float32) * 0.39)
    w = jax.nn.sigmoid(jax.random.normal(k5, (B, L, Hv, V), jnp.float32) * 0.39)
    do = jax.random.normal(jax.random.PRNGKey(711), (B, L, Hv, V), jnp.float32)

    # Reference computation: repeats q, k, b to Hv heads
    repeat_factor = Hv // Hk
    q_rep = jnp.repeat(q, repeat_factor, axis=2)
    k_rep = jnp.repeat(k, repeat_factor, axis=2)
    b_rep = jnp.repeat(b, repeat_factor, axis=2)

    def run_ref(q_r, k_r, v, g, b_r, w):
        return token_serial_gdn2(q_r, k_r, v, g, b_r, w)

    def run_kernel(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, chunk_size=chunk_size, precision="fp32")

    o_ref, vjp_ref = jax.vjp(run_ref, q_rep, k_rep, v, g, b_rep, w)
    o_kernel, vjp_kernel = jax.vjp(run_kernel, q, k, v, g, b, w)

    err_o = rel_l2(o_kernel, o_ref)
    assert err_o < 1e-5, f"Forward rel L2 error {err_o:.2e} exceeds 1e-5"

    grads_ref = vjp_ref(do)
    grads_kernel = vjp_kernel(do)

    # Fold reference gradients across repeated key-heads
    dq_ref_folded = grads_ref[0].reshape(B, L, Hk, repeat_factor, K).sum(axis=3)
    dk_ref_folded = grads_ref[1].reshape(B, L, Hk, repeat_factor, K).sum(axis=3)
    db_ref_folded = grads_ref[4].reshape(B, L, Hk, repeat_factor, K).sum(axis=3)
    grads_ref_folded = (dq_ref_folded, dk_ref_folded, grads_ref[2], grads_ref[3], db_ref_folded, grads_ref[5])

    for name, g_k, g_r in zip(["dq", "dk", "dv", "dg", "db", "dw"], grads_kernel, grads_ref_folded):
        err_g = rel_l2(g_k, g_r)
        assert err_g < 1e-5, f"GDN2 {name} rel L2 error {err_g:.2e} exceeds 1e-5"

    # Decode step test and value check vs reference
    state = jnp.zeros((B, Hv, K, V), dtype=jnp.float32)
    q_tok, k_tok, v_tok = q[:, 0], k[:, 0], v[:, 0]
    g_tok, b_tok, w_tok = g[:, 0], b[:, 0], w[:, 0]
    out_dec, state_dec = gdn2_decode_step(q_tok, k_tok, v_tok, g_tok, b_tok, w_tok, state)
    assert out_dec.shape == (B, Hv, V)
    assert state_dec.shape == (B, Hv, K, V)

    # Reference decode step via token_serial_gdn2 at L=1
    q_tok_rep = jnp.repeat(q_tok[:, None], repeat_factor, axis=2)
    k_tok_rep = jnp.repeat(k_tok[:, None], repeat_factor, axis=2)
    b_tok_rep = jnp.repeat(b_tok[:, None], repeat_factor, axis=2)
    out_ref_dec, state_ref_dec = token_serial_gdn2(
        q_tok_rep, k_tok_rep, v_tok[:, None], g_tok[:, None], b_tok_rep, w_tok[:, None],
        initial_state=state, output_final_state=True
    )
    assert rel_l2(out_dec[:, None], out_ref_dec) < 1e-5
    assert rel_l2(state_dec, state_ref_dec) < 1e-5


def test_chunk_gdn_hk_decay_hv_beta():
    """Verify chunk_gdn with g at Hk heads and beta at Hv heads."""
    B, L, Hk, Hv, K, V = 1, 128, 2, 4, 128, 128
    chunk_size = 128
    key = jax.random.PRNGKey(712)
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    g = -jnp.exp(jax.random.normal(k3, (B, L, Hk, 1), jnp.float32) * 0.39 - 1.0)
    beta = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hv, 1), jnp.float32) * 0.39)
    do = jax.random.normal(jax.random.PRNGKey(713), (B, L, Hv, V), jnp.float32)

    def run_ref(q, k, v, g, beta):
        return reference_gdn(q, k, v, g, beta)

    def run_kernel(q, k, v, g, beta):
        return chunk_gdn(q, k, v, g, beta, chunk_size=chunk_size, precision="fp32")

    o_ref, vjp_ref = jax.vjp(run_ref, q, k, v, g, beta)
    o_kernel, vjp_kernel = jax.vjp(run_kernel, q, k, v, g, beta)

    err_o = rel_l2(o_kernel, o_ref)
    assert err_o < 1e-5, f"Forward rel L2 error {err_o:.2e} exceeds 1e-5"

    grads_ref = vjp_ref(do)
    grads_kernel = vjp_kernel(do)

    assert o_kernel.shape == (B, L, Hv, V)
    for name, g_k, g_r in zip(["dq", "dk", "dv", "dg", "dbeta"], grads_kernel, grads_ref):
        err_g = rel_l2(g_k, g_r)
        assert err_g < 1e-5, f"GDN {name} rel L2 error {err_g:.2e} exceeds 1e-5"

    # Decode step test and value check vs reference
    state = jnp.zeros((B, Hv, K, V), dtype=jnp.float32)
    q_tok, k_tok, v_tok = q[:, 0], k[:, 0], v[:, 0]
    g_tok, beta_tok = g[:, 0], beta[:, 0]
    out_dec, state_dec = gdn_decode_step(q_tok, k_tok, v_tok, g_tok, beta_tok, state)
    assert out_dec.shape == (B, Hv, V)
    assert state_dec.shape == (B, Hv, K, V)

    out_ref_dec, state_ref_dec = reference_gdn(
        q_tok[:, None], k_tok[:, None], v_tok[:, None], g_tok[:, None], beta_tok[:, None],
        initial_state=state, output_final_state=True
    )
    assert rel_l2(out_dec[:, None], out_ref_dec) < 1e-5
    assert rel_l2(state_dec, state_ref_dec) < 1e-5


@pytest.mark.parametrize("Hv,Hk", [(2, 2), (4, 2)])
def test_chunk_kda_parity(Hv, Hk):
    """Verify Kimi Delta Attention (channel-wise decay, shared gate) matches reference."""
    B, L, K, V = 1, 256, 128, 128
    chunk_size = 128
    key = jax.random.PRNGKey(801)
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    g = -jnp.exp(jax.random.normal(k3, (B, L, Hk, K), jnp.float32) * 0.39 - 1.0)
    beta = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hk, 1), jnp.float32) * 0.39)
    do = jax.random.normal(jax.random.PRNGKey(802), (B, L, Hv, V), jnp.float32)

    def run_ref(q, k, v, g, beta):
        return reference_kda(q, k, v, g, beta)

    def run_kernel(q, k, v, g, beta):
        return chunk_kda(q, k, v, g, beta, chunk_size=chunk_size, precision="fp32")

    o_ref, vjp_ref = jax.vjp(run_ref, q, k, v, g, beta)
    o_kernel, vjp_kernel = jax.vjp(run_kernel, q, k, v, g, beta)

    err_o = rel_l2(o_kernel, o_ref)
    assert err_o < 1e-5, f"KDA forward rel L2 error {err_o:.2e} exceeds 1e-5"

    grads_ref = vjp_ref(do)
    grads_kernel = vjp_kernel(do)

    for name, g_k, g_r in zip(["dq", "dk", "dv", "dg", "dbeta"], grads_kernel, grads_ref):
        err_g = rel_l2(g_k, g_r)
        assert err_g < 1e-5, f"KDA {name} rel L2 error {err_g:.2e} exceeds 1e-5"


@pytest.mark.parametrize("Hv,Hk", [(2, 2), (4, 2)])
def test_chunk_gla_parity(Hv, Hk):
    """Verify Gated Linear Attention (decay, zero erase gate) matches reference."""
    B, L, K, V = 1, 256, 128, 128
    chunk_size = 128
    key = jax.random.PRNGKey(901)
    k0, k1, k2, k3 = jax.random.split(key, 4)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    g = -jnp.exp(jax.random.normal(k3, (B, L, Hk, K), jnp.float32) * 0.39 - 1.0)
    do = jax.random.normal(jax.random.PRNGKey(902), (B, L, Hv, V), jnp.float32)

    def run_ref(q, k, v, g):
        return reference_gla(q, k, v, g)

    def run_kernel(q, k, v, g):
        return chunk_gla(q, k, v, g, chunk_size=chunk_size, precision="fp32")

    o_ref, vjp_ref = jax.vjp(run_ref, q, k, v, g)
    o_kernel, vjp_kernel = jax.vjp(run_kernel, q, k, v, g)

    err_o = rel_l2(o_kernel, o_ref)
    assert err_o < 1e-5, f"GLA forward rel L2 error {err_o:.2e} exceeds 1e-5"

    grads_ref = vjp_ref(do)
    grads_kernel = vjp_kernel(do)

    for name, g_k, g_r in zip(["dq", "dk", "dv", "dg"], grads_kernel, grads_ref):
        err_g = rel_l2(g_k, g_r)
        assert err_g < 1e-5, f"GLA {name} rel L2 error {err_g:.2e} exceeds 1e-5"


@pytest.mark.parametrize("Hv,Hk", [(2, 2), (4, 2)])
def test_chunk_delta_rule_parity(Hv, Hk):
    """Verify DeltaNet (plain delta rule, no decay) matches reference."""
    B, L, K, V = 1, 256, 128, 128
    chunk_size = 128
    key = jax.random.PRNGKey(951)
    k0, k1, k2, k3 = jax.random.split(key, 4)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    beta = jax.nn.sigmoid(jax.random.normal(k3, (B, L, Hk, 1), jnp.float32) * 0.39)
    do = jax.random.normal(jax.random.PRNGKey(952), (B, L, Hv, V), jnp.float32)

    def run_ref(q, k, v, beta):
        return reference_delta_rule(q, k, v, beta)

    def run_kernel(q, k, v, beta):
        return chunk_delta_rule(q, k, v, beta, chunk_size=chunk_size, precision="fp32")

    o_ref, vjp_ref = jax.vjp(run_ref, q, k, v, beta)
    o_kernel, vjp_kernel = jax.vjp(run_kernel, q, k, v, beta)

    err_o = rel_l2(o_kernel, o_ref)
    assert err_o < 1e-5, f"DeltaRule forward rel L2 error {err_o:.2e} exceeds 1e-5"

    grads_ref = vjp_ref(do)
    grads_kernel = vjp_kernel(do)

    for name, g_k, g_r in zip(["dq", "dk", "dv", "dbeta"], grads_kernel, grads_ref):
        err_g = rel_l2(g_k, g_r)
        assert err_g < 1e-5, f"DeltaRule {name} rel L2 error {err_g:.2e} exceeds 1e-5"


def test_variants_inference_prefill():
    """Verify is_inference=True returns identical outputs across all variants."""
    B, L, Hk, Hv, K, V = 1, 128, 2, 4, 128, 128
    key = jax.random.PRNGKey(960)
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)
    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    g = -jnp.exp(jax.random.normal(k3, (B, L, Hk, K), jnp.float32) - 1.0)
    beta = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hk, 1), jnp.float32))

    # GDN
    o_train = chunk_gdn(q, k, v, g[:, :, :, :1], beta, is_inference=False)
    o_inf = chunk_gdn(q, k, v, g[:, :, :, :1], beta, is_inference=True)
    assert rel_l2(o_inf, o_train) < 1e-5

    # KDA
    o_train = chunk_kda(q, k, v, g, beta, is_inference=False)
    o_inf = chunk_kda(q, k, v, g, beta, is_inference=True)
    assert rel_l2(o_inf, o_train) < 1e-5

    # GLA
    o_train = chunk_gla(q, k, v, g, is_inference=False)
    o_inf = chunk_gla(q, k, v, g, is_inference=True)
    assert rel_l2(o_inf, o_train) < 1e-5

    # DeltaRule
    o_train = chunk_delta_rule(q, k, v, beta, is_inference=False)
    o_inf = chunk_delta_rule(q, k, v, beta, is_inference=True)
    assert rel_l2(o_inf, o_train) < 1e-5


def test_variants_decode_steps():
    """Verify decode steps advance state and produce output matching chunk execution."""
    B, Hk, Hv, K, V = 1, 2, 4, 128, 128
    key = jax.random.PRNGKey(970)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    q = jax.random.normal(k0, (B, Hk, K), jnp.float32)
    k = jax.random.normal(k1, (B, Hk, K), jnp.float32)
    v = jax.random.normal(k2, (B, Hv, V), jnp.float32)
    g = -jnp.exp(jax.random.normal(k3, (B, Hk, K), jnp.float32) - 1.0)
    beta = jax.nn.sigmoid(jax.random.normal(k4, (B, Hk, 1), jnp.float32))
    state = jax.random.normal(k5, (B, Hv, K, V), jnp.float32) * 0.05

    # Test decode step calls execute cleanly and match single-token chunk execution
    o_gdn, s_gdn = gdn_decode_step(q, k, v, g[:, :, :1], beta, state)
    assert o_gdn.shape == (B, Hv, V) and s_gdn.shape == (B, Hv, K, V)
    o_gdn_c, s_gdn_c = chunk_gdn(
        q[:, None], k[:, None], v[:, None], g[:, None, :, :1], beta[:, None],
        initial_state=state, output_final_state=True, precision="fp32"
    )
    assert rel_l2(o_gdn, o_gdn_c[:, 0]) < 1e-5
    assert rel_l2(s_gdn, s_gdn_c) < 1e-5

    o_kda, s_kda = kda_decode_step(q, k, v, g, beta, state)
    assert o_kda.shape == (B, Hv, V) and s_kda.shape == (B, Hv, K, V)
    o_kda_c, s_kda_c = chunk_kda(
        q[:, None], k[:, None], v[:, None], g[:, None], beta[:, None],
        initial_state=state, output_final_state=True, precision="fp32"
    )
    assert rel_l2(o_kda, o_kda_c[:, 0]) < 1e-5
    assert rel_l2(s_kda, s_kda_c) < 1e-5

    o_gla, s_gla = gla_decode_step(q, k, v, g, state)
    assert o_gla.shape == (B, Hv, V) and s_gla.shape == (B, Hv, K, V)
    o_gla_c, s_gla_c = chunk_gla(
        q[:, None], k[:, None], v[:, None], g[:, None],
        initial_state=state, output_final_state=True, precision="fp32"
    )
    assert rel_l2(o_gla, o_gla_c[:, 0]) < 1e-5
    assert rel_l2(s_gla, s_gla_c) < 1e-5

    o_dr, s_dr = delta_rule_decode_step(q, k, v, beta, state)
    assert o_dr.shape == (B, Hv, V) and s_dr.shape == (B, Hv, K, V)
    o_dr_c, s_dr_c = chunk_delta_rule(
        q[:, None], k[:, None], v[:, None], beta[:, None],
        initial_state=state, output_final_state=True, precision="fp32"
    )
    assert rel_l2(o_dr, o_dr_c[:, 0]) < 1e-5
    assert rel_l2(s_dr, s_dr_c) < 1e-5


@pytest.mark.parametrize("Hk,Hv", [(1, 2), (1, 1)])
def test_decode_step_hk1(Hk, Hv):
    """Verify all decode steps with single key head Hk=1 match chunk execution."""
    B, K, V = 1, 128, 128
    key = jax.random.PRNGKey(980)
    k0, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 8)
    q = jax.random.normal(k0, (B, Hk, K), jnp.float32)
    k = jax.random.normal(k1, (B, Hk, K), jnp.float32)
    v = jax.random.normal(k2, (B, Hv, V), jnp.float32)
    g = -jnp.exp(jax.random.normal(k3, (B, Hk, K), jnp.float32) - 1.0)
    b = jax.nn.sigmoid(jax.random.normal(k4, (B, Hk, K), jnp.float32) * 0.39)
    w = jax.nn.sigmoid(jax.random.normal(k5, (B, Hv, V), jnp.float32) * 0.39)
    beta = jax.nn.sigmoid(jax.random.normal(k6, (B, Hk, 1), jnp.float32) * 0.39)
    state = jax.random.normal(k7, (B, Hv, K, V), jnp.float32) * 0.05

    # GDN-2
    o_gdn2, s_gdn2 = gdn2_decode_step(q, k, v, g, b, w, state)
    o_gdn2_c, s_gdn2_c = chunk_gdn2(
        q[:, None], k[:, None], v[:, None], g[:, None], b[:, None], w[:, None],
        initial_state=state, output_final_state=True, precision="fp32"
    )
    assert rel_l2(o_gdn2, o_gdn2_c[:, 0]) < 1e-5
    assert rel_l2(s_gdn2, s_gdn2_c) < 1e-5

    # GDN
    g_scalar = g[:, :, :1]
    o_gdn, s_gdn = gdn_decode_step(q, k, v, g_scalar, beta, state)
    o_gdn_c, s_gdn_c = chunk_gdn(
        q[:, None], k[:, None], v[:, None], g_scalar[:, None], beta[:, None],
        initial_state=state, output_final_state=True, precision="fp32"
    )
    assert rel_l2(o_gdn, o_gdn_c[:, 0]) < 1e-5
    assert rel_l2(s_gdn, s_gdn_c) < 1e-5

    # KDA
    o_kda, s_kda = kda_decode_step(q, k, v, g, beta, state)
    o_kda_c, s_kda_c = chunk_kda(
        q[:, None], k[:, None], v[:, None], g[:, None], beta[:, None],
        initial_state=state, output_final_state=True, precision="fp32"
    )
    assert rel_l2(o_kda, o_kda_c[:, 0]) < 1e-5
    assert rel_l2(s_kda, s_kda_c) < 1e-5

    # GLA
    o_gla, s_gla = gla_decode_step(q, k, v, g, state)
    o_gla_c, s_gla_c = chunk_gla(
        q[:, None], k[:, None], v[:, None], g[:, None],
        initial_state=state, output_final_state=True, precision="fp32"
    )
    assert rel_l2(o_gla, o_gla_c[:, 0]) < 1e-5
    assert rel_l2(s_gla, s_gla_c) < 1e-5

    # DeltaRule
    o_dr, s_dr = delta_rule_decode_step(q, k, v, beta, state)
    o_dr_c, s_dr_c = chunk_delta_rule(
        q[:, None], k[:, None], v[:, None], beta[:, None],
        initial_state=state, output_final_state=True, precision="fp32"
    )
    assert rel_l2(o_dr, o_dr_c[:, 0]) < 1e-5
    assert rel_l2(s_dr, s_dr_c) < 1e-5


@pytest.mark.parametrize("target_dtype", ["bfloat16", "float32", jnp.bfloat16, jnp.float32])
def test_variant_decode_state_dtype(target_dtype):
    """Verify that all variant decode steps support the state_dtype parameter."""
    B, Hk, Hv, K, V = 1, 2, 4, 128, 128
    key = jax.random.PRNGKey(909)
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)

    q = jax.random.normal(k0, (B, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, Hv, V), jnp.float32) * 0.1
    g = -jnp.exp(jax.random.normal(k3, (B, Hk, K), jnp.float32) - 1.0)
    beta = jax.nn.sigmoid(jax.random.normal(k4, (B, Hk, 1), jnp.float32) * 0.39)
    state = jnp.zeros((B, Hv, K, V), dtype=jnp.float32)
    expected_dtype = jnp.dtype(target_dtype)

    # GDN
    _, s_gdn = gdn_decode_step(q, k, v, g[:, :, :1], beta, state, state_dtype=target_dtype)
    assert s_gdn.dtype == expected_dtype

    # KDA
    _, s_kda = kda_decode_step(q, k, v, g, beta, state, state_dtype=target_dtype)
    assert s_kda.dtype == expected_dtype

    # GLA
    _, s_gla = gla_decode_step(q, k, v, g, state, state_dtype=target_dtype)
    assert s_gla.dtype == expected_dtype

    # DeltaRule
    _, s_dr = delta_rule_decode_step(q, k, v, beta, state, state_dtype=target_dtype)
    assert s_dr.dtype == expected_dtype
