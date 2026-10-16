"""bf16 backward without dT truncation, and the same-document decay mask (mask_lambda_by_doc)."""
import jax
import jax.numpy as jnp
import numpy as np

from deltanet_tpu.gdn2 import (
    _single_chunk_prepass_bwd_analytical,
)


def _token_scan_with_seg_reference(q, k, v, g, b, w, seg):
    """Pure-JAX token-by-token reference with document resets: shapes (L, K/V)."""
    L, K = q.shape
    V = v.shape[-1]
    g_np = jnp.where(g <= 0.0, g, 0.0)

    def step(carry, xs):
        S, prev_s = carry
        q_t, k_t, v_t, g_t, b_t, w_t, s_t = xs
        S = jnp.where(s_t != prev_s, jnp.zeros_like(S), S)
        S_dec = S * jnp.exp(g_t)[:, None]
        beta_t = b_t * k_t
        err = jnp.sum(beta_t[:, None] * S_dec, axis=0)
        v_new = w_t * v_t - err
        S_next = S_dec + k_t[:, None] * v_new[None, :]
        o_t = jnp.sum(S_next * q_t[:, None], axis=0)
        return (S_next, s_t), o_t

    _, out = jax.lax.scan(
        step,
        (jnp.zeros((K, V), dtype=jnp.float32), jnp.int32(-1)),
        (q, k, v, g_np, b, w, seg),
    )
    return out


def test_bf16_bwd_dT_no_truncation_improves_gradient_accuracy():
    """Verify removing .astype(bfloat16) on dT_W/dT_U/d_beta_exp_masked/d_wv reduces bf16 backward error."""
    key = jax.random.PRNGKey(404)
    ks = jax.random.split(key, 12)
    C, K, V = 128, 128, 128

    q_c = jax.random.normal(ks[0], (C, K), dtype=jnp.float32) * 0.2
    k_c = jax.random.normal(ks[1], (C, K), dtype=jnp.float32) * 0.2
    v_c = jax.random.normal(ks[2], (C, V), dtype=jnp.float32) * 0.2
    g_c = -jax.nn.softplus(jax.random.normal(ks[3], (C, K), dtype=jnp.float32)) * 0.05
    b_c = jax.nn.sigmoid(jax.random.normal(ks[4], (C, K), dtype=jnp.float32))
    w_c = jax.nn.sigmoid(jax.random.normal(ks[5], (C, V), dtype=jnp.float32))
    seg_c = jnp.zeros((C,), dtype=jnp.int32)

    dW = jax.random.normal(ks[6], (C, K), dtype=jnp.float32) * 0.1
    dU = jax.random.normal(ks[7], (C, V), dtype=jnp.float32) * 0.1
    dM = jax.random.normal(ks[8], (C, C), dtype=jnp.float32) * 0.1
    dqe = jax.random.normal(ks[9], (C, K), dtype=jnp.float32) * 0.1
    dke = jax.random.normal(ks[10], (C, K), dtype=jnp.float32) * 0.1
    ddecay = jax.random.normal(ks[11], (K,), dtype=jnp.float32) * 0.1

    fp32_grads = _single_chunk_prepass_bwd_analytical(
        q_c, k_c, v_c, g_c, b_c, w_c, seg_c, dW, dU, dM, dqe, dke, ddecay,
        prec=jax.lax.Precision.HIGHEST,
    )
    bf16_fixed_grads = _single_chunk_prepass_bwd_analytical(
        q_c, k_c, v_c, g_c, b_c, w_c, seg_c, dW, dU, dM, dqe, dke, ddecay,
        prec=jax.lax.Precision.DEFAULT, truncate_bwd_dT_bf16=False,
    )
    bf16_legacy_grads = _single_chunk_prepass_bwd_analytical(
        q_c, k_c, v_c, g_c, b_c, w_c, seg_c, dW, dU, dM, dqe, dke, ddecay,
        prec=jax.lax.Precision.DEFAULT, truncate_bwd_dT_bf16=True,
    )

    # Check dv and dw (directly computed from d_wv) and db (from dA and d_beta_exp_masked) have strictly smaller L2 error vs FP32
    for idx in (2, 4, 5):  # dv, db, dw
        err_fixed = jnp.linalg.norm(bf16_fixed_grads[idx] - fp32_grads[idx])
        err_legacy = jnp.linalg.norm(bf16_legacy_grads[idx] - fp32_grads[idx])
        assert err_fixed < err_legacy


def test_mask_lambda_by_doc_prevents_mantissa_loss_after_strong_decay():
    """When doc 0 has huge negative decay (-500 per token) and doc 1 has tiny decay (-1e-4), mask_lambda_by_doc preserves doc 1."""
    C, K, V = 128, 128, 128
    key = jax.random.PRNGKey(405)
    ks = jax.random.split(key, 6)

    q_c = jax.random.normal(ks[0], (C, K), dtype=jnp.float32) * 0.2
    k_c = jax.random.normal(ks[1], (C, K), dtype=jnp.float32) * 0.2
    v_c = jax.random.normal(ks[2], (C, V), dtype=jnp.float32) * 0.2
    # Doc 0 (tokens 0..63) has huge decay -2000.0; Doc 1 (tokens 64..127) has tiny decay -1e-4
    seg_c = jnp.where(jnp.arange(C) < 64, 0, 1).astype(jnp.int32)
    g_c = jnp.where(seg_c[:, None] == 0, -2000.0, -1e-4) * jnp.ones((C, K), dtype=jnp.float32)
    b_c = jax.nn.sigmoid(jax.random.normal(ks[3], (C, K), dtype=jnp.float32))
    w_c = jax.nn.sigmoid(jax.random.normal(ks[4], (C, V), dtype=jnp.float32))

    dW = jax.random.normal(ks[5], (C, K), dtype=jnp.float32) * 0.1
    dU = jnp.zeros((C, V), dtype=jnp.float32)
    dM = jnp.ones((C, C), dtype=jnp.float32) * 0.1
    dqe = jnp.zeros((C, K), dtype=jnp.float32)
    dke = jnp.zeros((C, K), dtype=jnp.float32)
    ddecay = jnp.zeros((K,), dtype=jnp.float32)

    # Reference: Doc 0 also has 0.0 decay so it cannot corrupt Doc 1's prefix sum
    g_c_clean = jnp.where(seg_c[:, None] == 0, 0.0, -1e-4) * jnp.ones((C, K), dtype=jnp.float32)
    ref_grads = _single_chunk_prepass_bwd_analytical(
        q_c, k_c, v_c, g_c_clean, b_c, w_c, seg_c, dW, dU, dM, dqe, dke, ddecay,
        prec=jax.lax.Precision.HIGHEST, mask_lambda_by_doc=True,
    )
    masked_grads = _single_chunk_prepass_bwd_analytical(
        q_c, k_c, v_c, g_c, b_c, w_c, seg_c, dW, dU, dM, dqe, dke, ddecay,
        prec=jax.lax.Precision.HIGHEST, mask_lambda_by_doc=True,
    )
    # Doc 1 gradients (tokens 64..127) must be 100% bit-identical regardless of Doc 0's decay magnitude
    for g_m, g_r in zip(masked_grads, ref_grads):
        np.testing.assert_array_equal(np.asarray(g_m[64:]), np.asarray(g_r[64:]))
