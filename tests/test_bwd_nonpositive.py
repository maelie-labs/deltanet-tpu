"""Analytical prepass backward under the nonpositive clamp: matches autodiff and the token scan."""
import jax
import jax.numpy as jnp
import numpy as np

from deltanet_tpu.gdn2 import (
    _single_chunk_prepass_bwd_analytical,
    chunk_gdn2,
)


def _token_scan_reference(q, k, v, g, b, w, scale=None):
    """Pure-JAX token-by-token reference for 1 batch, 1 head: shapes (L, K/V)."""
    L, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(K)
    q_s = q * scale
    g_np = jnp.where(g <= 0.0, g, 0.0)

    def step(S, xs):
        q_t, k_t, v_t, g_t, b_t, w_t = xs
        S_dec = S * jnp.exp(g_t)[:, None]
        beta_t = b_t * k_t
        err = jnp.sum(beta_t[:, None] * S_dec, axis=0)
        v_new = w_t * v_t - err
        S_next = S_dec + k_t[:, None] * v_new[None, :]
        o_t = jnp.sum(S_next * q_t[:, None], axis=0)
        return S_next, o_t

    _, out = jax.lax.scan(step, jnp.zeros((K, V), dtype=jnp.float32), (q_s, k, v, g_np, b, w))
    return out


def _pure_prepass_fwd(q_c, k_c, v_c, g_c, b_c, w_c, seg_c, chunk_size=128, BC=32):
    """Pure-JAX mirror of _prepass_chunk_fwd_kernel for jax.vjp comparison."""
    NS = chunk_size // BC
    from deltanet_tpu.gdn2 import _nonpositive, _invert_unit_triangular_core
    tril_ones = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
    Lambda = jnp.matmul(tril_ones, g_c, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    beta = b_c * k_c

    iota_bc = jnp.arange(BC)
    strictly_lower_bc = (iota_bc[:, None] > iota_bc[None, :])
    lower_bc = (iota_bc[:, None] >= iota_bc[None, :])

    rows_A, rows_M = [], []
    for a in range(NS):
        sa, ea = a * BC, (a + 1) * BC
        G_ref = Lambda[sa:sa+1, :]
        exp_row = jnp.exp(_nonpositive(Lambda[sa:ea, :] - G_ref))
        b_a = beta[sa:ea, :] * exp_row
        q_a = q_c[sa:ea, :] * exp_row

        row_a_A, row_a_M = [], []
        for b_idx in range(a):
            sb, eb = b_idx * BC, (b_idx + 1) * BC
            exp_col_off = jnp.exp(_nonpositive(G_ref - Lambda[sb:eb, :]))
            k_b = k_c[sb:eb, :] * exp_col_off
            A_ab = jnp.matmul(b_a, k_b.T, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
            M_ab = jnp.matmul(q_a, k_b.T, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
            row_a_A.append(A_ab)
            row_a_M.append(M_ab)

        Lambda_a = Lambda[sa:ea, :]
        decay_diag = jnp.exp(_nonpositive(Lambda_a[:, None, :] - Lambda_a[None, :, :]))
        A_aa = jnp.where(strictly_lower_bc, jnp.sum((beta[sa:ea, None, :] * decay_diag) * k_c[None, sa:ea, :], axis=-1), 0.0)
        M_aa = jnp.where(lower_bc, jnp.sum((q_c[sa:ea, None, :] * decay_diag) * k_c[None, sa:ea, :], axis=-1), 0.0)
        row_a_A.append(A_aa)
        row_a_M.append(M_aa)
        if ea < chunk_size:
            pad_cols = chunk_size - ea
            row_a_A.append(jnp.zeros((BC, pad_cols), dtype=jnp.float32))
            row_a_M.append(jnp.zeros((BC, pad_cols), dtype=jnp.float32))
        rows_A.append(jnp.concatenate(row_a_A, axis=1))
        rows_M.append(jnp.concatenate(row_a_M, axis=1))

    A_raw = jnp.concatenate(rows_A, axis=0)
    M_raw = jnp.concatenate(rows_M, axis=0)
    same_doc_pair = (seg_c[:, None] == seg_c[None, :])
    iota = jnp.arange(chunk_size)
    strictly_causal = (iota[:, None] > iota[None, :]) & same_doc_pair
    causal = (iota[:, None] >= iota[None, :]) & same_doc_pair
    A = jnp.where(strictly_causal, A_raw, 0.0)
    M = jnp.where(causal, M_raw, 0.0)

    same_doc_as_start = (seg_c[:, None] == seg_c[0])
    beta_exp_masked = jnp.where(same_doc_as_start, beta * jnp.exp(_nonpositive(Lambda)), 0.0)
    q_exp_masked = jnp.where(same_doc_as_start, q_c * jnp.exp(_nonpositive(Lambda)), 0.0)

    same_doc_as_end = (seg_c[:, None] == seg_c[-1])
    Lambda_end = Lambda[-1:, :]
    k_to_end_masked = jnp.where(same_doc_as_end, k_c * jnp.exp(_nonpositive(Lambda_end - Lambda)), 0.0)

    T = _invert_unit_triangular_core(A, precision=jax.lax.Precision.HIGHEST)
    W = jnp.matmul(T, beta_exp_masked, precision=jax.lax.Precision.HIGHEST)
    U = jnp.matmul(T, w_c * v_c, precision=jax.lax.Precision.HIGHEST)
    return W, U, M, q_exp_masked, k_to_end_masked, Lambda_end[0, :]


def test_analytical_bwd_matches_autodiff_under_nonpositive_clamping(monkeypatch):
    """Verify _single_chunk_prepass_bwd_analytical matches jax.vjp when Lambda diffs > 0."""
    monkeypatch.setenv("DELTANET_NONPOSITIVE_GUARD", "1")

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 12)
    C, K, V = 128, 128, 128

    q_c = jax.random.normal(keys[0], (C, K), dtype=jnp.float32) * 0.2
    k_c = jax.random.normal(keys[1], (C, K), dtype=jnp.float32) * 0.2
    v_c = jax.random.normal(keys[2], (C, V), dtype=jnp.float32) * 0.2
    # Construct g_c with mixed signs / strong early decay so Lambda[sa:ea] - G_ref has positive entries
    g_c = jnp.where(jnp.arange(C)[:, None] < 16, -25.0, 0.05) * jnp.ones((C, K), dtype=jnp.float32)
    b_c = jax.nn.sigmoid(jax.random.normal(keys[3], (C, K), dtype=jnp.float32))
    w_c = jax.nn.sigmoid(jax.random.normal(keys[4], (C, V), dtype=jnp.float32))
    seg_c = jnp.zeros((C,), dtype=jnp.int32)

    dW = jax.random.normal(keys[5], (C, K), dtype=jnp.float32) * 0.1
    dU = jax.random.normal(keys[6], (C, V), dtype=jnp.float32) * 0.1
    dM = jax.random.normal(keys[7], (C, C), dtype=jnp.float32) * 0.1
    dqe = jax.random.normal(keys[8], (C, K), dtype=jnp.float32) * 0.1
    dke = jax.random.normal(keys[9], (C, K), dtype=jnp.float32) * 0.1
    ddecay = jax.random.normal(keys[10], (K,), dtype=jnp.float32) * 0.1

    _, vjp_fn = jax.vjp(
        lambda q, k, v, g, b, w: _pure_prepass_fwd(q, k, v, g, b, w, seg_c),
        q_c, k_c, v_c, g_c, b_c, w_c,
    )
    ref_grads = vjp_fn((dW, dU, dM, dqe, dke, ddecay))

    ana_grads = _single_chunk_prepass_bwd_analytical(
        q_c, k_c, v_c, g_c, b_c, w_c, seg_c,
        dW, dU, dM, dqe, dke, ddecay,
        chunk_size=C, BC=32, prec=jax.lax.Precision.HIGHEST, T=None,
    )

    for name, g_ref, g_ana in zip(["dq", "dk", "dv", "dg", "db", "dw"], ref_grads, ana_grads):
        assert jnp.all(jnp.isfinite(g_ana)), f"{name} contains NaN/Inf"
        np.testing.assert_allclose(np.asarray(g_ana), np.asarray(g_ref), rtol=1e-4, atol=1e-4, err_msg=f"Mismatch in {name}")


def test_end_to_end_chunk_gdn2_grad_matches_token_scan():
    key = jax.random.PRNGKey(7)
    ks = jax.random.split(key, 6)
    B, L, H, K, V = 1, 128, 1, 128, 128

    q = jax.random.normal(ks[0], (B, L, H, K), dtype=jnp.float32) * 0.1
    k = jax.random.normal(ks[1], (B, L, H, K), dtype=jnp.float32) * 0.1
    v = jax.random.normal(ks[2], (B, L, H, V), dtype=jnp.float32) * 0.1
    g = -jax.nn.softplus(jax.random.normal(ks[3], (B, L, H, K), dtype=jnp.float32))
    b = jax.nn.sigmoid(jax.random.normal(ks[4], (B, L, H, K), dtype=jnp.float32))
    w = jax.nn.sigmoid(jax.random.normal(ks[5], (B, L, H, V), dtype=jnp.float32))

    def loss_pallas(q, k, v, g, b, w):
        return jnp.sum(chunk_gdn2(q, k, v, g, b, w, precision="fp32"))

    def loss_ref(q, k, v, g, b, w):
        return jnp.sum(_token_scan_reference(q[0, :, 0], k[0, :, 0], v[0, :, 0], g[0, :, 0], b[0, :, 0], w[0, :, 0]))

    grads_pallas = jax.grad(loss_pallas, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, g, b, w)
    grads_ref = jax.grad(loss_ref, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, g, b, w)

    for g_p, g_r in zip(grads_pallas, grads_ref):
        np.testing.assert_allclose(np.asarray(g_p), np.asarray(g_r), rtol=1e-4, atol=1e-4)
