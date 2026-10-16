"""recompute_states residuals, bf16 storage, and grouped-query gradients with value-head gates."""
import jax
import jax.numpy as jnp
import numpy as np

from deltanet_tpu.gdn2 import _chunked_recurrence_fwd, chunk_gdn2


def _token_scan_gqa_reference(q, k, v, g, b, w, scale=None):
    """Pure-JAX token-by-token GQA reference: q, k have Hk heads; v, g, b, w have Hv heads."""
    B, L, Hk, K = q.shape
    Hv, V = v.shape[2], v.shape[3]
    G = Hv // Hk
    if scale is None:
        scale = 1.0 / np.sqrt(K)
    q_rep = jnp.repeat(q.astype(jnp.float32), G, axis=2) * scale
    k_rep = jnp.repeat(k.astype(jnp.float32), G, axis=2)
    v_f = v.astype(jnp.float32)
    g_f = jnp.where(g.astype(jnp.float32) <= 0.0, g.astype(jnp.float32), 0.0)
    b_f = b.astype(jnp.float32)
    w_f = w.astype(jnp.float32)

    def step(S, xs):
        q_t, k_t, v_t, g_t, b_t, w_t = xs
        S_dec = S * jnp.exp(g_t)[:, :, :, None]
        beta_t = b_t * k_t
        err = jnp.sum(beta_t[:, :, :, None] * S_dec, axis=2)
        v_new = w_t * v_t - err
        S_next = S_dec + k_t[:, :, :, None] * v_new[:, :, None, :]
        o_t = jnp.sum(S_next * q_t[:, :, :, None], axis=2)
        return S_next, o_t

    init_S = jnp.zeros((B, Hv, K, V), dtype=jnp.float32)
    _, outs = jax.lax.scan(
        step,
        init_S,
        (
            jnp.swapaxes(q_rep, 0, 1),
            jnp.swapaxes(k_rep, 0, 1),
            jnp.swapaxes(v_f, 0, 1),
            jnp.swapaxes(g_f, 0, 1),
            jnp.swapaxes(b_f, 0, 1),
            jnp.swapaxes(w_f, 0, 1),
        ),
    )
    return jnp.swapaxes(outs, 0, 1)


def test_recompute_states_drops_prepass_residuals_and_preserves_bf16():
    """Prove recompute_states=True sets W, U, M, qe, ke, Lambda_end, all_states, T to None in res and keeps bf16."""
    B, L, Hk, Hv, K, V = 1, 256, 2, 4, 128, 128
    q = jnp.ones((B, L, Hk, K), dtype=jnp.bfloat16) * 0.1
    k = jnp.ones((B, L, Hk, K), dtype=jnp.bfloat16) * 0.1
    v = jnp.ones((B, L, Hv, V), dtype=jnp.bfloat16) * 0.1
    g = jnp.full((B, L, Hv, K), -0.05, dtype=jnp.bfloat16)
    b = jnp.full((B, L, Hv, K), 0.5, dtype=jnp.bfloat16)
    w = jnp.full((B, L, Hv, V), 0.5, dtype=jnp.bfloat16)
    seg_ids = jnp.zeros((B, L), dtype=jnp.int32)

    _, _, res = _chunked_recurrence_fwd(
        q, k, v, g, b, w, seg_ids, None,
        chunk_size=128, precision="bf16", recompute_states=True,
    )
    q_s, k_s, v_s, g_s, b_s, w_s, _, W, U, M, qe, ke, lend, _, all_states, _, T = res

    # 1. Prepass outputs and all_states are None in res
    assert all(x is None for x in (W, U, M, qe, ke, lend, all_states, T))
    # 2. Saved inputs stay in bfloat16 and q/k stay at Hk=2 heads (not repeated to Hv=4)
    assert q_s.dtype == jnp.bfloat16 and q_s.shape == (B, L, Hk, K)
    assert k_s.dtype == jnp.bfloat16 and k_s.shape == (B, L, Hk, K)
    assert v_s.dtype == jnp.bfloat16 and v_s.shape == (B, L, Hv, V)


def test_recompute_states_and_gqa_gradients_match_reference():
    """Verify forward and backward with recompute_states=True and Hk < Hv match token scan and recompute_states=False."""
    key = jax.random.PRNGKey(303)
    ks = jax.random.split(key, 6)
    B, L, Hk, Hv, K, V = 1, 256, 2, 4, 128, 128

    q = jax.random.normal(ks[0], (B, L, Hk, K), dtype=jnp.float32) * 0.2
    k = jax.random.normal(ks[1], (B, L, Hk, K), dtype=jnp.float32) * 0.2
    v = jax.random.normal(ks[2], (B, L, Hv, V), dtype=jnp.float32) * 0.2
    g = -jax.nn.softplus(jax.random.normal(ks[3], (B, L, Hv, K), dtype=jnp.float32)) * 0.1
    b = jax.nn.sigmoid(jax.random.normal(ks[4], (B, L, Hv, K), dtype=jnp.float32))
    w = jax.nn.sigmoid(jax.random.normal(ks[5], (B, L, Hv, V), dtype=jnp.float32))

    grads_no_remat = jax.grad(
        lambda q, k, v, g, b, w: jnp.sum(chunk_gdn2(q, k, v, g, b, w, precision="fp32", recompute_states=False)),
        argnums=(0, 1, 2, 3, 4, 5),
    )(q, k, v, g, b, w)

    grads_remat = jax.grad(
        lambda q, k, v, g, b, w: jnp.sum(chunk_gdn2(q, k, v, g, b, w, precision="fp32", recompute_states=True)),
        argnums=(0, 1, 2, 3, 4, 5),
    )(q, k, v, g, b, w)

    grads_ref = jax.grad(
        lambda q, k, v, g, b, w: jnp.sum(_token_scan_gqa_reference(q, k, v, g, b, w)),
        argnums=(0, 1, 2, 3, 4, 5),
    )(q, k, v, g, b, w)

    for gn, gr, gref in zip(grads_no_remat, grads_remat, grads_ref):
        np.testing.assert_allclose(np.asarray(gr), np.asarray(gn), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(gr), np.asarray(gref), rtol=1e-4, atol=1e-4)
