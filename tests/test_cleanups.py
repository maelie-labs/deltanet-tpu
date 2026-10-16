"""Compact segment-ID residuals and stability_guard with fuse_prologue (fused matches unfused)."""
import jax
import jax.numpy as jnp
import numpy as np

from deltanet_tpu.gdn2 import _chunked_recurrence_fwd, chunk_gdn2


def test_segs_padded_is_compact_1x128():
    """Verify segs_padded in res has shape (B, 1, NC, 1, 128) instead of (B, 1, NC, 128, 128)."""
    B, L, H, K, V = 2, 256, 2, 128, 128
    q = jnp.ones((B, L, H, K), dtype=jnp.float32) * 0.1
    k = jnp.ones((B, L, H, K), dtype=jnp.float32) * 0.1
    v = jnp.ones((B, L, H, V), dtype=jnp.float32) * 0.1
    g = jnp.full((B, L, H, K), -0.1, dtype=jnp.float32)
    b = jnp.full((B, L, H, K), 0.5, dtype=jnp.float32)
    w = jnp.full((B, L, H, V), 0.5, dtype=jnp.float32)
    seg_ids = jnp.zeros((B, L), dtype=jnp.int32)

    _, _, res = _chunked_recurrence_fwd(q, k, v, g, b, w, seg_ids, None, chunk_size=128, precision="fp32")
    segs_padded = res[6]
    assert segs_padded.shape == (B, 1, L // 128, 1, 128)


def test_stability_guard_with_fuse_prologue_matches_unfused():
    """Verify fuse_prologue=True + stability_guard=True clamps g_act <= -1e-6 and matches unfused prologue."""
    key = jax.random.PRNGKey(505)
    ks = jax.random.split(key, 6)
    B, L, H, K, V = 1, 128, 1, 128, 128

    q = jax.random.normal(ks[0], (B, L, H, K), dtype=jnp.float32)
    k = jax.random.normal(ks[1], (B, L, H, K), dtype=jnp.float32)
    v = jax.random.normal(ks[2], (B, L, H, V), dtype=jnp.float32)
    # Very negative pre-activation logits (-50.0) so -exp(g) underflows above -1e-6 without stability_guard
    g_raw = jnp.where(jnp.arange(L)[None, :, None, None] < 64, -50.0, -0.5) * jnp.ones((B, L, H, K), dtype=jnp.float32)
    b_raw = jax.random.normal(ks[4], (B, L, H, K), dtype=jnp.float32)
    w_raw = jax.random.normal(ks[5], (B, L, H, V), dtype=jnp.float32)

    def fused_loss(q, k, v, g_r, b_r, w_r):
        return jnp.sum(
            chunk_gdn2(
                q, k, v, g_r, b_r, w_r,
                qk_l2norm=True, fuse_prologue=True, stability_guard=True, precision="fp32",
            )
        )

    def unfused_loss(q, k, v, g_r, b_r, w_r):
        g_act = jnp.minimum(-jnp.exp(jnp.minimum(g_r, 5.0)), -1e-6)
        b_act = jax.nn.sigmoid(b_r)
        w_act = jax.nn.sigmoid(w_r)
        return jnp.sum(
            chunk_gdn2(
                q, k, v, g_act, b_act, w_act,
                qk_l2norm=True, fuse_prologue=False, stability_guard=True, precision="fp32",
            )
        )

    val_f, grads_f = jax.value_and_grad(fused_loss, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, g_raw, b_raw, w_raw)
    val_u, grads_u = jax.value_and_grad(unfused_loss, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, g_raw, b_raw, w_raw)

    np.testing.assert_allclose(np.asarray(val_f), np.asarray(val_u), rtol=1e-5, atol=1e-5)
    for gf, gu in zip(grads_f, grads_u):
        np.testing.assert_allclose(np.asarray(gf), np.asarray(gu), rtol=1e-4, atol=1e-4)
    # Specifically verify dg is 0 where g_raw == -50.0 (clamped at -1e-6)
    np.testing.assert_allclose(np.asarray(grads_f[3][:, :64]), 0.0, atol=0.0)
