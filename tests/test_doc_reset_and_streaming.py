"""initial_state reset at a document boundary on token 0 (initial_state_segment_id) and streaming prefill parity."""
import jax
import jax.numpy as jnp
import numpy as np

from deltanet_tpu.gdn2 import chunk_gdn2


def test_initial_state_reset_at_chunk_0():
    """When initial_state is given and token 0 starts a new document, output and d_init_state equal zero-init."""
    key = jax.random.PRNGKey(101)
    ks = jax.random.split(key, 7)
    B, L, H, K, V = 2, 256, 2, 128, 128

    q = jax.random.normal(ks[0], (B, L, H, K), dtype=jnp.float32) * 0.2
    k = jax.random.normal(ks[1], (B, L, H, K), dtype=jnp.float32) * 0.2
    v = jax.random.normal(ks[2], (B, L, H, V), dtype=jnp.float32) * 0.2
    g = -jax.nn.softplus(jax.random.normal(ks[3], (B, L, H, K), dtype=jnp.float32)) * 0.1
    b = jax.nn.sigmoid(jax.random.normal(ks[4], (B, L, H, K), dtype=jnp.float32))
    w = jax.nn.sigmoid(jax.random.normal(ks[5], (B, L, H, V), dtype=jnp.float32))
    init_state = jax.random.normal(ks[6], (B, H, K, V), dtype=jnp.float32) * 5.0

    # Batch 0 and 1 start with segment ID 7 at token 0, while initial_state came from segment ID 3
    seg_ids = jnp.full((B, L), 7, dtype=jnp.int32)
    init_seg_id = jnp.array([3, 3], dtype=jnp.int32)

    out_with_reset, state_with_reset = chunk_gdn2(
        q, k, v, g, b, w,
        segment_ids=seg_ids,
        initial_state=init_state,
        initial_state_segment_id=init_seg_id,
        output_final_state=True,
        precision="fp32",
    )
    out_no_init, state_no_init = chunk_gdn2(
        q, k, v, g, b, w,
        segment_ids=seg_ids,
        initial_state=None,
        output_final_state=True,
        precision="fp32",
    )

    np.testing.assert_allclose(np.asarray(out_with_reset), np.asarray(out_no_init), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(state_with_reset), np.asarray(state_no_init), rtol=1e-6, atol=1e-6)

    # Gradient w.r.t. init_state must be identically zero when reset at token 0
    d_init = jax.grad(
        lambda s0: jnp.sum(
            chunk_gdn2(
                q, k, v, g, b, w,
                segment_ids=seg_ids,
                initial_state=s0,
                initial_state_segment_id=init_seg_id,
                precision="fp32",
            )
        )
    )(init_state)
    np.testing.assert_allclose(np.asarray(d_init), 0.0, atol=0.0)


def test_streaming_prefill_matches_single_pass_with_boundaries_and_padding():
    """Verify streaming_prefill matches single-pass prefill with 1D segment_ids, a non-multiple length, block-boundary resets, qk_l2norm and scale."""
    key = jax.random.PRNGKey(202)
    ks = jax.random.split(key, 6)
    B, L, Hk, Hv, K, V = 1, 300, 1, 2, 128, 128  # L=300 (non-multiple of 128), G=2

    q = jax.random.normal(ks[0], (B, L, Hk, K), dtype=jnp.float32)
    k = jax.random.normal(ks[1], (B, L, Hk, K), dtype=jnp.float32)
    v = jax.random.normal(ks[2], (B, L, Hv, V), dtype=jnp.float32)
    g = jax.random.normal(ks[3], (B, L, Hk, K), dtype=jnp.float32) * 0.5  # positive entries test _nonpositive_gate
    b = jax.nn.sigmoid(jax.random.normal(ks[4], (B, L, Hk, K), dtype=jnp.float32))
    w = jax.nn.sigmoid(jax.random.normal(ks[5], (B, L, Hv, V), dtype=jnp.float32))

    # 1D segment_ids with a document boundary right at prefill_block_size=128 and another at 256
    seg_1d = jnp.concatenate([
        jnp.zeros((128,), dtype=jnp.int32),
        jnp.ones((128,), dtype=jnp.int32),
        jnp.zeros((44,), dtype=jnp.int32),  # reused ID 0 at block boundary 256!
    ])

    out_single, state_single = chunk_gdn2(
        q, k, v, g, b, w,
        segment_ids=seg_1d,
        output_final_state=True,
        qk_l2norm=True,
        scale=0.25,
        is_inference=True,
        streaming_prefill=False,
        precision="fp32",
    )
    out_stream, state_stream = chunk_gdn2(
        q, k, v, g, b, w,
        segment_ids=seg_1d,
        output_final_state=True,
        qk_l2norm=True,
        scale=0.25,
        is_inference=True,
        streaming_prefill=True,
        prefill_block_size=128,
        precision="fp32",
    )

    np.testing.assert_allclose(np.asarray(out_stream), np.asarray(out_single), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(state_stream), np.asarray(state_single), rtol=1e-5, atol=1e-5)
