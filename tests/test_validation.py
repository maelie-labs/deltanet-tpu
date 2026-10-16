"""Tests for API argument validation and error handling in deltanet-tpu."""

import jax
import jax.numpy as jnp
import pytest

from deltanet_tpu import chunk_gdn2, gdn2_decode_step


def test_invalid_precision():
    """Verify that an invalid precision string raises ValueError."""
    q = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    k = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    b = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Invalid precision"):
        chunk_gdn2(q, k, v, g, b, w, precision="invalid_precision")


def test_unsupported_chunk_size():
    """Verify that chunk_size != 128 raises ValueError."""
    q = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    k = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    b = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Unsupported chunk_size"):
        chunk_gdn2(q, k, v, g, b, w, chunk_size=64)


def test_unsupported_head_dims():
    """Verify that K != 128 or V != 128 raises ValueError."""
    q = jnp.zeros((1, 128, 2, 64), dtype=jnp.float32)
    k = jnp.zeros((1, 128, 2, 64), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 64), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 2, 64), dtype=jnp.float32)
    b = jnp.zeros((1, 128, 2, 64), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 64), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Unsupported head dimensions"):
        chunk_gdn2(q, k, v, g, b, w)


def test_unsupported_head_ratio():
    """Verify that Hv % Hk != 0 raises ValueError."""
    q = jnp.zeros((1, 128, 3, 128), dtype=jnp.float32)
    k = jnp.zeros((1, 128, 3, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 3, 128), dtype=jnp.float32)
    b = jnp.zeros((1, 128, 3, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Value heads .* must be divisible by key heads"):
        chunk_gdn2(q, k, v, g, b, w)


def test_invalid_cu_seqlens():
    """Verify that cu_seqlens with last element not matching sequence length raises ValueError."""
    q = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    k = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    b = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)

    with pytest.raises(ValueError, match="cu_seqlens.* must equal sequence length"):
        chunk_gdn2(q, k, v, g, b, w, cu_seqlens=[0, 64, 100])


def test_wrong_k_shape():
    """Verify wrong k shape raises ValueError."""
    q = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    b = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)

    # Wrong Hk on k
    k_bad = jnp.zeros((1, 128, 3, 128), dtype=jnp.float32)
    with pytest.raises(ValueError, match="Key tensor k shape"):
        chunk_gdn2(q, k_bad, v, g, b, w)


def test_wrong_v_shape():
    """Verify wrong v batch or seq length raises ValueError."""
    q = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    k = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    b = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)

    # Wrong sequence length on v
    v_bad = jnp.zeros((1, 64, 4, 128), dtype=jnp.float32)
    with pytest.raises(ValueError, match="Value tensor v shape"):
        chunk_gdn2(q, k, v_bad, g, b, w)


def test_wrong_gate_shapes():
    """Verify wrong last dimension on gates raises ValueError."""
    q = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    k = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    b = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)

    # Wrong last dim on g (not 1 or K=128)
    g_bad = jnp.zeros((1, 128, 2, 64), dtype=jnp.float32)
    with pytest.raises(ValueError, match="g last dimension"):
        chunk_gdn2(q, k, v, g_bad, b, w)

    # Wrong last dim on b
    b_bad = jnp.zeros((1, 128, 2, 64), dtype=jnp.float32)
    with pytest.raises(ValueError, match="b last dimension"):
        chunk_gdn2(q, k, v, jnp.zeros((1, 128, 2, 128)), b_bad, w)

    # Wrong last dim on w
    w_bad = jnp.zeros((1, 128, 4, 64), dtype=jnp.float32)
    with pytest.raises(ValueError, match="w last dimension"):
        chunk_gdn2(q, k, v, jnp.zeros((1, 128, 2, 128)), b, w_bad)


def test_segment_ids_length_mismatch():
    """Verify segment_ids length mismatch raises ValueError."""
    q = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    k = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    b = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)

    with pytest.raises(ValueError, match="segment_ids length"):
        chunk_gdn2(q, k, v, g, b, w, segment_ids=jnp.zeros(64, dtype=jnp.int32))


def test_initial_state_shape_mismatch():
    """Verify initial_state shape mismatch raises ValueError."""
    q = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    k = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    b = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)

    with pytest.raises(ValueError, match="initial_state shape"):
        chunk_gdn2(q, k, v, g, b, w, initial_state=jnp.zeros((1, 2, 128, 128)))


def test_differentiating_inference_mode_raises():
    """Verify that differentiating chunk_gdn2 when is_inference=True raises ValueError."""
    q = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    k = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    b = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)

    def loss(q):
        out = chunk_gdn2(q, k, v, g, b, w, is_inference=True)
        return jnp.sum(out)

    with pytest.raises(ValueError, match="differentiation is not supported when is_inference=True"):
        jax.grad(loss)(q)


def test_decode_step_validation():
    """Verify gdn2_decode_step shape validation."""
    q = jnp.zeros((1, 2, 128), dtype=jnp.float32)
    k = jnp.zeros((1, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 2, 128), dtype=jnp.float32)
    b = jnp.zeros((1, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 4, 128), dtype=jnp.float32)
    state = jnp.zeros((1, 4, 128, 128), dtype=jnp.float32)

    # Wrong state shape
    with pytest.raises(ValueError, match="state shape"):
        gdn2_decode_step(q, k, v, g, b, w, jnp.zeros((1, 2, 128, 128)))

    # Wrong k shape
    with pytest.raises(ValueError, match="Key tensor k shape"):
        gdn2_decode_step(q, jnp.zeros((1, 3, 128)), v, g, b, w, state)

    # Wrong v batch
    with pytest.raises(ValueError, match="Value tensor v batch"):
        gdn2_decode_step(q, k, jnp.zeros((2, 4, 128)), g, b, w, state)


def test_precision_none_raises():
    """Verify precision=None raises ValueError instead of AttributeError."""
    q = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    k = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    b = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)

    with pytest.raises(ValueError, match="precision must be a string"):
        chunk_gdn2(q, k, v, g, b, w, precision=None)


def test_float_segment_ids_raises():
    """Verify float segment_ids raises ValueError."""
    q = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    k = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    b = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)

    with pytest.raises(ValueError, match="segment_ids must have integer dtype"):
        chunk_gdn2(q, k, v, g, b, w, segment_ids=jnp.zeros((1, 128), dtype=jnp.float32))


def test_non_monotone_cu_seqlens_raises():
    """Verify non-monotone cu_seqlens raises ValueError."""
    q = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    k = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    b = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)

    with pytest.raises(ValueError, match="strictly monotonically increasing"):
        chunk_gdn2(q, k, v, g, b, w, cu_seqlens=[0, 64, 32, 128])


def test_decode_reset_shape_raises():
    """Verify decode reset with wrong batch shape raises ValueError."""
    q = jnp.zeros((2, 2, 128), dtype=jnp.float32)
    k = jnp.zeros((2, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((2, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((2, 2, 128), dtype=jnp.float32)
    b = jnp.zeros((2, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((2, 4, 128), dtype=jnp.float32)
    state = jnp.zeros((2, 4, 128, 128), dtype=jnp.float32)

    with pytest.raises(ValueError, match="reset batch dimension"):
        gdn2_decode_step(q, k, v, g, b, w, state, reset=jnp.array([True]))


def test_channel_wise_beta_in_variants_raises():
    """Verify channel-wise beta in variants raises ValueError."""
    from deltanet_tpu import chunk_gdn, chunk_kda, chunk_delta_rule
    q = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    k = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    beta_cw = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)

    with pytest.raises(ValueError, match="channel-wise beta with shape"):
        chunk_gdn(q, k, v, g[:, :, :, :1], beta_cw)

    with pytest.raises(ValueError, match="channel-wise beta with shape"):
        chunk_kda(q, k, v, g, beta_cw)

    with pytest.raises(ValueError, match="channel-wise beta with shape"):
        chunk_delta_rule(q, k, v, beta_cw)


def test_nonpositive_stability_tau_and_block_size_raise():
    """stability_tau <= 0 with stability_guard, and prefill_block_size <= 0 with streaming_prefill, raise ValueError."""
    q = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    v = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    g = jnp.zeros((1, 128, 2, 128), dtype=jnp.float32)
    w = jnp.zeros((1, 128, 4, 128), dtype=jnp.float32)
    with pytest.raises(ValueError, match="stability_tau"):
        chunk_gdn2(q, q, v, g, g, w, stability_guard=True, stability_tau=-1.0)
    with pytest.raises(ValueError, match="prefill_block_size"):
        chunk_gdn2(q, q, v, g, g, w, is_inference=True, streaming_prefill=True, prefill_block_size=0)
    state = jnp.zeros((1, 4, 128, 128), dtype=jnp.float32)
    with pytest.raises(ValueError, match="stability_tau"):
        gdn2_decode_step(q[:, 0], q[:, 0], v[:, 0], g[:, 0], g[:, 0], w[:, 0], state, stability_guard=True, stability_tau=0.0)
