"""End-to-end checks of the chunk_gdn2 precision options against the token-serial reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deltanet_tpu import chunk_gdn2
from deltanet_tpu.reference import token_serial_gdn2


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _inputs(strong_first_doc):
    B, L, Hk, Hv, K, V = 1, 256, 1, 2, 128, 128
    ks = jax.random.split(jax.random.PRNGKey(3), 7)
    q = jax.random.normal(ks[0], (B, L, Hk, K)) * 0.1
    k = jax.random.normal(ks[1], (B, L, Hk, K)) * 0.1
    v = jax.random.normal(ks[2], (B, L, Hv, V)) * 0.1
    g = -jnp.exp(jax.random.normal(ks[3], (B, L, Hk, K)) * 0.39 - 1.0)
    b = jax.nn.sigmoid(jax.random.normal(ks[4], (B, L, Hk, K)))
    w = jax.nn.sigmoid(jax.random.normal(ks[5], (B, L, Hv, V)))
    # Three documents; the first two share chunk 0.
    seg = jnp.concatenate([jnp.zeros(50), jnp.ones(150), 2 * jnp.ones(56)]).astype(jnp.int32)[None]
    if strong_first_doc:
        # Strong decay in document 0, weak decay afterwards.
        g = jnp.where((jnp.arange(L) < 50)[None, :, None, None], -2000.0, -1e-4 * jnp.ones_like(g))
    do = jax.random.normal(ks[6], (B, L, Hv, V))
    return (q, k, v, g, b, w), seg, do


def _errors(args, seg, do, **kw):
    def ref_loss(*a):
        return jnp.sum(token_serial_gdn2(*a, segment_ids=seg) * do)

    def loss(*a):
        return jnp.sum(chunk_gdn2(*a, segment_ids=seg, **kw).astype(jnp.float32) * do)

    o_ref = token_serial_gdn2(*args, segment_ids=seg)
    g_ref = jax.grad(ref_loss, argnums=tuple(range(6)))(*args)
    o = chunk_gdn2(*args, segment_ids=seg, **kw)
    g = jax.grad(loss, argnums=tuple(range(6)))(*args)
    assert bool(jnp.all(jnp.isfinite(o)))
    for x in g:
        assert bool(jnp.all(jnp.isfinite(x)))
    return _rel_l2(o, o_ref), [_rel_l2(x, y) for x, y in zip(g, g_ref)]


@pytest.mark.parametrize("fallback_scan", [False, True])
def test_mask_lambda_by_doc_matches_reference_after_strong_decay(fallback_scan):
    """With a strongly decayed document earlier in the chunk, mask_lambda_by_doc=True matches the fp32 reference."""
    args, seg, do = _inputs(strong_first_doc=True)
    out_err, grad_errs = _errors(args, seg, do, precision="fp32", fallback_scan=fallback_scan, mask_lambda_by_doc=True)
    assert out_err < 1e-4, out_err
    assert max(grad_errs) < 1e-4, grad_errs
    # Without the mask the result may lose precision, but never by less than the masked run.
    out_err_nomask, grad_errs_nomask = _errors(args, seg, do, precision="fp32", fallback_scan=fallback_scan)
    assert out_err <= out_err_nomask + 1e-7
    assert max(grad_errs) <= max(grad_errs_nomask) + 1e-7


@pytest.mark.parametrize("flags", [{"state_fp32_matmul": True}, {"truncate_bwd_dT_bf16": True}, {"mask_lambda_by_doc": True}])
def test_precision_flags_bf16_match_reference(flags):
    """Each precision option stays within bf16 tolerance of the fp32 reference, with document boundaries."""
    args, seg, do = _inputs(strong_first_doc=False)
    out_err, grad_errs = _errors(args, seg, do, precision="bf16", **flags)
    assert out_err < 2e-2, out_err
    assert max(grad_errs) < 2e-2, grad_errs


def test_truncate_bwd_dT_bf16_leaves_forward_unchanged():
    """truncate_bwd_dT_bf16 only affects the backward pass."""
    args, seg, _ = _inputs(strong_first_doc=False)
    o0 = chunk_gdn2(*args, segment_ids=seg, precision="bf16")
    o1 = chunk_gdn2(*args, segment_ids=seg, precision="bf16", truncate_bwd_dT_bf16=True)
    np.testing.assert_array_equal(np.asarray(o0), np.asarray(o1))
