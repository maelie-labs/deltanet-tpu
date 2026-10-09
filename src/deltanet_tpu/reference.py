"""Reference implementations for delta-rule linear attention layers in pure JAX.

Provides token-serial recurrence implementations using `jax.lax.scan` for:
- GDN-2 (Gated DeltaNet 2 with channel-wise decay and dual gates)
- Gated DeltaNet (scalar decay per head, shared gate)
- Kimi Delta Attention (channel-wise decay, shared gate)
- Gated Linear Attention (decay, zero erase gate)
"""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp
from deltanet_tpu.gdn2 import _nonpositive_gate


def token_serial_gdn2(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    *,
    segment_ids: jax.Array | None = None,
    cu_seqlens: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    qk_l2norm: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Token-serial ground truth scan for GDN-2 in pure JAX.

    Args:
        q: Query tensor of shape (B, L, Hk, K).
        k: Key tensor of shape (B, L, Hk, K).
        v: Value tensor of shape (B, L, Hv, V).
        g: Log-decay tensor of shape (B, L, Hk, K) or (B, L, Hk, 1) or (B, L, Hk).
        b: Erase gate tensor of shape (B, L, Hk, K) or (B, L, Hk, 1) or (B, L, Hk).
        w: Write gate tensor of shape (B, L, Hv, V) or (B, L, Hv, 1) or (B, L, Hv).
        segment_ids: Document/segment IDs of shape (B, L) or (L,).
        cu_seqlens: Cumulative sequence lengths for document boundaries.
        initial_state: Optional initial state tensor of shape (B, Hv, K, V).
        output_final_state: Whether to return the final state alongside output.
        scale: Softmax/query scaling factor. Defaults to 1.0 / sqrt(K).
        qk_l2norm: Whether to apply in-kernel L2 norm to query and key. Defaults to False.

    Returns:
        Output tensor of shape (B, L, Hv, V), or tuple of (output, final_state)
        where final_state has shape (B, Hv, K, V).
    """
    B, L, Hk, K = q.shape
    Hv, V = v.shape[2], v.shape[3]
    G = Hv // Hk

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    # In-kernel L2 normalization
    if qk_l2norm:
        q_eps = jnp.array(1e-6, dtype=q.dtype)
        k_eps = jnp.array(1e-6, dtype=k.dtype)
        scale_val = jnp.array(scale, dtype=q.dtype)
        q = (q / jnp.sqrt(jnp.sum(q ** 2, axis=-1, keepdims=True) + q_eps)) * scale_val
        k = k / jnp.sqrt(jnp.sum(k ** 2, axis=-1, keepdims=True) + k_eps)
    else:
        q = q * jnp.array(scale, dtype=q.dtype)

    # Grouped value heads: repeat key-side along heads to match Hv
    if G > 1:
        q = jnp.repeat(q, G, axis=2)
        k = jnp.repeat(k, G, axis=2)
        if g.shape[2] == Hk:
            g = jnp.repeat(g, G, axis=2)
        if b.shape[2] == Hk:
            b = jnp.repeat(b, G, axis=2)
        if w.shape[2] == Hk:
            w = jnp.repeat(w, G, axis=2)

    # Expand scalar gates/decays if needed
    if g.ndim == 3:
        g = g[:, :, :, None]
    if g.shape[-1] == 1:
        g = jnp.broadcast_to(g, (B, L, Hv, K))

    if b.ndim == 3:
        b = b[:, :, :, None]
    if b.shape[-1] == 1:
        b = jnp.broadcast_to(b, (B, L, Hv, K))

    if w.ndim == 3:
        w = w[:, :, :, None]
    if w.shape[-1] == 1:
        w = jnp.broadcast_to(w, (B, L, Hv, V))

    # Segment IDs
    if segment_ids is not None:
        if segment_ids.ndim == 1:
            seg_ids = jnp.broadcast_to(segment_ids[None, :], (B, L))
        else:
            seg_ids = segment_ids
    elif cu_seqlens is not None:
        bounds = jnp.asarray(cu_seqlens)
        doc_lens = jnp.diff(bounds)
        doc_ids = jnp.repeat(jnp.arange(len(doc_lens), dtype=jnp.int32), doc_lens)
        seg_ids = jnp.broadcast_to(doc_ids[None, :], (B, L))
    else:
        seg_ids = jnp.zeros((B, L), dtype=jnp.int32)

    def step_fn(carry, inputs):
        S, prev_seg = carry
        q_t, k_t, v_t, g_t, b_t, w_t, seg_t = inputs

        # Reset state on document boundary
        reset = (seg_t != prev_seg)[:, None, None, None]
        S = jnp.where(reset, jnp.zeros_like(S), S)

        # Decay: exp(g_t) * S
        decay = jnp.exp(_nonpositive_gate(g_t))[:, :, :, None]
        S_decayed = decay * S

        # Erase: e_t = (b_t * k_t)^T S_decayed
        beta_t = b_t * k_t
        e_t = jnp.sum(beta_t[:, :, :, None] * S_decayed, axis=2)

        # Write: v_new = w_t * v_t - e_t
        v_new = w_t * v_t - e_t

        # State update: S_next = S_decayed + k_t (v_new)^T
        S_next = S_decayed + k_t[:, :, :, None] * v_new[:, :, None, :]

        # Output: o_t = S_next^T q_t
        o_t = jnp.sum(S_next * q_t[:, :, :, None], axis=2)

        return (S_next, seg_t), o_t

    if initial_state is not None:
        init_S = initial_state.astype(jnp.float32)
        init_seg = seg_ids[:, 0]
    else:
        init_S = jnp.zeros((B, Hv, K, V), dtype=jnp.float32)
        init_seg = jnp.full((B,), -1, dtype=jnp.int32)

    # Transpose to (L, B, Hv, ...) for scan
    inputs = (
        q.swapaxes(0, 1),
        k.swapaxes(0, 1),
        v.swapaxes(0, 1),
        g.swapaxes(0, 1),
        b.swapaxes(0, 1),
        w.swapaxes(0, 1),
        seg_ids.swapaxes(0, 1),
    )

    (final_S, _), outs = jax.lax.scan(step_fn, (init_S, init_seg), inputs)
    out = outs.swapaxes(0, 1)

    if output_final_state:
        return out, final_S
    return out


# Reference recurrences adapted from fla-org/flash-linear-attention (MIT, release v0.5.2)
# and the respective variant papers.


def reference_gdn(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    *,
    segment_ids: jax.Array | None = None,
    cu_seqlens: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    qk_l2norm: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Reference recurrence for Gated DeltaNet (scalar decay per head, shared gate beta).

    Computes token-serial recurrence with scalar decay per head:
        S_t = exp(g_t) * S_{t-1} + k_t * (beta_t * v_t - (beta_t * k_t)^T (exp(g_t) * S_{t-1}))
        o_t = S_t^T * q_t
    """
    B, L, Hk, K = q.shape
    Hv, V = v.shape[2], v.shape[3]
    G = Hv // Hk

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    if qk_l2norm:
        q_eps = jnp.array(1e-6, dtype=q.dtype)
        k_eps = jnp.array(1e-6, dtype=k.dtype)
        scale_val = jnp.array(scale, dtype=q.dtype)
        q = (q / jnp.sqrt(jnp.sum(q ** 2, axis=-1, keepdims=True) + q_eps)) * scale_val
        k = k / jnp.sqrt(jnp.sum(k ** 2, axis=-1, keepdims=True) + k_eps)
    else:
        q = q * jnp.array(scale, dtype=q.dtype)

    if G > 1:
        q = jnp.repeat(q, G, axis=2)
        k = jnp.repeat(k, G, axis=2)
        if g.shape[2] == Hk:
            g = jnp.repeat(g, G, axis=2)
        if beta.shape[2] == Hk:
            beta = jnp.repeat(beta, G, axis=2)

    if g.ndim == 3:
        g = g[:, :, :, None]
    if g.shape[-1] == 1:
        g = jnp.broadcast_to(g, (B, L, Hv, 1))

    if beta.ndim == 3:
        beta = beta[:, :, :, None]
    if beta.shape[-1] == 1:
        beta = jnp.broadcast_to(beta, (B, L, Hv, K))

    if segment_ids is not None:
        if segment_ids.ndim == 1:
            seg_ids = jnp.broadcast_to(segment_ids[None, :], (B, L))
        else:
            seg_ids = segment_ids
    elif cu_seqlens is not None:
        bounds = jnp.asarray(cu_seqlens)
        doc_lens = jnp.diff(bounds)
        doc_ids = jnp.repeat(jnp.arange(len(doc_lens), dtype=jnp.int32), doc_lens)
        seg_ids = jnp.broadcast_to(doc_ids[None, :], (B, L))
    else:
        seg_ids = jnp.zeros((B, L), dtype=jnp.int32)

    def step_fn(carry, inputs):
        S, prev_seg = carry
        q_t, k_t, v_t, g_t, beta_t, seg_t = inputs

        reset = (seg_t != prev_seg)[:, None, None, None]
        S = jnp.where(reset, jnp.zeros_like(S), S)

        # Scalar decay per head
        decay = jnp.exp(_nonpositive_gate(g_t))[:, :, :, None]
        S_decayed = decay * S

        # Erase: e_t = (beta_t * k_t)^T S_decayed
        b_k = beta_t * k_t
        e_t = jnp.sum(b_k[:, :, :, None] * S_decayed, axis=2)

        # Write: v_new = beta_t * v_t - e_t
        v_new = beta_t * v_t - e_t

        # State update
        S_next = S_decayed + k_t[:, :, :, None] * v_new[:, :, None, :]

        # Output
        o_t = jnp.sum(S_next * q_t[:, :, :, None], axis=2)
        return (S_next, seg_t), o_t

    if initial_state is not None:
        init_S = initial_state.astype(jnp.float32)
        init_seg = seg_ids[:, 0]
    else:
        init_S = jnp.zeros((B, Hv, K, V), dtype=jnp.float32)
        init_seg = jnp.full((B,), -1, dtype=jnp.int32)

    inputs = (
        q.swapaxes(0, 1),
        k.swapaxes(0, 1),
        v.swapaxes(0, 1),
        g.swapaxes(0, 1),
        beta.swapaxes(0, 1),
        seg_ids.swapaxes(0, 1),
    )

    (final_S, _), outs = jax.lax.scan(step_fn, (init_S, init_seg), inputs)
    out = outs.swapaxes(0, 1)

    if output_final_state:
        return out, final_S
    return out


def reference_kda(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    *,
    segment_ids: jax.Array | None = None,
    cu_seqlens: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    qk_l2norm: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Reference recurrence for Kimi Delta Attention (channel-wise decay, shared gate beta).

    Computes token-serial recurrence with channel-wise decay:
        S_t = diag(exp(g_t)) * S_{t-1} + k_t * (beta_t * v_t - (beta_t * k_t)^T (diag(exp(g_t)) * S_{t-1}))
        o_t = S_t^T * q_t
    """
    B, L, Hk, K = q.shape
    Hv, V = v.shape[2], v.shape[3]
    G = Hv // Hk

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    if qk_l2norm:
        q_eps = jnp.array(1e-6, dtype=q.dtype)
        k_eps = jnp.array(1e-6, dtype=k.dtype)
        scale_val = jnp.array(scale, dtype=q.dtype)
        q = (q / jnp.sqrt(jnp.sum(q ** 2, axis=-1, keepdims=True) + q_eps)) * scale_val
        k = k / jnp.sqrt(jnp.sum(k ** 2, axis=-1, keepdims=True) + k_eps)
    else:
        q = q * jnp.array(scale, dtype=q.dtype)

    if G > 1:
        q = jnp.repeat(q, G, axis=2)
        k = jnp.repeat(k, G, axis=2)
        if g.shape[2] == Hk:
            g = jnp.repeat(g, G, axis=2)
        if beta.shape[2] == Hk:
            beta = jnp.repeat(beta, G, axis=2)

    if g.ndim == 3:
        g = g[:, :, :, None]
    if g.shape[-1] == 1:
        g = jnp.broadcast_to(g, (B, L, Hv, K))

    if beta.ndim == 3:
        beta = beta[:, :, :, None]
    if beta.shape[-1] == 1:
        beta = jnp.broadcast_to(beta, (B, L, Hv, K))

    if segment_ids is not None:
        if segment_ids.ndim == 1:
            seg_ids = jnp.broadcast_to(segment_ids[None, :], (B, L))
        else:
            seg_ids = segment_ids
    elif cu_seqlens is not None:
        bounds = jnp.asarray(cu_seqlens)
        doc_lens = jnp.diff(bounds)
        doc_ids = jnp.repeat(jnp.arange(len(doc_lens), dtype=jnp.int32), doc_lens)
        seg_ids = jnp.broadcast_to(doc_ids[None, :], (B, L))
    else:
        seg_ids = jnp.zeros((B, L), dtype=jnp.int32)

    def step_fn(carry, inputs):
        S, prev_seg = carry
        q_t, k_t, v_t, g_t, beta_t, seg_t = inputs

        reset = (seg_t != prev_seg)[:, None, None, None]
        S = jnp.where(reset, jnp.zeros_like(S), S)

        # Channel-wise decay
        decay = jnp.exp(_nonpositive_gate(g_t))[:, :, :, None]
        S_decayed = decay * S

        # Erase: e_t = (beta_t * k_t)^T S_decayed
        b_k = beta_t * k_t
        e_t = jnp.sum(b_k[:, :, :, None] * S_decayed, axis=2)

        # Write: v_new = beta_t * v_t - e_t
        v_new = beta_t * v_t - e_t

        # State update
        S_next = S_decayed + k_t[:, :, :, None] * v_new[:, :, None, :]

        # Output
        o_t = jnp.sum(S_next * q_t[:, :, :, None], axis=2)
        return (S_next, seg_t), o_t

    if initial_state is not None:
        init_S = initial_state.astype(jnp.float32)
        init_seg = seg_ids[:, 0]
    else:
        init_S = jnp.zeros((B, Hv, K, V), dtype=jnp.float32)
        init_seg = jnp.full((B,), -1, dtype=jnp.int32)

    inputs = (
        q.swapaxes(0, 1),
        k.swapaxes(0, 1),
        v.swapaxes(0, 1),
        g.swapaxes(0, 1),
        beta.swapaxes(0, 1),
        seg_ids.swapaxes(0, 1),
    )

    (final_S, _), outs = jax.lax.scan(step_fn, (init_S, init_seg), inputs)
    out = outs.swapaxes(0, 1)

    if output_final_state:
        return out, final_S
    return out


def reference_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    *,
    segment_ids: jax.Array | None = None,
    cu_seqlens: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    qk_l2norm: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Reference recurrence for Gated Linear Attention (decay, zero erase gate).

    Computes token-serial linear recurrence:
        S_t = exp(g_t) * S_{t-1} + k_t * v_t^T
        o_t = S_t^T * q_t
    """
    B, L, Hk, K = q.shape
    Hv, V = v.shape[2], v.shape[3]
    G = Hv // Hk

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    if qk_l2norm:
        q_eps = jnp.array(1e-6, dtype=q.dtype)
        k_eps = jnp.array(1e-6, dtype=k.dtype)
        scale_val = jnp.array(scale, dtype=q.dtype)
        q = (q / jnp.sqrt(jnp.sum(q ** 2, axis=-1, keepdims=True) + q_eps)) * scale_val
        k = k / jnp.sqrt(jnp.sum(k ** 2, axis=-1, keepdims=True) + k_eps)
    else:
        q = q * jnp.array(scale, dtype=q.dtype)

    if G > 1:
        q = jnp.repeat(q, G, axis=2)
        k = jnp.repeat(k, G, axis=2)
        if g.shape[2] == Hk:
            g = jnp.repeat(g, G, axis=2)

    if g.ndim == 3:
        g = g[:, :, :, None]
    if g.shape[-1] == 1:
        g = jnp.broadcast_to(g, (B, L, Hv, K))

    if segment_ids is not None:
        if segment_ids.ndim == 1:
            seg_ids = jnp.broadcast_to(segment_ids[None, :], (B, L))
        else:
            seg_ids = segment_ids
    elif cu_seqlens is not None:
        bounds = jnp.asarray(cu_seqlens)
        doc_lens = jnp.diff(bounds)
        doc_ids = jnp.repeat(jnp.arange(len(doc_lens), dtype=jnp.int32), doc_lens)
        seg_ids = jnp.broadcast_to(doc_ids[None, :], (B, L))
    else:
        seg_ids = jnp.zeros((B, L), dtype=jnp.int32)

    def step_fn(carry, inputs):
        S, prev_seg = carry
        q_t, k_t, v_t, g_t, seg_t = inputs

        reset = (seg_t != prev_seg)[:, None, None, None]
        S = jnp.where(reset, jnp.zeros_like(S), S)

        # Decay state
        decay = jnp.exp(_nonpositive_gate(g_t))[:, :, :, None]
        S_decayed = decay * S

        # Pure additive write without erase subtraction
        S_next = S_decayed + k_t[:, :, :, None] * v_t[:, :, None, :]

        # Output
        o_t = jnp.sum(S_next * q_t[:, :, :, None], axis=2)
        return (S_next, seg_t), o_t

    if initial_state is not None:
        init_S = initial_state.astype(jnp.float32)
        init_seg = seg_ids[:, 0]
    else:
        init_S = jnp.zeros((B, Hv, K, V), dtype=jnp.float32)
        init_seg = jnp.full((B,), -1, dtype=jnp.int32)

    inputs = (
        q.swapaxes(0, 1),
        k.swapaxes(0, 1),
        v.swapaxes(0, 1),
        g.swapaxes(0, 1),
        seg_ids.swapaxes(0, 1),
    )

    (final_S, _), outs = jax.lax.scan(step_fn, (init_S, init_seg), inputs)
    out = outs.swapaxes(0, 1)

    if output_final_state:
        return out, final_S
    return out


def reference_delta_rule(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta: jax.Array,
    *,
    segment_ids: jax.Array | None = None,
    cu_seqlens: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    qk_l2norm: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Reference recurrence for DeltaNet (no decay, shared gate beta).

    Computes token-serial delta-rule recurrence without state decay:
        e_t = S_{t-1}^T (beta_t * k_t)
        v_new = beta_t * v_t - e_t
        S_t = S_{t-1} + k_t * (v_new)^T
        o_t = S_t^T * q_t
    """
    g = jnp.zeros_like(k)
    return reference_kda(
        q, k, v, g, beta,
        segment_ids=segment_ids,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_final_state=output_final_state,
        scale=scale,
        qk_l2norm=qk_l2norm,
    )


