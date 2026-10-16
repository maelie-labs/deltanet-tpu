"""Specialized variants of delta-rule linear attention layers implemented on TPU.

Variants:
- chunk_gdn / gdn_decode_step: Gated DeltaNet (scalar decay per head, shared gate beta)
- chunk_kda / kda_decode_step: Kimi Delta Attention (channel-wise decay, shared gate beta)
- chunk_gla / gla_decode_step: Gated Linear Attention (decay, zero erase gate)
- chunk_delta_rule / delta_rule_decode_step: DeltaNet (plain delta rule without decay)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from deltanet_tpu.gdn2 import chunk_gdn2, gdn2_decode_step


def chunk_gdn(
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
    chunk_size: int = 128,
    precision: str = "bf16",
    scale: float | None = None,
    qk_l2norm: bool = False,
    fallback_scan: bool = False,
    is_inference: bool = False,
    fuse_prologue: bool = False,
    recompute_states: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Gated DeltaNet forward pass on Google Cloud TPU.

    Uses scalar log-decay per head `g` and shared gate `beta`.

    Args:
        q: Query tensor of shape (B, L, Hk, K).
        k: Key tensor of shape (B, L, Hk, K).
        v: Value tensor of shape (B, L, Hv, V).
        g: Scalar log-decay tensor of shape (B, L, Hk), (B, L, Hk, 1), (B, L, Hv), or (B, L, Hv, 1).
            When fuse_prologue=False, values must satisfy g <= 0; positive values are automatically clamped to 0.0.
            When fuse_prologue=True, g is the pre-activation log decay magnitude: g_act = -exp(min(g, 5.0)).
        beta: Shared gate tensor of shape (B, L, Hk), (B, L, Hv), (B, L, Hk, 1), or (B, L, Hv, 1).
            When fuse_prologue=False, beta is the activated gate in [0, 1].
            When fuse_prologue=True, beta is the gate logit: beta_act = sigmoid(beta).
        segment_ids: Document/segment IDs of shape (B, L) or (L,).
        cu_seqlens: Cumulative sequence lengths for document boundaries.
        initial_state: Optional initial state tensor of shape (B, Hv, K, V).
        output_final_state: Whether to return the final recurrent state alongside output.
        chunk_size: Recurrence chunk size. Defaults to 128.
        precision: Matmul precision ("bf16" or "fp32"). Defaults to "bf16".
        scale: Query scaling factor. Defaults to 1.0 / sqrt(K).
        qk_l2norm: Whether to apply in-kernel L2 norm to query and key. Defaults to False.
        fallback_scan: Use pure-JAX chunked scan fallback instead of Pallas. Defaults to False.
        is_inference: Run in forward-only inference mode without saving backward residuals. Defaults to False.
        fuse_prologue: Fuse decay exponentiation (-exp(min(g, 5.0))), gate sigmoid (sigmoid(beta)),
            and q/k L2-normalization into VMEM prologue. Requires qk_l2norm=True. Defaults to False.
        recompute_states: In backward pass, recompute chunk boundary states on the fly instead of stashing
            them in forward memory. Defaults to False.

    Returns:
        Output tensor of shape (B, L, Hv, V), or tuple of (output, final_state).
    """
    if beta.ndim == 4 and beta.shape[-1] > 1:
        raise ValueError(
            f"beta must be a per-head scalar gate with shape (B, L, H) or (B, L, H, 1); "
            f"channel-wise beta with shape {beta.shape} is not supported."
        )
    return chunk_gdn2(
        q, k, v, g, beta, beta,
        segment_ids=segment_ids,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        precision=precision,
        scale=scale,
        qk_l2norm=qk_l2norm,
        fallback_scan=fallback_scan,
        is_inference=is_inference,
        fuse_prologue=fuse_prologue,
        recompute_states=recompute_states,
    )


def chunk_kda(
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
    chunk_size: int = 128,
    precision: str = "bf16",
    scale: float | None = None,
    qk_l2norm: bool = False,
    fallback_scan: bool = False,
    is_inference: bool = False,
    fuse_prologue: bool = False,
    recompute_states: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Kimi Delta Attention (KDA) forward pass on Google Cloud TPU.

    Uses channel-wise log-decay `g` and shared gate `beta`.

    Args:
        q: Query tensor of shape (B, L, Hk, K).
        k: Key tensor of shape (B, L, Hk, K).
        v: Value tensor of shape (B, L, Hv, V).
        g: Channel-wise log-decay tensor of shape (B, L, Hk, K).
            When fuse_prologue=False, values must satisfy g <= 0; positive values are automatically clamped to 0.0.
            When fuse_prologue=True, g is the pre-activation log decay magnitude: g_act = -exp(min(g, 5.0)).
        beta: Shared gate tensor of shape (B, L, Hk), (B, L, Hv), (B, L, Hk, 1), or (B, L, Hv, 1).
            When fuse_prologue=False, beta is the activated gate in [0, 1].
            When fuse_prologue=True, beta is the gate logit: beta_act = sigmoid(beta).
        segment_ids: Document/segment IDs of shape (B, L) or (L,).
        cu_seqlens: Cumulative sequence lengths for document boundaries.
        initial_state: Optional initial state tensor of shape (B, Hv, K, V).
        output_final_state: Whether to return the final recurrent state alongside output.
        chunk_size: Recurrence chunk size. Defaults to 128.
        precision: Matmul precision ("bf16" or "fp32"). Defaults to "bf16".
        scale: Query scaling factor. Defaults to 1.0 / sqrt(K).
        qk_l2norm: Whether to apply in-kernel L2 norm to query and key. Defaults to False.
        fallback_scan: Use pure-JAX chunked scan fallback instead of Pallas. Defaults to False.
        is_inference: Run in forward-only inference mode without saving backward residuals. Defaults to False.
        fuse_prologue: Fuse decay exponentiation (-exp(min(g, 5.0))), gate sigmoid (sigmoid(beta)),
            and q/k L2-normalization into VMEM prologue. Requires qk_l2norm=True. Defaults to False.
        recompute_states: In backward pass, recompute chunk boundary states on the fly. Defaults to False.

    Returns:
        Output tensor of shape (B, L, Hv, V), or tuple of (output, final_state).
    """
    if beta.ndim == 4 and beta.shape[-1] > 1:
        raise ValueError(
            f"beta must be a per-head scalar gate with shape (B, L, H) or (B, L, H, 1); "
            f"channel-wise beta with shape {beta.shape} is not supported."
        )
    return chunk_gdn2(
        q, k, v, g, beta, beta,
        segment_ids=segment_ids,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        precision=precision,
        scale=scale,
        qk_l2norm=qk_l2norm,
        fallback_scan=fallback_scan,
        is_inference=is_inference,
        fuse_prologue=fuse_prologue,
        recompute_states=recompute_states,
    )


def chunk_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    *,
    segment_ids: jax.Array | None = None,
    cu_seqlens: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 128,
    precision: str = "bf16",
    scale: float | None = None,
    qk_l2norm: bool = False,
    fallback_scan: bool = False,
    is_inference: bool = False,
    recompute_states: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Gated Linear Attention (GLA) forward pass on Google Cloud TPU.

    Special case of GDN-2 where the erase gate b = 0 and write gate w = 1.

    Args:
        q: Query tensor of shape (B, L, Hk, K).
        k: Key tensor of shape (B, L, Hk, K).
        v: Value tensor of shape (B, L, Hv, V).
        g: Log-decay tensor of shape (B, L, Hk, K) or (B, L, Hk, 1) or (B, L, Hk).
        segment_ids: Document/segment IDs of shape (B, L) or (L,).
        cu_seqlens: Cumulative sequence lengths for document boundaries.
        initial_state: Optional initial state tensor of shape (B, Hv, K, V).
        output_final_state: Whether to return the final recurrent state alongside output.
        chunk_size: Recurrence chunk size. Defaults to 128.
        precision: Matmul precision ("bf16" or "fp32"). Defaults to "bf16".
        scale: Query scaling factor. Defaults to 1.0 / sqrt(K).
        qk_l2norm: Whether to apply in-kernel L2 norm to query and key. Defaults to False.
        fallback_scan: Use pure-JAX chunked scan fallback instead of Pallas. Defaults to False.
        is_inference: Run in forward-only inference mode without saving backward residuals. Defaults to False.
        recompute_states: In backward pass, recompute chunk boundary states on the fly. Defaults to False.

    Returns:
        Output tensor of shape (B, L, Hv, V), or tuple of (output, final_state).
    """
    b = jnp.zeros_like(k)
    w = jnp.ones_like(v)
    return chunk_gdn2(
        q, k, v, g, b, w,
        segment_ids=segment_ids,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        precision=precision,
        scale=scale,
        qk_l2norm=qk_l2norm,
        fallback_scan=fallback_scan,
        is_inference=is_inference,
        recompute_states=recompute_states,
    )


def chunk_delta_rule(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta: jax.Array,
    *,
    segment_ids: jax.Array | None = None,
    cu_seqlens: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 128,
    precision: str = "bf16",
    scale: float | None = None,
    qk_l2norm: bool = False,
    fallback_scan: bool = False,
    is_inference: bool = False,
    recompute_states: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """DeltaNet (plain delta rule without decay) forward pass on Google Cloud TPU.

    Special case of GDN-2 where log-decay g = 0 and b = w = beta.

    Args:
        q: Query tensor of shape (B, L, Hk, K).
        k: Key tensor of shape (B, L, Hk, K).
        v: Value tensor of shape (B, L, Hv, V).
        beta: Shared gate tensor of shape (B, L, Hk), (B, L, Hv), (B, L, Hk, 1), or (B, L, Hv, 1).
        segment_ids: Document/segment IDs of shape (B, L) or (L,).
        cu_seqlens: Cumulative sequence lengths for document boundaries.
        initial_state: Optional initial state tensor of shape (B, Hv, K, V).
        output_final_state: Whether to return the final recurrent state alongside output.
        chunk_size: Recurrence chunk size. Defaults to 128.
        precision: Matmul precision ("bf16" or "fp32"). Defaults to "bf16".
        scale: Query scaling factor. Defaults to 1.0 / sqrt(K).
        qk_l2norm: Whether to apply in-kernel L2 norm to query and key. Defaults to False.
        fallback_scan: Use pure-JAX chunked scan fallback instead of Pallas. Defaults to False.
        is_inference: Run in forward-only inference mode without saving backward residuals. Defaults to False.
        recompute_states: In backward pass, recompute chunk boundary states on the fly. Defaults to False.

    Returns:
        Output tensor of shape (B, L, Hv, V), or tuple of (output, final_state).
    """
    if beta.ndim == 4 and beta.shape[-1] > 1:
        raise ValueError(
            f"beta must be a per-head scalar gate with shape (B, L, H) or (B, L, H, 1); "
            f"channel-wise beta with shape {beta.shape} is not supported."
        )
    g = jnp.zeros_like(k)
    return chunk_gdn2(
        q, k, v, g, beta, beta,
        segment_ids=segment_ids,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        precision=precision,
        scale=scale,
        qk_l2norm=qk_l2norm,
        fallback_scan=fallback_scan,
        is_inference=is_inference,
        recompute_states=recompute_states,
    )


def _validate_decode_beta(beta: jax.Array) -> jax.Array:
    if beta.ndim == 3:
        if beta.shape[1] == 1:
            beta = beta[:, 0]
        elif beta.shape[-1] > 1:
            raise ValueError(
                f"beta must be a per-head scalar gate with shape (B, H), (B, H, 1) or (B, 1, H); "
                f"channel-wise beta with shape {beta.shape} is not supported."
            )
    elif beta.ndim == 4:
        if beta.shape[-1] > 1:
            raise ValueError(
                f"beta must be a per-head scalar gate with shape (B, H), (B, H, 1) or (B, 1, H); "
                f"channel-wise beta with shape {beta.shape} is not supported."
            )
        if beta.shape[1] == 1:
            beta = beta[:, 0]
    return beta


def gdn_decode_step(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    state: jax.Array,
    *,
    reset: bool | jax.Array | None = None,
    scale: float | None = None,
    qk_l2norm: bool = False,
    state_dtype: str | jnp.dtype | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Single token decode step for Gated DeltaNet (GDN) in vectorized JAX.

    Note: bf16 state validated for channel-wise GDN-2 only; scalar-decay decode not validated.
    state_dtype defaults to float32.
    """
    beta = _validate_decode_beta(beta)
    return gdn2_decode_step(
        q, k, v, g, beta, beta, state,
        reset=reset,
        scale=scale,
        qk_l2norm=qk_l2norm,
        state_dtype=state_dtype,
    )


def kda_decode_step(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    state: jax.Array,
    *,
    reset: bool | jax.Array | None = None,
    scale: float | None = None,
    qk_l2norm: bool = False,
    state_dtype: str | jnp.dtype | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Single token decode step for Kimi Delta Attention (KDA) in vectorized JAX.

    Note: bf16 state validated for channel-wise GDN-2 only; scalar-decay decode not validated.
    state_dtype defaults to float32.
    """
    beta = _validate_decode_beta(beta)
    return gdn2_decode_step(
        q, k, v, g, beta, beta, state,
        reset=reset,
        scale=scale,
        qk_l2norm=qk_l2norm,
        state_dtype=state_dtype,
    )


def gla_decode_step(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    state: jax.Array,
    *,
    reset: bool | jax.Array | None = None,
    scale: float | None = None,
    qk_l2norm: bool = False,
    state_dtype: str | jnp.dtype | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Single token decode step for Gated Linear Attention (GLA) in vectorized JAX.

    Note: bf16 state validated for channel-wise GDN-2 only; scalar-decay decode not validated.
    state_dtype defaults to float32.
    """
    b = jnp.zeros_like(k)
    w = jnp.ones_like(v)
    return gdn2_decode_step(
        q, k, v, g, b, w, state,
        reset=reset,
        scale=scale,
        qk_l2norm=qk_l2norm,
        state_dtype=state_dtype,
    )


def delta_rule_decode_step(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta: jax.Array,
    state: jax.Array,
    *,
    reset: bool | jax.Array | None = None,
    scale: float | None = None,
    qk_l2norm: bool = False,
    state_dtype: str | jnp.dtype | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Single token decode step for DeltaNet (no decay) in vectorized JAX.

    Note: bf16 state validated for channel-wise GDN-2 only; scalar-decay decode not validated.
    state_dtype defaults to float32.
    """
    beta = _validate_decode_beta(beta)
    g = jnp.zeros_like(k)
    return gdn2_decode_step(
        q, k, v, g, beta, beta, state,
        reset=reset,
        scale=scale,
        qk_l2norm=qk_l2norm,
        state_dtype=state_dtype,
    )
