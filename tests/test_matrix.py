"""Systematic parity matrix test suite for deltanet-tpu.

Tests all 5 chunk variants and all 5 decode steps against reference implementations
across:
- Pallas kernel and fallback_scan
- With and without initial_state
- With and without document segments
- Batch size B in {1, 2}
- Grouped heads: Hv = Hk and Hv = 2 * Hk
- Gate layouts: channel-wise, per-key-head, per-value-head, scalar, single-head
- Decay regimes: negative (g < 0), zero (g = 0), positive (g > 0, clamped)
- Forward output and final state
- Gradients of all inputs including initial_state
- Loss on final_state alone (validating d_final_state adjoint propagation)
- Chained calls: call 1 final_state -> call 2 initial_state
"""

import jax
import jax.numpy as jnp
import pytest

from deltanet_tpu import (
    chunk_delta_rule,
    chunk_gdn,
    chunk_gdn2,
    chunk_gla,
    chunk_kda,
    delta_rule_decode_step,
    gdn_decode_step,
    gdn2_decode_step,
    gla_decode_step,
    kda_decode_step,
)
from deltanet_tpu.reference import (
    reference_gdn,
    reference_gla,
    reference_kda,
    token_serial_gdn2,
)
import gc


@pytest.fixture(scope="function", autouse=True)
def _clear_jax_cache():
    yield
    jax.clear_caches()
    gc.collect()


def _rel_err(a: jax.Array | None, b: jax.Array | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1.0
    norm_b = float(jnp.linalg.norm(b))
    norm_diff = float(jnp.linalg.norm(a - b))
    if norm_b == 0.0:
        return norm_diff
    return norm_diff / norm_b


def _make_chunk_inputs(
    variant: str,
    B: int,
    L: int,
    Hk: int,
    Hv: int,
    K: int,
    V: int,
    gate_layout: str,
    g_regime: str,
    key_seed: int = 42,
):
    key = jax.random.PRNGKey(key_seed)
    k_q, k_k, k_v, k_g, k_b, k_w, k_init = jax.random.split(key, 7)

    q = jax.random.normal(k_q, (B, L, Hk, K), dtype=jnp.float32) * 0.1
    k = jax.random.normal(k_k, (B, L, Hk, K), dtype=jnp.float32) * 0.1
    v = jax.random.normal(k_v, (B, L, Hv, V), dtype=jnp.float32) * 0.1
    init_state = jax.random.normal(k_init, (B, Hv, K, V), dtype=jnp.float32) * 0.05

    # Decay g
    if g_regime == "negative":
        base_g = -jax.random.uniform(k_g, (B, L, Hk, K), minval=0.1, maxval=2.0)
    elif g_regime == "zero":
        base_g = jnp.zeros((B, L, Hk, K), dtype=jnp.float32)
    elif g_regime == "positive":
        base_g = jax.random.uniform(k_g, (B, L, Hk, K), minval=0.1, maxval=1.0)
    else:
        raise ValueError(f"Unknown g_regime {g_regime}")

    # Format gates based on variant and gate_layout
    if variant == "gdn2":
        if gate_layout == "channel_wise":
            g = base_g
            b = jax.nn.sigmoid(jax.random.normal(k_b, (B, L, Hk, K)) * 0.39)
            w = jax.nn.sigmoid(jax.random.normal(k_w, (B, L, Hv, V)) * 0.39)
        elif gate_layout == "per_key_head":
            g = base_g[:, :, :, :1]
            b = jax.nn.sigmoid(jax.random.normal(k_b, (B, L, Hk, 1)) * 0.39)
            w = jax.nn.sigmoid(jax.random.normal(k_w, (B, L, Hv, V)) * 0.39)
        elif gate_layout == "per_value_head":
            if g_regime == "negative":
                g = -jax.random.uniform(k_g, (B, L, Hv, 1), minval=0.1, maxval=2.0)
            elif g_regime == "zero":
                g = jnp.zeros((B, L, Hv, 1), dtype=jnp.float32)
            elif g_regime == "positive":
                g = jax.random.uniform(k_g, (B, L, Hv, 1), minval=0.1, maxval=1.0)
            else:
                raise ValueError(f"Unknown g_regime {g_regime}")
            b = jax.nn.sigmoid(jax.random.normal(k_b, (B, L, Hv, 1)) * 0.39)
            w = jax.nn.sigmoid(jax.random.normal(k_w, (B, L, Hv, V)) * 0.39)
        elif gate_layout == "scalar":
            g = base_g[:, :, :, 0]
            b = jax.nn.sigmoid(jax.random.normal(k_b, (B, L, Hk)) * 0.39)
            w = jax.nn.sigmoid(jax.random.normal(k_w, (B, L, Hv)) * 0.39)
        elif gate_layout == "single_head":
            g = base_g[:, :, :1, :]
            b = jax.nn.sigmoid(jax.random.normal(k_b, (B, L, 1, K)) * 0.39)
            w = jax.nn.sigmoid(jax.random.normal(k_w, (B, L, 1, V)) * 0.39)
        else:
            raise ValueError(f"Unknown gate_layout {gate_layout}")
        return q, k, v, g, b, w, init_state

    elif variant in ("gdn", "kda", "delta_rule"):
        if gate_layout == "per_value_head":
            beta = jax.nn.sigmoid(jax.random.normal(k_b, (B, L, Hv, 1)) * 0.39)
            if variant == "gdn":
                if g_regime == "negative":
                    g = -jax.random.uniform(k_g, (B, L, Hv, 1), minval=0.1, maxval=2.0)
                elif g_regime == "zero":
                    g = jnp.zeros((B, L, Hv, 1), dtype=jnp.float32)
                elif g_regime == "positive":
                    g = jax.random.uniform(k_g, (B, L, Hv, 1), minval=0.1, maxval=1.0)
                else:
                    raise ValueError(f"Unknown g_regime {g_regime}")
            else:
                g = base_g
        else:
            beta = jax.nn.sigmoid(jax.random.normal(k_b, (B, L, Hk, 1)) * 0.39)
            g = base_g[:, :, :, :1] if variant == "gdn" else base_g
        return q, k, v, g, beta, init_state

    elif variant == "gla":
        g = base_g
        return q, k, v, g, init_state

    else:
        raise ValueError(f"Unknown variant {variant}")


# ---------------------------------------------------------------------------
# Matrix test 1: chunk_gdn2 across all gate layouts, regimes, and head ratios
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fallback_scan", [False, True])
@pytest.mark.parametrize("Hk,Hv", [(2, 2), (2, 4)])
@pytest.mark.parametrize("gate_layout", ["channel_wise", "per_key_head", "per_value_head", "scalar", "single_head"])
@pytest.mark.parametrize("g_regime", ["negative", "zero", "positive"])
def test_matrix_chunk_gdn2_gates(fallback_scan, Hk, Hv, gate_layout, g_regime):
    B, L, K, V = 2, 300, 128, 128
    q, k, v, g, b, w, init_state = _make_chunk_inputs("gdn2", B, L, Hk, Hv, K, V, gate_layout, g_regime)
    init_arg = init_state
    segment_ids = jnp.broadcast_to(
        jnp.array([0] * (L // 2) + [1] * (L - L // 2), dtype=jnp.int32),
        (B, L)
    )
    _run_chunk_gdn2_assertions(q, k, v, g, b, w, init_arg, segment_ids, fallback_scan)


@pytest.mark.parametrize("fallback_scan", [False, True])
@pytest.mark.parametrize("Hk,Hv", [(2, 2), (2, 4)])
@pytest.mark.parametrize("B,has_init,has_segs", [
    (1, False, False),
    (1, True, False),
    (2, False, True),
    (2, True, False),
])
def test_matrix_chunk_gdn2_configurations(fallback_scan, Hk, Hv, B, has_init, has_segs):
    L, K, V = 300, 128, 128
    q, k, v, g, b, w, init_state = _make_chunk_inputs("gdn2", B, L, Hk, Hv, K, V, "channel_wise", "negative")
    init_arg = init_state if has_init else None
    if has_segs:
        segment_ids = jnp.broadcast_to(
            jnp.array([0] * (L // 2) + [1] * (L - L // 2), dtype=jnp.int32),
            (B, L)
        )
    else:
        segment_ids = None
    _run_chunk_gdn2_assertions(q, k, v, g, b, w, init_arg, segment_ids, fallback_scan)


def _run_chunk_gdn2_assertions(q, k, v, g, b, w, init_arg, segment_ids, fallback_scan):
    # Reference call
    ref_out, ref_fs = token_serial_gdn2(
        q, k, v, g, b, w,
        segment_ids=segment_ids,
        initial_state=init_arg,
        output_final_state=True,
    )

    # Kernel call
    kern_out, kern_fs = chunk_gdn2(
        q, k, v, g, b, w,
        segment_ids=segment_ids,
        initial_state=init_arg,
        output_final_state=True,
        precision="fp32",
        fallback_scan=fallback_scan,
    )

    assert _rel_err(kern_out, ref_out) < 1e-4, f"Forward out mismatch: rel_err={_rel_err(kern_out, ref_out)}"
    assert _rel_err(kern_fs, ref_fs) < 1e-4, f"Final state mismatch: rel_err={_rel_err(kern_fs, ref_fs)}"

    # Forward + Backward on output loss for all inputs (q, k, v, g, b, w)
    def loss_out_ref(q_in, k_in, v_in, g_in, b_in, w_in):
        out, _ = token_serial_gdn2(q_in, k_in, v_in, g_in, b_in, w_in, segment_ids=segment_ids, initial_state=init_arg, output_final_state=True)
        return jnp.sum(out)

    def loss_out_kern(q_in, k_in, v_in, g_in, b_in, w_in):
        out, _ = chunk_gdn2(q_in, k_in, v_in, g_in, b_in, w_in, segment_ids=segment_ids, initial_state=init_arg, output_final_state=True, precision="fp32", fallback_scan=fallback_scan)
        return jnp.sum(out)

    ref_grads = jax.grad(loss_out_ref, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, g, b, w)
    kern_grads = jax.grad(loss_out_kern, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, g, b, w)

    names = ["dq", "dk", "dv", "dg", "db", "dw"]
    for name, kern_g, ref_g in zip(names, kern_grads, ref_grads):
        assert _rel_err(kern_g, ref_g) < 1e-4, f"{name} mismatch: rel_err={_rel_err(kern_g, ref_g)}"

    # Loss on final_state alone (validates d_final_state propagation)
    def loss_fs_ref(k_in):
        _, fs = token_serial_gdn2(q, k_in, v, g, b, w, segment_ids=segment_ids, initial_state=init_arg, output_final_state=True)
        return jnp.sum(fs)

    def loss_fs_kern(k_in):
        _, fs = chunk_gdn2(q, k_in, v, g, b, w, segment_ids=segment_ids, initial_state=init_arg, output_final_state=True, precision="fp32", fallback_scan=fallback_scan)
        return jnp.sum(fs)

    ref_fs_dk = jax.grad(loss_fs_ref)(k)
    kern_fs_dk = jax.grad(loss_fs_kern)(k)
    assert _rel_err(kern_fs_dk, ref_fs_dk) < 1e-4, f"final_state dk mismatch: rel_err={_rel_err(kern_fs_dk, ref_fs_dk)}"

    # If initial_state present, test d_init_state on final_state loss
    if init_arg is not None:
        def loss_init_ref(s_in):
            _, fs = token_serial_gdn2(q, k, v, g, b, w, segment_ids=segment_ids, initial_state=s_in, output_final_state=True)
            return jnp.sum(fs)

        def loss_init_kern(s_in):
            _, fs = chunk_gdn2(q, k, v, g, b, w, segment_ids=segment_ids, initial_state=s_in, output_final_state=True, precision="fp32", fallback_scan=fallback_scan)
            return jnp.sum(fs)

        ref_ds = jax.grad(loss_init_ref)(init_arg)
        kern_ds = jax.grad(loss_init_kern)(init_arg)
        assert _rel_err(kern_ds, ref_ds) < 1e-4, f"initial_state grad mismatch: rel_err={_rel_err(kern_ds, ref_ds)}"


# ---------------------------------------------------------------------------
# Matrix test 2: other 4 chunk variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["gdn", "kda", "gla", "delta_rule"])
@pytest.mark.parametrize("fallback_scan", [False, True])
@pytest.mark.parametrize("Hk,Hv", [(2, 2), (2, 4)])
@pytest.mark.parametrize("has_init", [False, True])
def test_matrix_other_variants(variant, fallback_scan, Hk, Hv, has_init):
    B, L, K, V = 2, 300, 128, 128
    init_state = jax.random.normal(jax.random.PRNGKey(10), (B, Hv, K, V)) * 0.05 if has_init else None

    if variant == "gdn":
        q, k, v, g, beta, _ = _make_chunk_inputs("gdn", B, L, Hk, Hv, K, V, "per_key_head", "negative")
        def loss_ref(q_i, k_i, v_i, g_i, b_i):
            out, fs = reference_gdn(q_i, k_i, v_i, g_i, b_i, initial_state=init_state, output_final_state=True)
            return jnp.sum(out) + jnp.sum(fs)
        def loss_kern(q_i, k_i, v_i, g_i, b_i):
            out, fs = chunk_gdn(q_i, k_i, v_i, g_i, b_i, initial_state=init_state, output_final_state=True, precision="fp32", fallback_scan=fallback_scan)
            return jnp.sum(out) + jnp.sum(fs)
        ref_out, ref_fs = reference_gdn(q, k, v, g, beta, initial_state=init_state, output_final_state=True)
        kern_out, kern_fs = chunk_gdn(q, k, v, g, beta, initial_state=init_state, output_final_state=True, precision="fp32", fallback_scan=fallback_scan)
        ref_grads = jax.grad(loss_ref, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
        kern_grads = jax.grad(loss_kern, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
        names = ["dq", "dk", "dv", "dg", "dbeta"]

    elif variant == "kda":
        q, k, v, g, beta, _ = _make_chunk_inputs("kda", B, L, Hk, Hv, K, V, "per_key_head", "negative")
        def loss_ref(q_i, k_i, v_i, g_i, b_i):
            out, fs = reference_kda(q_i, k_i, v_i, g_i, b_i, initial_state=init_state, output_final_state=True)
            return jnp.sum(out) + jnp.sum(fs)
        def loss_kern(q_i, k_i, v_i, g_i, b_i):
            out, fs = chunk_kda(q_i, k_i, v_i, g_i, b_i, initial_state=init_state, output_final_state=True, precision="fp32", fallback_scan=fallback_scan)
            return jnp.sum(out) + jnp.sum(fs)
        ref_out, ref_fs = reference_kda(q, k, v, g, beta, initial_state=init_state, output_final_state=True)
        kern_out, kern_fs = chunk_kda(q, k, v, g, beta, initial_state=init_state, output_final_state=True, precision="fp32", fallback_scan=fallback_scan)
        ref_grads = jax.grad(loss_ref, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
        kern_grads = jax.grad(loss_kern, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
        names = ["dq", "dk", "dv", "dg", "dbeta"]

    elif variant == "gla":
        q, k, v, g, _ = _make_chunk_inputs("gla", B, L, Hk, Hv, K, V, "channel_wise", "negative")
        def loss_ref(q_i, k_i, v_i, g_i):
            out, fs = reference_gla(q_i, k_i, v_i, g_i, initial_state=init_state, output_final_state=True)
            return jnp.sum(out) + jnp.sum(fs)
        def loss_kern(q_i, k_i, v_i, g_i):
            out, fs = chunk_gla(q_i, k_i, v_i, g_i, initial_state=init_state, output_final_state=True, precision="fp32", fallback_scan=fallback_scan)
            return jnp.sum(out) + jnp.sum(fs)
        ref_out, ref_fs = reference_gla(q, k, v, g, initial_state=init_state, output_final_state=True)
        kern_out, kern_fs = chunk_gla(q, k, v, g, initial_state=init_state, output_final_state=True, precision="fp32", fallback_scan=fallback_scan)
        ref_grads = jax.grad(loss_ref, argnums=(0, 1, 2, 3))(q, k, v, g)
        kern_grads = jax.grad(loss_kern, argnums=(0, 1, 2, 3))(q, k, v, g)
        names = ["dq", "dk", "dv", "dg"]

    elif variant == "delta_rule":
        q, k, v, g, beta, _ = _make_chunk_inputs("delta_rule", B, L, Hk, Hv, K, V, "per_key_head", "negative")
        zeros_g = jnp.zeros_like(k)
        def loss_ref(q_i, k_i, v_i, b_i):
            out, fs = token_serial_gdn2(q_i, k_i, v_i, zeros_g, b_i, b_i, initial_state=init_state, output_final_state=True, qk_l2norm=False)
            return jnp.sum(out) + jnp.sum(fs)
        def loss_kern(q_i, k_i, v_i, b_i):
            out, fs = chunk_delta_rule(q_i, k_i, v_i, b_i, initial_state=init_state, output_final_state=True, precision="fp32", fallback_scan=fallback_scan)
            return jnp.sum(out) + jnp.sum(fs)
        ref_out, ref_fs = token_serial_gdn2(q, k, v, zeros_g, beta, beta, initial_state=init_state, output_final_state=True, qk_l2norm=False)
        kern_out, kern_fs = chunk_delta_rule(q, k, v, beta, initial_state=init_state, output_final_state=True, precision="fp32", fallback_scan=fallback_scan)
        ref_grads = jax.grad(loss_ref, argnums=(0, 1, 2, 3))(q, k, v, beta)
        kern_grads = jax.grad(loss_kern, argnums=(0, 1, 2, 3))(q, k, v, beta)
        names = ["dq", "dk", "dv", "dbeta"]

    assert _rel_err(kern_out, ref_out) < 1e-4, f"{variant} forward out mismatch"
    assert _rel_err(kern_fs, ref_fs) < 1e-4, f"{variant} final_state mismatch"

    for name, kern_g, ref_g in zip(names, kern_grads, ref_grads):
        assert _rel_err(kern_g, ref_g) < 1e-4, f"{variant} {name} mismatch"


# ---------------------------------------------------------------------------
# Matrix test 3: Chained calls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fallback_scan", [False, True])
@pytest.mark.parametrize("Hk,Hv", [(2, 2), (2, 4)])
def test_matrix_chained_calls(fallback_scan, Hk, Hv):
    B, L, K, V = 2, 300, 128, 128
    q1, k1, v1, g1, b1, w1, _ = _make_chunk_inputs("gdn2", B, L, Hk, Hv, K, V, "channel_wise", "negative", key_seed=1)
    q2, k2, v2, g2, b2, w2, _ = _make_chunk_inputs("gdn2", B, L, Hk, Hv, K, V, "channel_wise", "negative", key_seed=2)

    def run_chained_ref(k1_in, k2_in):
        out1, s1 = token_serial_gdn2(q1, k1_in, v1, g1, b1, w1, output_final_state=True)
        out2, s2 = token_serial_gdn2(q2, k2_in, v2, g2, b2, w2, initial_state=s1, output_final_state=True)
        return jnp.sum(out2) + jnp.sum(s2)

    def run_chained_kern(k1_in, k2_in):
        out1, s1 = chunk_gdn2(q1, k1_in, v1, g1, b1, w1, output_final_state=True, precision="fp32", fallback_scan=fallback_scan)
        out2, s2 = chunk_gdn2(q2, k2_in, v2, g2, b2, w2, initial_state=s1, output_final_state=True, precision="fp32", fallback_scan=fallback_scan)
        return jnp.sum(out2) + jnp.sum(s2)

    ref_dk1, ref_dk2 = jax.grad(run_chained_ref, argnums=(0, 1))(k1, k2)
    kern_dk1, kern_dk2 = jax.grad(run_chained_kern, argnums=(0, 1))(k1, k2)

    assert _rel_err(kern_dk1, ref_dk1) < 1e-4, f"Chained call 1 dk mismatch: rel_err={_rel_err(kern_dk1, ref_dk1)}"
    assert _rel_err(kern_dk2, ref_dk2) < 1e-4, f"Chained call 2 dk mismatch: rel_err={_rel_err(kern_dk2, ref_dk2)}"


# ---------------------------------------------------------------------------
# Matrix test 4: all 5 decode steps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("step_name", ["gdn2", "gdn", "kda", "gla", "delta_rule"])
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("Hk,Hv", [(2, 2), (2, 4)])
@pytest.mark.parametrize("has_reset", [False, True])
@pytest.mark.parametrize("g_regime", ["negative", "zero", "positive"])
def test_matrix_decode_steps(step_name, B, Hk, Hv, has_reset, g_regime):
    K, V = 128, 128
    key = jax.random.PRNGKey(77)
    kq, kk, kv, kg, kb, kw, ks = jax.random.split(key, 7)

    q = jax.random.normal(kq, (B, Hk, K)) * 0.1
    k = jax.random.normal(kk, (B, Hk, K)) * 0.1
    v = jax.random.normal(kv, (B, Hv, V)) * 0.1
    state = jax.random.normal(ks, (B, Hv, K, V)) * 0.05
    reset = jnp.array([True] + [False] * (B - 1)) if has_reset else None

    if g_regime == "negative":
        g = -jax.random.uniform(kg, (B, Hk, K), minval=0.1, maxval=2.0)
    elif g_regime == "zero":
        g = jnp.zeros((B, Hk, K), dtype=jnp.float32)
    else:
        g = jax.random.uniform(kg, (B, Hk, K), minval=0.1, maxval=1.0)

    b = jax.nn.sigmoid(jax.random.normal(kb, (B, Hk, 1)) * 0.39)
    w = jax.nn.sigmoid(jax.random.normal(kw, (B, Hv, 1)) * 0.39)

    # Reference decode: token_serial_gdn2 on sequence of length 1
    q_4d = q[:, None, :, :]
    k_4d = k[:, None, :, :]
    v_4d = v[:, None, :, :]
    state_eff = jnp.where(jnp.asarray(reset)[:, None, None, None], jnp.zeros_like(state), state) if has_reset else state

    if step_name == "gdn2":
        g_4d = g[:, None, :, :]
        b_4d = b[:, None, :, :]
        w_4d = w[:, None, :, :]
        ref_o_4d, ref_next_s = token_serial_gdn2(q_4d, k_4d, v_4d, g_4d, b_4d, w_4d, initial_state=state_eff, output_final_state=True)
        kern_o, kern_next_s = gdn2_decode_step(q, k, v, g, b, w, state, reset=reset)
    elif step_name == "gdn":
        g_scalar = g[:, :, :1]
        g_4d = g_scalar[:, None, :, :]
        b_4d = b[:, None, :, :]
        ref_o_4d, ref_next_s = reference_gdn(q_4d, k_4d, v_4d, g_4d, b_4d, initial_state=state_eff, output_final_state=True)
        kern_o, kern_next_s = gdn_decode_step(q, k, v, g_scalar, b, state, reset=reset)
    elif step_name == "kda":
        g_4d = g[:, None, :, :]
        b_4d = b[:, None, :, :]
        ref_o_4d, ref_next_s = reference_kda(q_4d, k_4d, v_4d, g_4d, b_4d, initial_state=state_eff, output_final_state=True)
        kern_o, kern_next_s = kda_decode_step(q, k, v, g, b, state, reset=reset)
    elif step_name == "gla":
        g_4d = g[:, None, :, :]
        ref_o_4d, ref_next_s = reference_gla(q_4d, k_4d, v_4d, g_4d, initial_state=state_eff, output_final_state=True)
        kern_o, kern_next_s = gla_decode_step(q, k, v, g, state, reset=reset)
    elif step_name == "delta_rule":
        zeros_g = jnp.zeros_like(k_4d)
        b_4d = b[:, None, :, :]
        ref_o_4d, ref_next_s = token_serial_gdn2(q_4d, k_4d, v_4d, zeros_g, b_4d, b_4d, initial_state=state_eff, output_final_state=True, qk_l2norm=False)
        kern_o, kern_next_s = delta_rule_decode_step(q, k, v, b, state, reset=reset)

    ref_o = ref_o_4d[:, 0, :, :]
    assert _rel_err(kern_o, ref_o) < 1.0e-4, f"Decode output mismatch for {step_name}"
    assert _rel_err(kern_next_s, ref_next_s) < 1.0e-4, f"Decode state mismatch for {step_name}"
