"""Pallas TPU kernels for GDN-2 chunked-WY linear attention in JAX.

Features:
- Sequential recurrence with state carried in TPU Vector Memory (VMEM) across chunks.
- Sub-chunk factored intra-chunk form (BC=32) lowered to TPU matrix units.
- Optional L2 normalization on query and key in the JAX wrapper (qk_l2norm=False by default to match fla).
- Grouped value heads fused across value head groups G = Hv // Hk.
- Document boundary resets via segment_ids or cu_seqlens.
- Reverse-mode analytical backward pass that keeps the intra-chunk solve in on-chip memory (VMEM).
- Contiguous zero-transpose layouts (B, L, kd) and (B, L, vd).
"""

from __future__ import annotations

import functools
import math
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from jax.ad_checkpoint import checkpoint_name

_NON_TPU_WARNED = False

def _get_matmul_precision(precision: str) -> jax.lax.Precision:
    p = precision.lower()
    if p in ("bf16", "bfloat16"):
        return jax.lax.Precision.DEFAULT
    elif p in ("fp32", "float32"):
        return jax.lax.Precision.HIGHEST
    else:
        raise ValueError(f"Invalid precision '{precision}'; must be 'bf16' or 'fp32'.")


def _nonpositive(x: jax.Array) -> jax.Array:
    return jnp.where(x <= 0.0, x, 0.0)


@jax.custom_vjp
def _nonpositive_gate(x: jax.Array) -> jax.Array:
    return _nonpositive(x)


def _nonpositive_gate_fwd(x: jax.Array):
    return _nonpositive(x), x


def _nonpositive_gate_bwd(res, g: jax.Array):
    x = res
    return (jnp.where(x <= 0.0, g, 0.0),)


_nonpositive_gate.defvjp(_nonpositive_gate_fwd, _nonpositive_gate_bwd)


def _invert_unit_triangular_core(A, precision=jax.lax.Precision.HIGHEST):
    r"""Unit lower-triangular block doubling inversion for chunk size C.

    Decomposes (I + A) into block-diagonal D (size B=32) and strictly lower block-triangular R.
    1. Inverts diagonal blocks D via 4 levels of doubling (since A_diag^32 == 0).
    2. M = D^-1 @ R is strictly lower block-triangular with 4 block rows, so M^4 == 0.
    3. (I + M)^-1 = (I - M)(I + M^2) in 2 matrix multiplications.
    4. One-step Newton-Schulz residual refinement squares roundoff error.
    """
    C = A.shape[-1]
    B = 32 if C >= 32 else C
    eye = jnp.eye(C, dtype=jnp.float32)

    mask_diag = (jnp.arange(C)[:, None] // B) == (jnp.arange(C)[None, :] // B)

    A_f32 = A.astype(jnp.float32)
    A_diag = jnp.where(mask_diag, A_f32, 0.0)
    R = jnp.where(~mask_diag, A_f32, 0.0)

    N_D = -A_diag
    P_D = eye + N_D
    cur_ND = N_D
    levels_diag = max(0, int(math.ceil(math.log2(B))) - 1)
    for _ in range(levels_diag):
        cur_ND = jnp.matmul(cur_ND, cur_ND, precision=precision)
        P_D = P_D + jnp.matmul(P_D, cur_ND, precision=precision)
    D_inv = P_D

    M = jnp.matmul(D_inv, R, precision=precision)

    M2 = jnp.matmul(M, M, precision=precision)
    inv_I_plus_M = jnp.matmul(eye - M, eye + M2, precision=precision)
    T = jnp.matmul(inv_I_plus_M, D_inv, precision=precision)

    L = eye + A_f32
    err = 2.0 * eye - jnp.matmul(L, T, precision=precision)
    return jnp.matmul(T, err, precision=precision)



def _prepass_chunk_fwd_kernel(
    q_ref, k_ref, v_ref, g_ref, b_ref, w_ref, seg_ref,
    w_out_ref, u_out_ref, m_out_ref, qe_out_ref, ke_out_ref, decay_out_ref,
    *extra_out_refs,
    chunk_size: int, BC: int, NS: int, K: int, V: int, prec,
    fuse_prologue: bool = False,
    scale: float | None = None,
):
    q_c = q_ref[0, :, :].astype(jnp.float32)
    k_c = k_ref[0, :, :].astype(jnp.float32)
    v_c = v_ref[0, :, :].astype(jnp.float32)
    g_c = g_ref[0, :, :].astype(jnp.float32)
    b_c = b_ref[0, :, :].astype(jnp.float32)
    w_c = w_ref[0, :, :].astype(jnp.float32)
    seg_c = seg_ref[0, 0, 0, :, 0]

    if fuse_prologue:
        scale_val = scale if scale is not None else 1.0 / math.sqrt(K)
        rq = jax.lax.rsqrt(jnp.sum(q_c * q_c, axis=-1, keepdims=True) + 1e-6)
        rk = jax.lax.rsqrt(jnp.sum(k_c * k_c, axis=-1, keepdims=True) + 1e-6)
        q_act = (q_c * rq) * scale_val
        k_act = k_c * rk
        g_act = -jnp.exp(jnp.minimum(g_c, 5.0))
        b_act = jax.nn.sigmoid(b_c)
        w_act = jax.nn.sigmoid(w_c)
    else:
        q_act = q_c
        k_act = k_c
        g_act = g_c
        b_act = b_c
        w_act = w_c

    tril_ones = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
    Lambda = jnp.matmul(tril_ones, g_act, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    beta = b_act * k_act

    iota_bc = jnp.arange(BC)
    strictly_lower_bc = (iota_bc[:, None] > iota_bc[None, :])
    lower_bc = (iota_bc[:, None] >= iota_bc[None, :])

    rows_A, rows_M = [], []
    for a in range(NS):
        sa, ea = a * BC, (a + 1) * BC
        G_ref = Lambda[sa:sa+1, :]
        exp_row = jnp.exp(Lambda[sa:ea, :] - G_ref)
        b_a = beta[sa:ea, :] * exp_row
        q_a = q_act[sa:ea, :] * exp_row

        row_a_A, row_a_M = [], []
        for b_idx in range(a):
            sb, eb = b_idx * BC, (b_idx + 1) * BC
            exp_col_off = jnp.exp(G_ref - Lambda[sb:eb, :])
            k_b = k_act[sb:eb, :] * exp_col_off
            A_ab = jnp.matmul(b_a, k_b.T, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
            M_ab = jnp.matmul(q_a, k_b.T, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
            row_a_A.append(A_ab)
            row_a_M.append(M_ab)

        Lambda_a = Lambda[sa:ea, :]
        decay_diag = jnp.exp(_nonpositive(Lambda_a[:, None, :] - Lambda_a[None, :, :]))
        b_a_raw = beta[sa:ea, :]
        q_a_raw = q_act[sa:ea, :]
        k_a_raw = k_act[sa:ea, :]

        A_aa = jnp.where(strictly_lower_bc, jnp.sum((b_a_raw[:, None, :] * decay_diag) * k_a_raw[None, :, :], axis=-1), 0.0)
        M_aa = jnp.where(lower_bc, jnp.sum((q_a_raw[:, None, :] * decay_diag) * k_a_raw[None, :, :], axis=-1), 0.0)
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
    beta_exp = beta * jnp.exp(_nonpositive(Lambda))
    q_exp = q_act * jnp.exp(_nonpositive(Lambda))
    beta_exp_masked = jnp.where(same_doc_as_start, beta_exp, 0.0)
    q_exp_masked = jnp.where(same_doc_as_start, q_exp, 0.0)

    seg_end = jax.lax.slice(seg_c, (chunk_size - 1,), (chunk_size,))
    same_doc_as_end = (seg_c[:, None] == seg_end[None, :])
    Lambda_end = jax.lax.slice(Lambda, (chunk_size - 1, 0), (chunk_size, K))
    k_to_end = k_act * jnp.exp(_nonpositive(Lambda_end - Lambda))
    k_to_end_masked = jnp.where(same_doc_as_end, k_to_end, 0.0)

    T = _invert_unit_triangular_core(A, precision=jax.lax.Precision.HIGHEST)

    if prec == jax.lax.Precision.HIGHEST:
        W = jnp.matmul(T, beta_exp_masked, precision=prec)
        U = jnp.matmul(T, w_act * v_c, precision=prec)
    else:
        wv = (w_act * v_c).astype(jnp.bfloat16)
        T_bf16 = T.astype(jnp.bfloat16)
        W = jnp.matmul(T_bf16, beta_exp_masked.astype(jnp.bfloat16), precision=prec, preferred_element_type=jnp.float32)
        U = jnp.matmul(T_bf16, wv, precision=prec, preferred_element_type=jnp.float32)

    w_out_ref[0, 0, 0, :, :] = W.astype(w_out_ref.dtype)
    u_out_ref[0, 0, 0, :, :] = U.astype(u_out_ref.dtype)
    m_out_ref[0, 0, 0, :, :] = M.astype(m_out_ref.dtype)
    qe_out_ref[0, 0, 0, :, :] = q_exp_masked.astype(qe_out_ref.dtype)
    ke_out_ref[0, 0, 0, :, :] = k_to_end_masked.astype(ke_out_ref.dtype)
    decay_out_ref[0, 0, 0, 0, :] = Lambda_end[0, :].astype(decay_out_ref.dtype)
    if extra_out_refs:
        t_out_ref = extra_out_refs[0]
        t_out_ref[0, 0, 0, :, :] = T.astype(t_out_ref.dtype)


def _single_chunk_prepass_bwd_analytical(
    q_c, k_c, v_c, g_c, b_c, w_c, seg_c,
    dW, dU, dM, dqe, dke, ddecay,
    chunk_size=128, BC=32, prec=jax.lax.Precision.DEFAULT,
    T=None,
):
    tril_ones = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
    Lambda = jnp.matmul(tril_ones, g_c, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    beta = b_c * k_c
    K = q_c.shape[-1]
    NS = chunk_size // BC

    iota_bc = jnp.arange(BC)
    strictly_lower_bc = (iota_bc[:, None] > iota_bc[None, :])
    lower_bc = (iota_bc[:, None] >= iota_bc[None, :])

    same_doc_pair = (seg_c[:, None] == seg_c[None, :])
    iota = jnp.arange(chunk_size)
    strictly_causal = (iota[:, None] > iota[None, :]) & same_doc_pair
    causal = (iota[:, None] >= iota[None, :]) & same_doc_pair

    same_doc_as_start = (seg_c[:, None] == seg_c[0])
    beta_exp = beta * jnp.exp(_nonpositive(Lambda))
    beta_exp_masked = jnp.where(same_doc_as_start, beta_exp, 0.0)

    seg_end = jax.lax.slice(seg_c, (chunk_size - 1,), (chunk_size,))
    same_doc_as_end = (seg_c[:, None] == seg_end[None, :])
    Lambda_end = jax.lax.slice(Lambda, (chunk_size - 1, 0), (chunk_size, K))
    k_to_end = k_c * jnp.exp(_nonpositive(Lambda_end - Lambda))
    k_to_end_masked = jnp.where(same_doc_as_end, k_to_end, 0.0)

    if T is None:
        rows_A = []
        for a in range(NS):
            sa, ea = a * BC, (a + 1) * BC
            G_ref = Lambda[sa:sa+1, :]
            exp_row = jnp.exp(Lambda[sa:ea, :] - G_ref)
            b_a = beta[sa:ea, :] * exp_row

            row_a_A = []
            for b_idx in range(a):
                sb, eb = b_idx * BC, (b_idx + 1) * BC
                exp_col_off = jnp.exp(G_ref - Lambda[sb:eb, :])
                k_b = k_c[sb:eb, :] * exp_col_off
                A_ab = jnp.matmul(b_a, k_b.T, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
                row_a_A.append(A_ab)

            Lambda_a = Lambda[sa:ea, :]
            decay_diag = jnp.exp(_nonpositive(Lambda_a[:, None, :] - Lambda_a[None, :, :]))
            b_a_raw = beta[sa:ea, :]
            k_a_raw = k_c[sa:ea, :]

            A_aa = jnp.where(strictly_lower_bc, jnp.sum((b_a_raw[:, None, :] * decay_diag) * k_a_raw[None, :, :], axis=-1), 0.0)
            row_a_A.append(A_aa)

            if ea < chunk_size:
                pad_cols = chunk_size - ea
                row_a_A.append(jnp.zeros((BC, pad_cols), dtype=jnp.float32))

            rows_A.append(jnp.concatenate(row_a_A, axis=1))

        A_raw = jnp.concatenate(rows_A, axis=0)
        A = jnp.where(strictly_causal, A_raw, 0.0)
        T = _invert_unit_triangular_core(A, precision=jax.lax.Precision.HIGHEST)

    # 1. Backward through W and U
    if prec == jax.lax.Precision.HIGHEST:
        wv = w_c * v_c
        dT = jnp.matmul(dW, beta_exp_masked.T, precision=prec) + jnp.matmul(dU, wv.T, precision=prec)
        d_beta_exp_masked = jnp.matmul(T.T, dW, precision=prec)
        d_wv = jnp.matmul(T.T, dU, precision=prec)
    else:
        wv = (w_c * v_c).astype(jnp.bfloat16)
        T_bf16 = T.astype(jnp.bfloat16)
        beta_exp_bf16 = beta_exp_masked.astype(jnp.bfloat16)
        dT_W = jnp.matmul(dW, beta_exp_bf16.T, precision=prec, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        dT_U = jnp.matmul(dU, wv.T, precision=prec, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        dT = dT_W.astype(jnp.float32) + dT_U.astype(jnp.float32)
        d_beta_exp_masked = jnp.matmul(T_bf16.T, dW, precision=prec, preferred_element_type=jnp.float32).astype(jnp.bfloat16).astype(jnp.float32)
        d_wv = jnp.matmul(T_bf16.T, dU, precision=prec, preferred_element_type=jnp.float32).astype(jnp.bfloat16).astype(jnp.float32)

    dw = d_wv * v_c
    dv = d_wv * w_c

    # 2. Backward through T = (I+A)^-1
    T_T = T.T.astype(jnp.float32)
    dT_f32 = dT.astype(jnp.float32)
    step1 = jnp.matmul(T_T, dT_f32, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    dA_raw = - jnp.matmul(step1, T_T, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    dA = jnp.where(strictly_causal, dA_raw, 0.0)

    dM_masked = jnp.where(causal, dM, 0.0)

    # 3. Backward through prefix/suffix masked vectors
    d_beta_exp = jnp.where(same_doc_as_start, d_beta_exp_masked, 0.0)
    d_q_exp = jnp.where(same_doc_as_start, dqe, 0.0)
    d_k_to_end = jnp.where(same_doc_as_end, dke, 0.0)

    exp_neg_L = jnp.exp(_nonpositive(Lambda))
    subgrad_L = jnp.where(Lambda <= 0.0, 1.0, 0.0)

    dbeta_init = d_beta_exp * exp_neg_L
    dq_init = d_q_exp * exp_neg_L

    dLambda_prefix = (d_beta_exp * beta * exp_neg_L * subgrad_L) + (d_q_exp * q_c * exp_neg_L * subgrad_L)

    exp_delta_L = jnp.exp(_nonpositive(Lambda_end - Lambda))
    subgrad_delta_L = jnp.where((Lambda_end - Lambda) <= 0.0, 1.0, 0.0)

    dk_init = d_k_to_end * exp_delta_L
    d_delta_L = d_k_to_end * k_c * exp_delta_L * subgrad_delta_L

    total_d_Lambda_end = ddecay + jnp.sum(d_delta_L, axis=0)

    # 4. Backward per sub-block
    db_off = [[None for _ in range(a)] for a in range(NS)]
    dq_off = [[None for _ in range(a)] for a in range(NS)]
    dk_off = [[None for _ in range(a)] for a in range(NS)]
    dL_row_off = [[None for _ in range(a)] for a in range(NS)]
    dL_col_off = [[None for _ in range(a)] for a in range(NS)]
    dG_ref_off = [[None for _ in range(a)] for a in range(NS)]

    for a in range(NS):
        sa, ea = a * BC, (a + 1) * BC
        G_ref = Lambda[sa:sa+1, :]
        exp_row = jnp.exp(Lambda[sa:ea, :] - G_ref)
        b_a = beta[sa:ea, :] * exp_row
        q_a = q_c[sa:ea, :] * exp_row

        for b_idx in range(a):
            sb, eb = b_idx * BC, (b_idx + 1) * BC
            exp_col_off = jnp.exp(G_ref - Lambda[sb:eb, :])
            k_b = k_c[sb:eb, :] * exp_col_off

            dA_ab = dA[sa:ea, sb:eb]
            dM_ab = dM_masked[sa:ea, sb:eb]

            db_a = jnp.matmul(dA_ab, k_b, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
            dq_a = jnp.matmul(dM_ab, k_b, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
            dk_b = jnp.matmul(dA_ab.T, b_a, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32) + \
                   jnp.matmul(dM_ab.T, q_a, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

            dL_row = (db_a * b_a + dq_a * q_a)
            dL_col = - (dk_b * k_b)
            dG_ref = jnp.sum(-dL_row - dL_col, axis=0, keepdims=True)

            db_off[a][b_idx] = db_a * exp_row
            dq_off[a][b_idx] = dq_a * exp_row
            dk_off[a][b_idx] = dk_b * exp_col_off
            dL_row_off[a][b_idx] = dL_row
            dL_col_off[a][b_idx] = dL_col
            dG_ref_off[a][b_idx] = dG_ref

    dq_blocks = []
    dk_blocks = []
    dbeta_blocks = []
    dLambda_blocks = []

    for a in range(NS):
        sa, ea = a * BC, (a + 1) * BC

        Lambda_a = Lambda[sa:ea, :]
        diff_diag = Lambda_a[:, None, :] - Lambda_a[None, :, :]
        decay_diag = jnp.exp(_nonpositive(diff_diag))
        subgrad_diag = jnp.where(diff_diag <= 0.0, 1.0, 0.0)

        b_a_raw = beta[sa:ea, :]
        q_a_raw = q_c[sa:ea, :]
        k_a_raw = k_c[sa:ea, :]

        dA_aa = jnp.where(strictly_lower_bc, dA[sa:ea, sa:ea], 0.0)
        dM_aa = jnp.where(lower_bc, dM_masked[sa:ea, sa:ea], 0.0)

        term_A = dA_aa[:, :, None] * decay_diag
        term_M = dM_aa[:, :, None] * decay_diag

        db_a_diag = jnp.sum(term_A * k_a_raw[None, :, :], axis=1)
        dq_a_diag = jnp.sum(term_M * k_a_raw[None, :, :], axis=1)
        dk_a_diag = jnp.sum(term_A * b_a_raw[:, None, :] + term_M * q_a_raw[:, None, :], axis=0)

        d_diff = (term_A * (b_a_raw[:, None, :] * k_a_raw[None, :, :]) +
                  term_M * (q_a_raw[:, None, :] * k_a_raw[None, :, :])) * subgrad_diag

        dL_i = jnp.sum(d_diff, axis=1)
        dL_j = jnp.sum(d_diff, axis=0)
        dLambda_diag = dL_i - dL_j

        dq_a = dq_init[sa:ea, :] + dq_a_diag
        for b_idx in range(a):
            dq_a = dq_a + dq_off[a][b_idx]
        dq_blocks.append(dq_a)

        dbeta_a = dbeta_init[sa:ea, :] + db_a_diag
        for b_idx in range(a):
            dbeta_a = dbeta_a + db_off[a][b_idx]
        dbeta_blocks.append(dbeta_a)

        dk_a = dk_init[sa:ea, :] + dk_a_diag
        for a_prime in range(a + 1, NS):
            dk_a = dk_a + dk_off[a_prime][a]
        dk_blocks.append(dk_a)

        dL_a = dLambda_prefix[sa:ea, :] - d_delta_L[sa:ea, :] + dLambda_diag

        for b_idx in range(a):
            dL_a = dL_a + dL_row_off[a][b_idx]

        for a_prime in range(a + 1, NS):
            dL_a = dL_a + dL_col_off[a_prime][a]

        dG_ref_total = jnp.zeros((1, g_c.shape[-1]), dtype=jnp.float32)
        for b_idx in range(a):
            dG_ref_total = dG_ref_total + dG_ref_off[a][b_idx]

        row_0 = dL_a[0:1, :] + dG_ref_total
        dL_a = jnp.concatenate([row_0, dL_a[1:, :]], axis=0)

        if a == NS - 1:
            row_last_base = jax.lax.slice(dL_a, (BC - 1, 0), (BC, dL_a.shape[1]))
            row_last = row_last_base + total_d_Lambda_end[None, :]
            top_rows = jax.lax.slice(dL_a, (0, 0), (BC - 1, dL_a.shape[1]))
            dL_a = jnp.concatenate([top_rows, row_last], axis=0)

        dLambda_blocks.append(dL_a)

    dq_acc = jnp.concatenate(dq_blocks, axis=0)
    dk_acc = jnp.concatenate(dk_blocks, axis=0)
    dbeta_acc = jnp.concatenate(dbeta_blocks, axis=0)
    dLambda_acc = jnp.concatenate(dLambda_blocks, axis=0)

    db_out = dbeta_acc * k_c
    dk_acc = dk_acc + dbeta_acc * b_c

    dg_out = jnp.matmul(tril_ones.T, dLambda_acc, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    return dq_acc, dk_acc, dv, dg_out, db_out, dw


def _prepass_chunk_bwd_kernel(
    q_ref, k_ref, v_ref, g_ref, b_ref, w_ref, seg_ref,
    dw_ref, du_ref, dm_ref, dqe_ref, dke_ref, ddecay_ref,
    *args,
    chunk_size: int, BC: int, K: int, V: int, prec,
    fuse_prologue: bool = False,
    scale: float | None = None,
):
    if len(args) == 7:
        t_ref = args[0]
        dq_out_ref, dk_out_ref, dv_out_ref, dg_out_ref, db_out_ref, dw_out_ref = args[1:]
        T_saved = t_ref[0, 0, 0, :, :].astype(jnp.float32)
    elif len(args) == 6:
        dq_out_ref, dk_out_ref, dv_out_ref, dg_out_ref, db_out_ref, dw_out_ref = args
        T_saved = None
    else:
        raise ValueError(f"Expected 6 or 7 trailing refs in _prepass_chunk_bwd_kernel, got {len(args)}")

    q_c = q_ref[0, :, :].astype(jnp.float32)
    k_c = k_ref[0, :, :].astype(jnp.float32)
    v_c = v_ref[0, :, :].astype(jnp.float32)
    g_c = g_ref[0, :, :].astype(jnp.float32)
    b_c = b_ref[0, :, :].astype(jnp.float32)
    w_c = w_ref[0, :, :].astype(jnp.float32)
    seg_c = seg_ref[0, 0, 0, :, 0]

    dW = dw_ref[0, 0, 0, :, :].astype(jnp.float32)
    dU = du_ref[0, 0, 0, :, :].astype(jnp.float32)
    dM = dm_ref[0, 0, 0, :, :].astype(jnp.float32)
    dqe = dqe_ref[0, 0, 0, :, :].astype(jnp.float32)
    dke = dke_ref[0, 0, 0, :, :].astype(jnp.float32)
    ddecay = ddecay_ref[0, 0, 0, 0, :].astype(jnp.float32)

    if fuse_prologue:
        scale_val = scale if scale is not None else 1.0 / math.sqrt(K)
        rq = jax.lax.rsqrt(jnp.sum(q_c * q_c, axis=-1, keepdims=True) + 1e-6)
        rk = jax.lax.rsqrt(jnp.sum(k_c * k_c, axis=-1, keepdims=True) + 1e-6)
        q_act = (q_c * rq) * scale_val
        k_act = k_c * rk
        g_act = -jnp.exp(jnp.minimum(g_c, 5.0))
        b_act = jax.nn.sigmoid(b_c)
        w_act = jax.nn.sigmoid(w_c)
    else:
        q_act = q_c
        k_act = k_c
        g_act = g_c
        b_act = b_c
        w_act = w_c

    dq, dk, dv, dg, db, dw = _single_chunk_prepass_bwd_analytical(
        q_act, k_act, v_c, g_act, b_act, w_act, seg_c,
        dW, dU, dM, dqe, dke, ddecay,
        chunk_size=chunk_size, BC=BC, prec=prec,
        T=T_saved,
    )

    if fuse_prologue:
        du_g = jnp.where(g_c <= 5.0, dg * g_act, 0.0)
        du_b = db * (b_act * (1.0 - b_act))
        du_w = dw * (w_act * (1.0 - w_act))

        q_norm = q_c * rq
        dq_norm = dq * scale_val
        ydy_q = jnp.sum(q_norm * dq_norm, axis=-1, keepdims=True)
        dq_raw = rq * (dq_norm - q_norm * ydy_q)

        k_norm = k_c * rk
        ydy_k = jnp.sum(k_norm * dk, axis=-1, keepdims=True)
        dk_raw = rk * (dk - k_norm * ydy_k)

        dq, dk, dg, db, dw = dq_raw, dk_raw, du_g, du_b, du_w

    dq_out_ref[0, :, :] = dq.astype(dq_out_ref.dtype)
    dk_out_ref[0, :, :] = dk.astype(dk_out_ref.dtype)
    dv_out_ref[0, :, :] = dv.astype(dv_out_ref.dtype)
    dg_out_ref[0, :, :] = dg.astype(dg_out_ref.dtype)
    db_out_ref[0, :, :] = db.astype(db_out_ref.dtype)
    dw_out_ref[0, :, :] = dw.astype(dw_out_ref.dtype)


def _split_chunk_fwd_kernel(
    w_ref, u_ref, m_ref, qe_ref, ke_ref, decay_ref, seg_ref, init_state_ref,
    *args,
    chunk_size: int, K: int, V: int, prec, has_init_state: bool = False
):
    """Pallas forward kernel body: executes sequential recurrence with state carried in VMEM."""
    if len(args) == 4:
        out_ref, all_states_ref, final_states_ref, scratch_S_ref = args
        save_all_states = True
    elif len(args) == 3:
        out_ref, final_states_ref, scratch_S_ref = args
        all_states_ref = None
        save_all_states = False
    else:
        raise ValueError(f"Expected 3 or 4 trailing refs in _split_chunk_fwd_kernel, got {len(args)}")

    c_idx = pl.program_id(2)
    w_c = w_ref[0, 0, 0, :, :]
    u_c = u_ref[0, 0, 0, :, :]
    m_c = m_ref[0, 0, 0, :, :]
    qe_c = qe_ref[0, 0, 0, :, :]
    ke_c = ke_ref[0, 0, 0, :, :]
    decay_c = decay_ref[0, 0, 0, 0, :]
    is_new_doc = (seg_ref[0, 0, 0, 0, 0] != 0)
    all_same_doc = (seg_ref[0, 0, 0, 0, 1] != 0)

    is_chunk_0 = (c_idx == 0)
    if has_init_state:
        init_s = init_state_ref[0, 0, 0, :, :].astype(jnp.float32)
        S_chunk_0 = jnp.where(is_new_doc, jnp.zeros((K, V), dtype=jnp.float32), init_s)
    else:
        S_chunk_0 = jnp.zeros((K, V), dtype=jnp.float32)

    S_carry = jnp.where(is_chunk_0, S_chunk_0, scratch_S_ref[...])
    S_carry = jnp.where((~is_chunk_0) & is_new_doc, jnp.zeros((K, V), dtype=jnp.float32), S_carry)

    if save_all_states:
        all_states_ref[0, 0, 0, :, :] = S_carry

    def _mm_fwd(a, b):
        if prec == jax.lax.Precision.HIGHEST:
            return jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32), precision=prec)
        else:
            return jnp.matmul(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16),
                              precision=prec, preferred_element_type=jnp.float32)

    V_new = u_c - _mm_fwd(w_c, S_carry)
    O_init = _mm_fwd(qe_c, S_carry)
    O_chunk = _mm_fwd(m_c, V_new)
    O_total = O_init + O_chunk

    out_ref[0, :, :] = O_total.astype(out_ref.dtype)

    S_init_decayed = jnp.where(all_same_doc, S_carry * jnp.exp(_nonpositive(decay_c[:, None])), 0.0)
    S_from_chunk = _mm_fwd(ke_c.T, V_new)
    S_next = S_init_decayed + S_from_chunk
    scratch_S_ref[...] = S_next
    final_states_ref[0, 0, 0, :, :] = S_next


def _split_chunk_bwd_kernel(
    w_ref, u_ref, m_ref, qe_ref, ke_ref, decay_ref, seg_ref,
    all_states_ref, do_ref, d_final_state_ref,
    dw_ref, du_ref, dm_ref, dqe_ref, dke_ref, dde_ref, d_init_state_ref,
    scratch_dH_ref,
    *, chunk_size: int, K: int, V: int, NC: int, prec, has_init_state: bool = False
):
    """Pallas backward kernel body: executes reverse scan with adjoint state carried in VMEM."""
    s = pl.program_id(2)

    is_highest = (prec == jax.lax.Precision.HIGHEST)
    in_dt = jnp.float32 if is_highest else jnp.bfloat16

    w_c = w_ref[0, 0, 0, :, :].astype(in_dt)
    u_c = u_ref[0, 0, 0, :, :].astype(jnp.float32)
    m_c = m_ref[0, 0, 0, :, :].astype(in_dt)
    qe_c = qe_ref[0, 0, 0, :, :].astype(in_dt)
    ke_c = ke_ref[0, 0, 0, :, :].astype(in_dt)
    decay_c = decay_ref[0, 0, 0, 0, :].astype(jnp.float32)
    is_new_doc = (seg_ref[0, 0, 0, 0, 0] != 0)
    all_same_doc = (seg_ref[0, 0, 0, 0, 1] != 0)

    S_c = all_states_ref[0, 0, 0, :, :].astype(jnp.float32)
    do_c = do_ref[0, :, :].astype(in_dt)

    d_final_s = d_final_state_ref[0, 0, 0, :, :].astype(jnp.float32)
    dH_next = jnp.where(s == 0, d_final_s, scratch_dH_ref[...])

    def _mm(a, b):
        if is_highest:
            return jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32), precision=prec)
        else:
            return jnp.matmul(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16),
                              precision=prec, preferred_element_type=jnp.float32)

    V_new = u_c - _mm(w_c, S_c)
    dV_new = _mm(m_c.T, do_c) + _mm(ke_c, dH_next)

    dw_ref[0, 0, 0, :, :] = (- _mm(dV_new, S_c.T)).astype(dw_ref.dtype)
    du_ref[0, 0, 0, :, :] = dV_new.astype(du_ref.dtype)
    dm_ref[0, 0, 0, :, :] = _mm(do_c, V_new.T).astype(dm_ref.dtype)
    dqe_ref[0, 0, 0, :, :] = _mm(do_c, S_c.T).astype(dqe_ref.dtype)
    dke_ref[0, 0, 0, :, :] = _mm(V_new, dH_next.T).astype(dke_ref.dtype)

    decay = jnp.exp(_nonpositive(decay_c[:, None]))
    subgrad = jnp.where(decay_c[:, None] <= 0.0, 1.0, 0.0)
    d_decay_c = jnp.where(all_same_doc, jnp.sum(dH_next * S_c * decay * subgrad, axis=-1), 0.0)
    dde_ref[0, 0, 0, 0, :] = d_decay_c.astype(dde_ref.dtype)

    dH_raw = jnp.where(all_same_doc, dH_next * decay, 0.0) + _mm(qe_c.T, do_c) - _mm(w_c.T, dV_new)
    dH_prev = jnp.where(is_new_doc, jnp.zeros_like(dH_raw), dH_raw)
    scratch_dH_ref[...] = dH_prev

    if has_init_state:
        d_init_state_ref[0, 0, 0, :, :] = jnp.where(s == NC - 1, dH_prev, 0.0)


def _single_chunk_forward_1head(carry, args, chunk_size: int, prec):
    S_prev, prev_seg = carry
    q_c, k_c, g_c, b_c, v_c, w_c, seg_c = args
    K = q_c.shape[-1]
    V = v_c.shape[-1]

    is_new_doc = (seg_c[0] != prev_seg)
    all_same_doc = (seg_c[0] == seg_c[-1])

    S_carry = jnp.where(is_new_doc, jnp.zeros((K, V), dtype=jnp.float32), S_prev)

    BC = 32
    NS = chunk_size // BC
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
        exp_row = jnp.exp(Lambda[sa:ea, :] - G_ref)
        b_a = beta[sa:ea, :] * exp_row
        q_a = q_c[sa:ea, :] * exp_row

        row_a_A, row_a_M = [], []
        for b_idx in range(a):
            sb, eb = b_idx * BC, (b_idx + 1) * BC
            exp_col_off = jnp.exp(G_ref - Lambda[sb:eb, :])
            k_b = k_c[sb:eb, :] * exp_col_off
            A_ab = jnp.matmul(b_a, k_b.T, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
            M_ab = jnp.matmul(q_a, k_b.T, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
            row_a_A.append(A_ab)
            row_a_M.append(M_ab)

        Lambda_a = Lambda[sa:ea, :]
        decay_diag = jnp.exp(_nonpositive(Lambda_a[:, None, :] - Lambda_a[None, :, :]))
        b_a_raw = beta[sa:ea, :]
        q_a_raw = q_c[sa:ea, :]
        k_a_raw = k_c[sa:ea, :]

        A_aa = jnp.where(strictly_lower_bc, jnp.sum((b_a_raw[:, None, :] * decay_diag) * k_a_raw[None, :, :], axis=-1), 0.0)
        M_aa = jnp.where(lower_bc, jnp.sum((q_a_raw[:, None, :] * decay_diag) * k_a_raw[None, :, :], axis=-1), 0.0)
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
    beta_exp = beta * jnp.exp(_nonpositive(Lambda))
    q_exp = q_c * jnp.exp(_nonpositive(Lambda))
    beta_exp_masked = jnp.where(same_doc_as_start, beta_exp, 0.0)
    q_exp_masked = jnp.where(same_doc_as_start, q_exp, 0.0)

    same_doc_as_end = (seg_c[:, None] == seg_c[-1])
    Lambda_end = Lambda[-1:, :]
    k_to_end = k_c * jnp.exp(_nonpositive(Lambda_end - Lambda))
    k_to_end_masked = jnp.where(same_doc_as_end, k_to_end, 0.0)

    T = _invert_unit_triangular_core(A, precision=jax.lax.Precision.HIGHEST)

    if prec == jax.lax.Precision.HIGHEST:
        W = jnp.matmul(T, beta_exp_masked, precision=prec)
        U = jnp.matmul(T, w_c * v_c, precision=prec)
    else:
        wv = (w_c * v_c).astype(jnp.bfloat16)
        T_bf16 = T.astype(jnp.bfloat16)
        W = jnp.matmul(T_bf16, beta_exp_masked.astype(jnp.bfloat16), precision=prec, preferred_element_type=jnp.float32)
        U = jnp.matmul(T_bf16, wv, precision=prec, preferred_element_type=jnp.float32)

    def _mm(a, b):
        if prec == jax.lax.Precision.HIGHEST:
            return jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32), precision=prec)
        else:
            return jnp.matmul(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), precision=prec, preferred_element_type=jnp.float32)

    V_new = U - _mm(W, S_carry)
    O_init = _mm(q_exp_masked, S_carry)
    O_chunk = _mm(M, V_new)
    O_total = O_init + O_chunk

    S_init_decayed = jnp.where(all_same_doc, S_carry * jnp.exp(_nonpositive(Lambda_end[0, :, None])), 0.0)
    S_from_chunk = _mm(k_to_end_masked.T, V_new)
    S_next = S_init_decayed + S_from_chunk

    return (S_next, seg_c[-1]), (S_carry, O_total)


def _single_chunk_forward(carry, args, chunk_size: int, prec):
    S_prev, prev_seg = carry
    q_c, k_c, g_c, b_c, v_c, w_c, seg_c = args
    B, Hv = q_c.shape[0], q_c.shape[1]

    def per_batch(s_b, p_b, q_b, k_b, g_b, b_b, v_b, w_b, seg_b):
        def per_head(s_h, q_h, k_h, g_h, b_h, v_h, w_h):
            return _single_chunk_forward_1head(
                (s_h, p_b),
                (q_h, k_h, g_h, b_h, v_h, w_h, seg_b),
                chunk_size=chunk_size,
                prec=prec,
            )
        (s_next, _), (s_carry, o_b) = jax.vmap(per_head, in_axes=(0, 0, 0, 0, 0, 0, 0))(
            s_b, q_b, k_b, g_b, b_b, v_b, w_b
        )
        return (s_next, seg_b[-1]), (s_carry, o_b)

    (S_next, last_segs), (all_S, O_chunks) = jax.vmap(per_batch, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))(
        S_prev, prev_seg, q_c, k_c, g_c, b_c, v_c, w_c, seg_c
    )
    return (S_next, last_segs), (all_S, O_chunks)


def _run_scan_gdn2_forward(
    q, k, v, g, b, w, seg_ids, chunk_size: int, precision: str = "bf16",
    initial_state=None, fuse_prologue: bool = False,
    scale: float | None = None,
):
    B, L, Hv, K = q.shape
    V = v.shape[-1]
    num_chunks = L // chunk_size
    prec = _get_matmul_precision(precision)

    if fuse_prologue:
        scale_val = scale if scale is not None else 1.0 / math.sqrt(K)
        rq = jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        rk = jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        q = (q * rq) * scale_val
        k = k * rk
        g = -jnp.exp(jnp.minimum(g, 5.0))
        b = jax.nn.sigmoid(b)
        w = jax.nn.sigmoid(w)

    qc = q.reshape(B, num_chunks, chunk_size, Hv, K).transpose(1, 0, 3, 2, 4)
    kc = k.reshape(B, num_chunks, chunk_size, Hv, K).transpose(1, 0, 3, 2, 4)
    gc = g.reshape(B, num_chunks, chunk_size, Hv, K).transpose(1, 0, 3, 2, 4)
    bc = b.reshape(B, num_chunks, chunk_size, Hv, K).transpose(1, 0, 3, 2, 4)
    vc = v.reshape(B, num_chunks, chunk_size, Hv, V).transpose(1, 0, 3, 2, 4)
    wc = w.reshape(B, num_chunks, chunk_size, Hv, V).transpose(1, 0, 3, 2, 4)
    if seg_ids is None:
        seg_ids = jnp.zeros((B, L), dtype=jnp.int32)
    elif seg_ids.ndim == 1:
        seg_ids = jnp.broadcast_to(seg_ids[None, :], (B, L))
    segs = seg_ids.reshape(B, num_chunks, chunk_size).transpose(1, 0, 2)

    def scan_step(carry, args):
        return _single_chunk_forward(carry, args, chunk_size=chunk_size, prec=prec)

    if initial_state is not None:
        init_S = initial_state.astype(jnp.float32)
        init_seg = seg_ids[:, 0]
    else:
        init_S = jnp.zeros((B, Hv, K, V), dtype=jnp.float32)
        init_seg = jnp.full((B,), -1, dtype=jnp.int32)

    (final_S, _), (all_S, chunk_outs) = jax.lax.scan(
        scan_step,
        (init_S, init_seg),
        (qc, kc, gc, bc, vc, wc, segs),
    )
    out = chunk_outs.transpose(1, 0, 3, 2, 4).reshape(B, L, Hv, V)
    return out, all_S, final_S


def _run_scan_backward(
    q, k, v, g, b, w, seg_ids, do, chunk_size: int, precision: str = "bf16",
    initial_state=None, d_final_state=None, fuse_prologue: bool = False,
    scale: float | None = None,
):
    B, L, Hv, K = q.shape
    V = v.shape[-1]
    num_chunks = L // chunk_size
    prec = _get_matmul_precision(precision)

    if fuse_prologue:
        scale_val = scale if scale is not None else 1.0 / math.sqrt(K)
        rq = jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        rk = jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        q_act = (q * rq) * scale_val
        k_act = k * rk
        g_act = -jnp.exp(jnp.minimum(g, 5.0))
        b_act = jax.nn.sigmoid(b)
        w_act = jax.nn.sigmoid(w)
    else:
        q_act = q
        k_act = k
        g_act = g
        b_act = b
        w_act = w

    qc = q_act.reshape(B, num_chunks, chunk_size, Hv, K).transpose(1, 0, 3, 2, 4)
    kc = k_act.reshape(B, num_chunks, chunk_size, Hv, K).transpose(1, 0, 3, 2, 4)
    gc = g_act.reshape(B, num_chunks, chunk_size, Hv, K).transpose(1, 0, 3, 2, 4)
    bc = b_act.reshape(B, num_chunks, chunk_size, Hv, K).transpose(1, 0, 3, 2, 4)
    vc = v.reshape(B, num_chunks, chunk_size, Hv, V).transpose(1, 0, 3, 2, 4)
    wc = w_act.reshape(B, num_chunks, chunk_size, Hv, V).transpose(1, 0, 3, 2, 4)
    if seg_ids is None:
        seg_ids = jnp.zeros((B, L), dtype=jnp.int32)
    elif seg_ids.ndim == 1:
        seg_ids = jnp.broadcast_to(seg_ids[None, :], (B, L))
    segs = seg_ids.reshape(B, num_chunks, chunk_size).transpose(1, 0, 2)
    doc = do.reshape(B, num_chunks, chunk_size, Hv, V).transpose(1, 0, 3, 2, 4)

    def recompute_step(carry, args):
        return _single_chunk_forward(carry, args, chunk_size=chunk_size, prec=prec)

    if initial_state is not None:
        init_S = initial_state.astype(jnp.float32)
        init_seg = seg_ids[:, 0]
    else:
        init_S = jnp.zeros((B, Hv, K, V), dtype=jnp.float32)
        init_seg = jnp.full((B,), -1, dtype=jnp.int32)

    _, (all_S, _) = jax.lax.scan(
        recompute_step,
        (init_S, init_seg),
        (qc, kc, gc, bc, vc, wc, segs),
    )

    rev_c = jnp.arange(num_chunks - 1, -1, -1)
    rev_qc = jnp.flip(qc, axis=0)
    rev_kc = jnp.flip(kc, axis=0)
    rev_vc = jnp.flip(vc, axis=0)
    rev_gc = jnp.flip(gc, axis=0)
    rev_bc = jnp.flip(bc, axis=0)
    rev_wc = jnp.flip(wc, axis=0)
    rev_segs = jnp.flip(segs, axis=0)
    rev_all_S = jnp.flip(all_S, axis=0)
    rev_doc = jnp.flip(doc, axis=0)

    def local_fwd(S_curr, c_idx, q_c, k_c, g_c, b_c, v_c, w_c, seg_c):
        prev_seg = jnp.where(c_idx == 0, init_seg, segs[jnp.maximum(c_idx - 1, 0), :, -1])
        (S_next, _), (_, O_c) = _single_chunk_forward(
            (S_curr, prev_seg),
            (q_c, k_c, g_c, b_c, v_c, w_c, seg_c),
            chunk_size=chunk_size,
            prec=prec,
        )
        return S_next, O_c

    def rev_scan_step(dH, args):
        c_idx, q_c, k_c, v_c, g_c, b_c, w_c, seg_c, S_c, do_c = args
        _, vjp_fn = jax.vjp(
            lambda s, q, k, g, b, v, w: local_fwd(s, c_idx, q, k, g, b, v, w, seg_c),
            S_c, q_c, k_c, g_c, b_c, v_c, w_c,
        )
        dS_prev, dq_c, dk_c, dg_c, db_c, dv_c, dw_c = vjp_fn((dH, do_c))
        return dS_prev, (dq_c, dk_c, dv_c, dg_c, db_c, dw_c)

    init_dH = jnp.zeros((B, Hv, K, V), dtype=jnp.float32) if d_final_state is None else d_final_state.astype(jnp.float32)
    final_dH, (rev_dq, rev_dk, rev_dv, rev_dg, rev_db, rev_dw) = jax.lax.scan(
        rev_scan_step,
        init_dH,
        (rev_c, rev_qc, rev_kc, rev_vc, rev_gc, rev_bc, rev_wc, rev_segs, rev_all_S, rev_doc),
    )

    dq = jnp.flip(rev_dq, axis=0).transpose(1, 0, 3, 2, 4).reshape(B, L, Hv, K)
    dk = jnp.flip(rev_dk, axis=0).transpose(1, 0, 3, 2, 4).reshape(B, L, Hv, K)
    dv = jnp.flip(rev_dv, axis=0).transpose(1, 0, 3, 2, 4).reshape(B, L, Hv, V)
    dg = jnp.flip(rev_dg, axis=0).transpose(1, 0, 3, 2, 4).reshape(B, L, Hv, K)
    db = jnp.flip(rev_db, axis=0).transpose(1, 0, 3, 2, 4).reshape(B, L, Hv, K)
    dw = jnp.flip(rev_dw, axis=0).transpose(1, 0, 3, 2, 4).reshape(B, L, Hv, V)
    d_init = final_dH if initial_state is not None else None

    if fuse_prologue:
        du_g = jnp.where(g <= 5.0, dg * g_act, 0.0)
        du_b = db * (b_act * (1.0 - b_act))
        du_w = dw * (w_act * (1.0 - w_act))

        q_norm = q * rq
        dq_norm = dq * scale_val
        ydy_q = jnp.sum(q_norm * dq_norm, axis=-1, keepdims=True)
        dq_raw = rq * (dq_norm - q_norm * ydy_q)

        k_norm = k * rk
        ydy_k = jnp.sum(k_norm * dk, axis=-1, keepdims=True)
        dk_raw = rk * (dk - k_norm * ydy_k)

        return (dq_raw, dk_raw, dv, du_g, du_b, du_w, None, d_init)

    return (dq, dk, dv, dg, db, dw, None, d_init)


def _chunked_recurrence_fwd(
    q, k, v, g, b, w, seg_ids, init_state,
    *, chunk_size: int, precision: str = "bf16", has_init_state: bool = False,
    fallback_scan: bool = False, is_inference: bool = False,
    fuse_prologue: bool = False,
    scale: float | None = None,
):
    B, L, Hv, K = q.shape
    V = v.shape[-1]
    NC = L // chunk_size
    prec = _get_matmul_precision(precision)

    use_interpret = (jax.default_backend() != "tpu") or (
        os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")
    )

    if fallback_scan:
        out, all_states, final_state = _run_scan_gdn2_forward(
            q, k, v, g, b, w, seg_ids, chunk_size, precision=precision, initial_state=init_state,
            fuse_prologue=fuse_prologue,
            scale=scale,
        )
        if is_inference:
            return (out, final_state)
        res = (q, k, v, g, b, w, seg_ids, all_states, init_state, None, None, None, None, None, None, None, None)
        return (out, final_state, res)

    BC = 32
    grid_pre = (B, Hv, NC)
    kd = Hv * K
    vd = Hv * V
    q_3d = q.reshape(B, L, kd)
    k_3d = k.reshape(B, L, kd)
    v_3d = v.reshape(B, L, vd)
    g_3d = g.reshape(B, L, kd)
    b_3d = b.reshape(B, L, kd)
    w_3d = w.reshape(B, L, vd)

    in_specs_pre = [
        pl.BlockSpec((1, chunk_size, K), lambda b, h, c: (b, c, h)),
        pl.BlockSpec((1, chunk_size, K), lambda b, h, c: (b, c, h)),
        pl.BlockSpec((1, chunk_size, V), lambda b, h, c: (b, c, h)),
        pl.BlockSpec((1, chunk_size, K), lambda b, h, c: (b, c, h)),
        pl.BlockSpec((1, chunk_size, K), lambda b, h, c: (b, c, h)),
        pl.BlockSpec((1, chunk_size, V), lambda b, h, c: (b, c, h)),
        pl.BlockSpec((1, 1, 1, chunk_size, 128), lambda b, h, c: (b, 0, c, 0, 0)),
    ]
    out_specs_pre = [
        pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, c: (b, h, c, 0, 0)),
        pl.BlockSpec((1, 1, 1, chunk_size, V), lambda b, h, c: (b, h, c, 0, 0)),
        pl.BlockSpec((1, 1, 1, chunk_size, chunk_size), lambda b, h, c: (b, h, c, 0, 0)),
        pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, c: (b, h, c, 0, 0)),
        pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, c: (b, h, c, 0, 0)),
        pl.BlockSpec((1, 1, 1, 1, K), lambda b, h, c: (b, h, c, 0, 0)),
    ]
    is_bf16_mode = (prec == jax.lax.Precision.DEFAULT)
    pre_dtype = jnp.bfloat16 if is_bf16_mode else jnp.float32
    out_shapes_pre = [
        jax.ShapeDtypeStruct((B, Hv, NC, chunk_size, K), pre_dtype),
        jax.ShapeDtypeStruct((B, Hv, NC, chunk_size, V), pre_dtype),
        jax.ShapeDtypeStruct((B, Hv, NC, chunk_size, chunk_size), pre_dtype),
        jax.ShapeDtypeStruct((B, Hv, NC, chunk_size, K), pre_dtype),
        jax.ShapeDtypeStruct((B, Hv, NC, chunk_size, K), pre_dtype),
        jax.ShapeDtypeStruct((B, Hv, NC, 1, K), jnp.float32),
    ]
    if not is_inference:
        out_specs_pre.append(pl.BlockSpec((1, 1, 1, chunk_size, chunk_size), lambda b, h, c: (b, h, c, 0, 0)))
        out_shapes_pre.append(jax.ShapeDtypeStruct((B, Hv, NC, chunk_size, chunk_size), jnp.float32))

    segs_3d = seg_ids.reshape(B, NC, chunk_size)
    segs_padded = jnp.zeros((B, 1, NC, chunk_size, 128), dtype=jnp.int32)
    segs_padded = segs_padded.at[:, 0, :, :, 0].set(segs_3d)

    prepass_outs = pl.pallas_call(
        functools.partial(
            _prepass_chunk_fwd_kernel,
            chunk_size=chunk_size, BC=BC, NS=chunk_size // BC, K=K, V=V, prec=prec,
            fuse_prologue=fuse_prologue,
            scale=scale,
        ),
        out_shape=out_shapes_pre,
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0, grid=grid_pre, in_specs=in_specs_pre, out_specs=out_specs_pre),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel"), disable_bounds_checks=True),
        name="pallas_prepass_fwd",
        interpret=use_interpret,
    )(q_3d, k_3d, v_3d, g_3d, b_3d, w_3d, segs_padded)

    if not is_inference:
        W, U, M, q_exp_masked, k_to_end_masked, Lambda_end, T = prepass_outs
        W = checkpoint_name(W, "gdn_W")
        U = checkpoint_name(U, "gdn_U")
        M = checkpoint_name(M, "gdn_M")
        q_exp_masked = checkpoint_name(q_exp_masked, "gdn_qe")
        k_to_end_masked = checkpoint_name(k_to_end_masked, "gdn_ke")
        Lambda_end = checkpoint_name(Lambda_end, "gdn_lend")
        T = checkpoint_name(T, "gdn_T")
    else:
        W, U, M, q_exp_masked, k_to_end_masked, Lambda_end = prepass_outs
        T = None

    prev_seg_per_chunk = jnp.concatenate([
        (segs_3d[:, 0, 0:1] if has_init_state else jnp.full((B, 1), -1, dtype=jnp.int32)),
        seg_ids[:, (jnp.arange(1, NC) * chunk_size - 1)]
    ], axis=1)
    seg_first = segs_3d[:, :, 0]
    seg_last = segs_3d[:, :, chunk_size - 1]
    is_new_doc_per_chunk = (seg_first != prev_seg_per_chunk).astype(jnp.int32)
    all_same_doc_per_chunk = (seg_first == seg_last).astype(jnp.int32)

    segs_state_padded = jnp.zeros((B, 1, NC, 1, 128), dtype=jnp.int32)
    segs_state_padded = segs_state_padded.at[:, 0, :, 0, 0].set(is_new_doc_per_chunk)
    segs_state_padded = segs_state_padded.at[:, 0, :, 0, 1].set(all_same_doc_per_chunk)

    if init_state is not None:
        init_s_input = init_state.reshape(B, Hv, 1, K, V)
    else:
        init_s_input = jnp.zeros((B, Hv, 1, K, V), dtype=jnp.float32)

    grid_state = (B, Hv, NC)
    in_specs_state = [
        pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, c: (b, h, c, 0, 0)),
        pl.BlockSpec((1, 1, 1, chunk_size, V), lambda b, h, c: (b, h, c, 0, 0)),
        pl.BlockSpec((1, 1, 1, chunk_size, chunk_size), lambda b, h, c: (b, h, c, 0, 0)),
        pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, c: (b, h, c, 0, 0)),
        pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, c: (b, h, c, 0, 0)),
        pl.BlockSpec((1, 1, 1, 1, K), lambda b, h, c: (b, h, c, 0, 0)),
        pl.BlockSpec((1, 1, 1, 1, 128), lambda b, h, c: (b, 0, c, 0, 0)),
        pl.BlockSpec((1, 1, 1, K, V), lambda b, h, c: (b, h, 0, 0, 0)),
    ]
    scratch_shapes_state = [pltpu.VMEM((K, V), jnp.float32)]

    if is_inference:
        out_specs_state = [
            pl.BlockSpec((1, chunk_size, V), lambda b, h, c: (b, c, h)),
            pl.BlockSpec((1, 1, 1, K, V), lambda b, h, c: (b, h, 0, 0, 0)),
        ]
        out_shapes_state = [
            jax.ShapeDtypeStruct((B, L, vd), v.dtype),
            jax.ShapeDtypeStruct((B, Hv, 1, K, V), jnp.float32),
        ]
        out_3d, final_states = pl.pallas_call(
            functools.partial(
                _split_chunk_fwd_kernel,
                chunk_size=chunk_size, K=K, V=V, prec=prec, has_init_state=has_init_state
            ),
            out_shape=out_shapes_state,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0, grid=grid_state, in_specs=in_specs_state,
                out_specs=out_specs_state, scratch_shapes=scratch_shapes_state
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"), disable_bounds_checks=True
            ),
            name="pallas_split_fwd",
            interpret=use_interpret,
        )(W, U, M, q_exp_masked, k_to_end_masked, Lambda_end, segs_state_padded, init_s_input)
        all_states = None
    else:
        out_specs_state = [
            pl.BlockSpec((1, chunk_size, V), lambda b, h, c: (b, c, h)),
            pl.BlockSpec((1, 1, 1, K, V), lambda b, h, c: (b, h, c, 0, 0)),
            pl.BlockSpec((1, 1, 1, K, V), lambda b, h, c: (b, h, 0, 0, 0)),
        ]
        out_shapes_state = [
            jax.ShapeDtypeStruct((B, L, vd), v.dtype),
            jax.ShapeDtypeStruct((B, Hv, NC, K, V), jnp.float32),
            jax.ShapeDtypeStruct((B, Hv, 1, K, V), jnp.float32),
        ]
        out_3d, all_states, final_states = pl.pallas_call(
            functools.partial(
                _split_chunk_fwd_kernel,
                chunk_size=chunk_size, K=K, V=V, prec=prec, has_init_state=has_init_state
            ),
            out_shape=out_shapes_state,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0, grid=grid_state, in_specs=in_specs_state,
                out_specs=out_specs_state, scratch_shapes=scratch_shapes_state
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"), disable_bounds_checks=True
            ),
            name="pallas_split_fwd",
            interpret=use_interpret,
        )(W, U, M, q_exp_masked, k_to_end_masked, Lambda_end, segs_state_padded, init_s_input)

    out = out_3d.reshape(B, L, Hv, V)
    final_state = final_states[:, :, 0]

    if is_inference:
        return (out, final_state)

    res = (q, k, v, g, b, w, segs_padded, W, U, M, q_exp_masked, k_to_end_masked, Lambda_end, segs_state_padded, all_states, init_state, T)
    return (out, final_state, res)


@functools.lru_cache(maxsize=32)
def _get_inference_custom_vjp(
    chunk_size: int,
    precision: str = "bf16",
    fallback_scan: bool = False,
    with_final_state: bool = False,
    has_init_state: bool = False,
    fuse_prologue: bool = False,
    scale: float | None = None,
):
    @jax.custom_vjp
    def op(q, k, v, g, b, w, seg_ids, init_state):
        out, final_state = _chunked_recurrence_fwd(
            q, k, v, g, b, w, seg_ids, init_state,
            chunk_size=chunk_size, precision=precision, has_init_state=has_init_state,
            fallback_scan=fallback_scan, is_inference=True,
            fuse_prologue=fuse_prologue,
            scale=scale,
        )
        if with_final_state:
            return out, final_state
        return out

    def fwd(q, k, v, g, b, w, seg_ids, init_state):
        out, final_state = _chunked_recurrence_fwd(
            q, k, v, g, b, w, seg_ids, init_state,
            chunk_size=chunk_size, precision=precision, has_init_state=has_init_state,
            fallback_scan=fallback_scan, is_inference=True,
            fuse_prologue=fuse_prologue,
            scale=scale,
        )
        if with_final_state:
            return (out, final_state), None
        return out, None

    def bwd(res, cotangents):
        raise ValueError(
            "chunk_gdn2: differentiation is not supported when is_inference=True; "
            "set is_inference=False for training."
        )

    op.defvjp(fwd, bwd)
    return op


@functools.lru_cache(maxsize=32)
def _get_chunked_recurrence_custom_vjp(
    chunk_size: int,
    precision: str = "bf16",
    fallback_scan: bool = False,
    with_final_state: bool = False,
    has_init_state: bool = False,
    fuse_prologue: bool = False,
    scale: float | None = None,
):
    @jax.custom_vjp
    def op(q, k, v, g, b, w, seg_ids, init_state):
        out, final_state, _ = _chunked_recurrence_fwd(
            q, k, v, g, b, w, seg_ids, init_state,
            chunk_size=chunk_size, precision=precision, has_init_state=has_init_state,
            fallback_scan=fallback_scan, is_inference=False,
            fuse_prologue=fuse_prologue,
            scale=scale,
        )
        if with_final_state:
            return out, final_state
        return out

    def fwd(q, k, v, g, b, w, seg_ids, init_state):
        out, final_state, res = _chunked_recurrence_fwd(
            q, k, v, g, b, w, seg_ids, init_state,
            chunk_size=chunk_size, precision=precision, has_init_state=has_init_state,
            fallback_scan=fallback_scan, is_inference=False,
            fuse_prologue=fuse_prologue,
            scale=scale,
        )
        if with_final_state:
            return (out, final_state), res
        return out, res

    def bwd(res, cotangents):
        if fallback_scan:
            q, k, v, g, b, w, seg_ids, all_states, init_state = res[:9]
            if with_final_state:
                do, d_final_state = cotangents
            else:
                do = cotangents
                d_final_state = None
            return _run_scan_backward(
                q, k, v, g, b, w, seg_ids, do, chunk_size=chunk_size, precision=precision,
                initial_state=init_state, d_final_state=d_final_state,
                fuse_prologue=fuse_prologue,
                scale=scale,
            )

        q, k, v, g, b, w, segs_padded, W, U, M, q_exp_masked, k_to_end_masked, Lambda_end, segs_state_padded, all_states, init_state, T_saved = res
        if with_final_state:
            do, d_final_state = cotangents
        else:
            do = cotangents
            d_final_state = None

        B, L, Hv, K = q.shape
        V = v.shape[-1]
        NC = L // chunk_size
        BC = 32
        kd = Hv * K
        vd = Hv * V
        prec = _get_matmul_precision(precision)

        use_interpret = (jax.default_backend() != "tpu") or (
            os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")
        )

        do_3d = do.reshape(B, L, vd)

        if d_final_state is not None:
            d_final_s_input = d_final_state.reshape(B, Hv, 1, K, V).astype(jnp.float32)
        else:
            d_final_s_input = jnp.zeros((B, Hv, 1, K, V), dtype=jnp.float32)

        grid_state = (B, Hv, NC)
        bwd_in_specs = [
            pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, s: (b, h, (NC - 1) - s, 0, 0)),
            pl.BlockSpec((1, 1, 1, chunk_size, V), lambda b, h, s: (b, h, (NC - 1) - s, 0, 0)),
            pl.BlockSpec((1, 1, 1, chunk_size, chunk_size), lambda b, h, s: (b, h, (NC - 1) - s, 0, 0)),
            pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, s: (b, h, (NC - 1) - s, 0, 0)),
            pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, s: (b, h, (NC - 1) - s, 0, 0)),
            pl.BlockSpec((1, 1, 1, 1, K), lambda b, h, s: (b, h, (NC - 1) - s, 0, 0)),
            pl.BlockSpec((1, 1, 1, 1, 128), lambda b, h, s: (b, 0, (NC - 1) - s, 0, 0)),
            pl.BlockSpec((1, 1, 1, K, V), lambda b, h, s: (b, h, (NC - 1) - s, 0, 0)),
            pl.BlockSpec((1, chunk_size, V), lambda b, h, s: (b, (NC - 1) - s, h)),
            pl.BlockSpec((1, 1, 1, K, V), lambda b, h, s: (b, h, 0, 0, 0)),
        ]
        bwd_out_specs = [
            pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, s: (b, h, (NC - 1) - s, 0, 0)),
            pl.BlockSpec((1, 1, 1, chunk_size, V), lambda b, h, s: (b, h, (NC - 1) - s, 0, 0)),
            pl.BlockSpec((1, 1, 1, chunk_size, chunk_size), lambda b, h, s: (b, h, (NC - 1) - s, 0, 0)),
            pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, s: (b, h, (NC - 1) - s, 0, 0)),
            pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, s: (b, h, (NC - 1) - s, 0, 0)),
            pl.BlockSpec((1, 1, 1, 1, K), lambda b, h, s: (b, h, (NC - 1) - s, 0, 0)),
            pl.BlockSpec((1, 1, 1, K, V), lambda b, h, s: (b, h, 0, 0, 0)),
        ]
        scratch_shapes = [pltpu.VMEM((K, V), jnp.float32)]

        grads_pallas = pl.pallas_call(
            functools.partial(
                _split_chunk_bwd_kernel,
                chunk_size=chunk_size, K=K, V=V, NC=NC, prec=prec, has_init_state=has_init_state
            ),
            out_shape=[
                jax.ShapeDtypeStruct((B, Hv, NC, chunk_size, K), jnp.float32),
                jax.ShapeDtypeStruct((B, Hv, NC, chunk_size, V), jnp.float32),
                jax.ShapeDtypeStruct((B, Hv, NC, chunk_size, chunk_size), jnp.float32),
                jax.ShapeDtypeStruct((B, Hv, NC, chunk_size, K), jnp.float32),
                jax.ShapeDtypeStruct((B, Hv, NC, chunk_size, K), jnp.float32),
                jax.ShapeDtypeStruct((B, Hv, NC, 1, K), jnp.float32),
                jax.ShapeDtypeStruct((B, Hv, 1, K, V), jnp.float32),
            ],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0, grid=grid_state, in_specs=bwd_in_specs,
                out_specs=bwd_out_specs, scratch_shapes=scratch_shapes
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"), disable_bounds_checks=True
            ),
            name="pallas_split_bwd",
            interpret=use_interpret,
        )(W, U, M, q_exp_masked, k_to_end_masked, Lambda_end, segs_state_padded, all_states, do_3d, d_final_s_input)

        dW, dU, dM, dqe, dke, ddecay, d_init_state_raw = grads_pallas

        grid_pre = (B, Hv, NC)
        q_3d = q.reshape(B, L, kd)
        k_3d = k.reshape(B, L, kd)
        v_3d = v.reshape(B, L, vd)
        g_3d = g.reshape(B, L, kd)
        b_3d = b.reshape(B, L, kd)
        w_3d = w.reshape(B, L, vd)

        in_specs_bwd_pre = [
            pl.BlockSpec((1, chunk_size, K), lambda b, h, c: (b, c, h)),
            pl.BlockSpec((1, chunk_size, K), lambda b, h, c: (b, c, h)),
            pl.BlockSpec((1, chunk_size, V), lambda b, h, c: (b, c, h)),
            pl.BlockSpec((1, chunk_size, K), lambda b, h, c: (b, c, h)),
            pl.BlockSpec((1, chunk_size, K), lambda b, h, c: (b, c, h)),
            pl.BlockSpec((1, chunk_size, V), lambda b, h, c: (b, c, h)),
            pl.BlockSpec((1, 1, 1, chunk_size, 128), lambda b, h, c: (b, 0, c, 0, 0)),
            pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, c: (b, h, c, 0, 0)),
            pl.BlockSpec((1, 1, 1, chunk_size, V), lambda b, h, c: (b, h, c, 0, 0)),
            pl.BlockSpec((1, 1, 1, chunk_size, chunk_size), lambda b, h, c: (b, h, c, 0, 0)),
            pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, c: (b, h, c, 0, 0)),
            pl.BlockSpec((1, 1, 1, chunk_size, K), lambda b, h, c: (b, h, c, 0, 0)),
            pl.BlockSpec((1, 1, 1, 1, K), lambda b, h, c: (b, h, c, 0, 0)),
        ]
        bwd_pre_inputs = [q_3d, k_3d, v_3d, g_3d, b_3d, w_3d, segs_padded, dW, dU, dM, dqe, dke, ddecay]
        if T_saved is not None:
            in_specs_bwd_pre.append(pl.BlockSpec((1, 1, 1, chunk_size, chunk_size), lambda b, h, c: (b, h, c, 0, 0)))
            bwd_pre_inputs.append(T_saved)

        out_specs_bwd_pre = [
            pl.BlockSpec((1, chunk_size, K), lambda b, h, c: (b, c, h)),
            pl.BlockSpec((1, chunk_size, K), lambda b, h, c: (b, c, h)),
            pl.BlockSpec((1, chunk_size, V), lambda b, h, c: (b, c, h)),
            pl.BlockSpec((1, chunk_size, K), lambda b, h, c: (b, c, h)),
            pl.BlockSpec((1, chunk_size, K), lambda b, h, c: (b, c, h)),
            pl.BlockSpec((1, chunk_size, V), lambda b, h, c: (b, c, h)),
        ]
        out_shapes_bwd_pre = [
            jax.ShapeDtypeStruct((B, L, kd), q.dtype),
            jax.ShapeDtypeStruct((B, L, kd), k.dtype),
            jax.ShapeDtypeStruct((B, L, vd), v.dtype),
            jax.ShapeDtypeStruct((B, L, kd), g.dtype),
            jax.ShapeDtypeStruct((B, L, kd), b.dtype),
            jax.ShapeDtypeStruct((B, L, vd), w.dtype),
        ]

        dq_3d, dk_3d, dv_3d, dg_3d, db_3d, dw_3d = pl.pallas_call(
            functools.partial(
                _prepass_chunk_bwd_kernel,
                chunk_size=chunk_size, BC=BC, K=K, V=V, prec=prec,
                fuse_prologue=fuse_prologue,
                scale=scale,
            ),
            out_shape=out_shapes_bwd_pre,
            grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0, grid=grid_pre, in_specs=in_specs_bwd_pre, out_specs=out_specs_bwd_pre),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel"), disable_bounds_checks=True),
            name="pallas_prepass_bwd",
            interpret=use_interpret,
        )(*bwd_pre_inputs)

        dq = dq_3d.reshape(B, L, Hv, K)
        dk = dk_3d.reshape(B, L, Hv, K)
        dv = dv_3d.reshape(B, L, Hv, V)
        dg = dg_3d.reshape(B, L, Hv, K)
        db = db_3d.reshape(B, L, Hv, K)
        dw = dw_3d.reshape(B, L, Hv, V)

        d_init = d_init_state_raw.squeeze(2).astype(init_state.dtype) if (has_init_state and init_state is not None) else None
        return (dq, dk, dv, dg, db, dw, None, d_init)

    op.defvjp(fwd, bwd)
    return op


def chunk_gdn2(
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
    chunk_size: int = 128,
    precision: str = "bf16",
    scale: float | None = None,
    qk_l2norm: bool = False,
    fallback_scan: bool = False,
    is_inference: bool = False,
    fuse_prologue: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Forward pass of chunked-WY GDN-2 on Google Cloud TPU.

    Args:
        q: Query tensor of shape (B, L, Hk, K).
        k: Key tensor of shape (B, L, Hk, K).
        v: Value tensor of shape (B, L, Hv, V).
        g: Log-decay tensor of shape (B, L, Hk, K), (B, L, Hk, 1), (B, L, Hk), or (B, L, Hv).
            When fuse_prologue=False, values must satisfy g <= 0; positive values are clamped to 0.0.
            When fuse_prologue=True, g is the pre-activation log decay magnitude: g_act = -exp(min(g, 5.0)).
        b: Erase gate tensor of shape (B, L, Hk, K), (B, L, Hk, 1), (B, L, Hk), or (B, L, Hv).
            When fuse_prologue=False, b is the activated erase gate in [0, 1].
            When fuse_prologue=True, b is the erase gate logit: b_act = sigmoid(b).
        w: Write gate tensor of shape (B, L, Hv, V), (B, L, Hv, 1), or (B, L, Hv).
            When fuse_prologue=False, w is the activated write gate in [0, 1].
            When fuse_prologue=True, w is the write gate logit: w_act = sigmoid(w).
        segment_ids: Document/segment IDs of shape (B, L) or (L,). Preferred for jax.jit.
        cu_seqlens: Cumulative sequence lengths for document boundaries (static list/tuple or 1D array).
        initial_state: Optional initial state tensor of shape (B, Hv, K, V).
        output_final_state: Whether to return the final recurrent state alongside output.
        chunk_size: Recurrence chunk size. Must be 128.
        precision: Matmul precision ("bf16" or "fp32"). Defaults to "bf16".
        scale: Query scaling factor. Defaults to 1.0 / sqrt(K).
        qk_l2norm: Whether to apply in-kernel L2 norm to query and key. Defaults to False.
        fallback_scan: Use pure-JAX chunked scan fallback instead of Pallas. Defaults to False.
        is_inference: Run in forward-only inference mode without stashing backward residuals. Defaults to False.
        fuse_prologue: Fuse decay exponentiation (-exp(min(g, 5.0))), gate sigmoids (sigmoid(b), sigmoid(w)),
            and q/k L2-normalization directly into the Pallas VMEM prepass kernel.
            Requires qk_l2norm=True. Defaults to False.

    Returns:
        Output tensor of shape (B, L, Hv, V), or tuple of (output, final_state)
        where final_state has shape (B, Hv, K, V).
    """
    if fuse_prologue and not qk_l2norm:
        raise ValueError(
            "fuse_prologue=True requires qk_l2norm=True because L2-normalization is fused directly into the VMEM kernel prologue."
        )

    if precision is None or not isinstance(precision, str):
        raise ValueError(f"precision must be a string ('bf16' or 'fp32'), got {precision!r}.")
    precision_clean = precision.lower()
    if precision_clean not in ("bf16", "bfloat16", "fp32", "float32"):
        raise ValueError(f"Invalid precision '{precision}'; must be 'bf16' or 'fp32'.")

    if chunk_size != 128:
        raise ValueError(f"Unsupported chunk_size={chunk_size}; Pallas TPU kernel requires chunk_size=128.")

    if q.ndim != 4:
        raise ValueError(f"Query tensor q must be 4D (B, L, Hk, K), got shape {q.shape}.")
    if k.ndim != 4:
        raise ValueError(f"Key tensor k must be 4D (B, L, Hk, K), got shape {k.shape}.")
    if v.ndim != 4:
        raise ValueError(f"Value tensor v must be 4D (B, L, Hv, V), got shape {v.shape}.")

    B, orig_L, Hk, K = q.shape
    L = orig_L
    Hv, V = v.shape[2], v.shape[3]

    if k.shape[0] != B or k.shape[1] != L or k.shape[2] != Hk or k.shape[3] != K:
        raise ValueError(
            f"Key tensor k shape {k.shape} must match query q shape ({B}, {L}, {Hk}, {K})."
        )
    if v.shape[0] != B or v.shape[1] != L:
        raise ValueError(
            f"Value tensor v shape {v.shape} must match query batch and seq dimensions ({B}, {L})."
        )

    if K != 128 or V != 128:
        raise ValueError(f"Unsupported head dimensions (K={K}, V={V}); Pallas TPU kernel requires K=128 and V=128.")

    if Hv % Hk != 0:
        raise ValueError(f"Value heads ({Hv}) must be divisible by key heads ({Hk}).")

    if g.ndim not in (3, 4):
        raise ValueError(f"g must have rank 3 (B, L, Hg) or rank 4 (B, L, Hg, Kg), got rank {g.ndim} with shape {g.shape}.")
    if g.shape[0] != B or g.shape[1] != L:
        raise ValueError(f"g shape {g.shape} must match query batch and seq dimensions ({B}, {L}).")
    if g.shape[2] not in (1, Hk, Hv):
        raise ValueError(f"g head dimension ({g.shape[2]}) must match key heads ({Hk}) or value heads ({Hv}) or be 1.")
    if g.ndim == 4 and g.shape[3] not in (1, K):
        raise ValueError(f"g last dimension ({g.shape[3]}) must be 1 or K ({K}).")

    if b.ndim not in (3, 4):
        raise ValueError(f"b must have rank 3 (B, L, Hb) or rank 4 (B, L, Hb, Kb), got rank {b.ndim} with shape {b.shape}.")
    if b.shape[0] != B or b.shape[1] != L:
        raise ValueError(f"b shape {b.shape} must match query batch and seq dimensions ({B}, {L}).")
    if b.shape[2] not in (1, Hk, Hv):
        raise ValueError(f"b head dimension ({b.shape[2]}) must match key heads ({Hk}) or value heads ({Hv}) or be 1.")
    if b.ndim == 4 and b.shape[3] not in (1, K):
        raise ValueError(f"b last dimension ({b.shape[3]}) must be 1 or K ({K}).")

    if w.ndim not in (3, 4):
        raise ValueError(f"w must have rank 3 (B, L, Hw) or rank 4 (B, L, Hw, Vw), got rank {w.ndim} with shape {w.shape}.")
    if w.shape[0] != B or w.shape[1] != L:
        raise ValueError(f"w shape {w.shape} must match query batch and seq dimensions ({B}, {L}).")
    if w.shape[2] not in (1, Hk, Hv):
        raise ValueError(f"w head dimension ({w.shape[2]}) must match key heads ({Hk}) or value heads ({Hv}) or be 1.")
    if w.ndim == 4 and w.shape[3] not in (1, V):
        raise ValueError(f"w last dimension ({w.shape[3]}) must be 1 or V ({V}).")

    if segment_ids is not None and cu_seqlens is not None:
        raise ValueError("Cannot pass both segment_ids and cu_seqlens; specify at most one.")

    if segment_ids is not None:
        if not jnp.issubdtype(segment_ids.dtype, jnp.integer):
            raise ValueError(f"segment_ids must have integer dtype, got {segment_ids.dtype}.")
        if segment_ids.ndim == 1:
            if segment_ids.shape[0] != L:
                raise ValueError(f"1D segment_ids length ({segment_ids.shape[0]}) must match sequence length ({L}).")
        elif segment_ids.ndim == 2:
            if segment_ids.shape != (B, L):
                raise ValueError(f"2D segment_ids shape {segment_ids.shape} must match (B={B}, L={L}).")
        else:
            raise ValueError(f"segment_ids must be 1D (L,) or 2D (B, L), got rank {segment_ids.ndim} with shape {segment_ids.shape}.")

    if initial_state is not None:
        if initial_state.shape != (B, Hv, K, V):
            raise ValueError(f"initial_state shape {initial_state.shape} must match (B={B}, Hv={Hv}, K={K}, V={V}).")

    global _NON_TPU_WARNED
    if not _NON_TPU_WARNED and jax.default_backend() != "tpu":
        _NON_TPU_WARNED = True
        import warnings
        import sys
        stacklevel = 2
        try:
            frame = sys._getframe(1)
            while frame and frame.f_globals.get("__name__", "").startswith("deltanet_tpu"):
                stacklevel += 1
                frame = frame.f_back
        except Exception:
            stacklevel = 2
        warnings.warn(
            f"deltanet-tpu: running on non-TPU backend '{jax.default_backend()}'. "
            "Pallas TPU kernels run in interpret mode or fallback_scan on CPU/GPU.",
            UserWarning,
            stacklevel=stacklevel,
        )

    out_dtype = v.dtype
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    if not fuse_prologue:
        g = _nonpositive_gate(g.astype(jnp.float32))
    else:
        g = g.astype(jnp.float32)
    b = b.astype(jnp.float32)
    w = w.astype(jnp.float32)
    if initial_state is not None:
        initial_state = initial_state.astype(jnp.float32)

    # If g or b has Hv heads and Hk < Hv (e.g. Qwen3.5-4B / Qwen3-Next),
    # repeat q and k to Hv heads and execute with Hk = Hv.
    # Whichever gate is still at Hk heads is also repeated to Hv heads.
    if ((g.ndim >= 3 and g.shape[2] == Hv) or (b.ndim >= 3 and b.shape[2] == Hv)) and Hk < Hv:
        repeat_factor = Hv // Hk
        q = jnp.repeat(q, repeat_factor, axis=2)
        k = jnp.repeat(k, repeat_factor, axis=2)
        if g.ndim >= 3 and g.shape[2] == Hk:
            g = jnp.repeat(g, repeat_factor, axis=2)
        if b.ndim >= 3 and b.shape[2] == Hk:
            b = jnp.repeat(b, repeat_factor, axis=2)
        if w.ndim >= 3 and w.shape[2] == Hk:
            w = jnp.repeat(w, repeat_factor, axis=2)
        Hk = Hv

    G = Hv // Hk
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    else:
        if isinstance(scale, jax.core.Tracer):
            raise ValueError(
                "chunk_gdn2: scale must be a concrete Python float or scalar, not a JAX tracer."
            )
        try:
            scale = float(scale)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"chunk_gdn2: scale must be convertible to a concrete float, got {type(scale)}: {e}"
            )

    # Expand scalar gates/decays or broadcast single-head gates across heads
    if g.ndim == 3:
        g = g[:, :, :, None]
    if g.shape[2] == 1 and Hk > 1:
        g = jnp.broadcast_to(g, (B, L, Hk, g.shape[-1]))
    if g.shape[-1] == 1:
        g = jnp.broadcast_to(g, (B, L, Hk, K))

    if b.ndim == 3:
        b = b[:, :, :, None]
    if b.shape[2] == 1 and Hk > 1:
        b = jnp.broadcast_to(b, (B, L, Hk, b.shape[-1]))
    if b.shape[-1] == 1:
        b = jnp.broadcast_to(b, (B, L, Hk, K))

    if w.ndim == 3:
        w = w[:, :, :, None]
    if w.shape[2] == 1 and Hv > 1:
        w = jnp.broadcast_to(w, (B, L, Hv, w.shape[-1]))
    elif w.shape[2] == Hk and G > 1:
        w = jnp.repeat(w, G, axis=2)
    if w.shape[-1] != V:
        w = jnp.broadcast_to(w, (B, L, Hv, V))


    # Document boundaries: create segment IDs per token (shape (B, L))
    if segment_ids is not None:
        if segment_ids.ndim == 1:
            seg_ids = jnp.broadcast_to(segment_ids[None, :], (B, L))
        else:
            seg_ids = segment_ids
    elif cu_seqlens is not None:
        import numpy as np
        if isinstance(cu_seqlens, (list, tuple)):
            if len(cu_seqlens) == 0 or cu_seqlens[0] != 0:
                raise ValueError(f"cu_seqlens[0] must be 0, got {cu_seqlens[0] if cu_seqlens else 'empty'}.")
            if cu_seqlens[-1] != orig_L:
                raise ValueError(f"cu_seqlens[-1] ({cu_seqlens[-1]}) must equal sequence length ({orig_L}).")
            bounds = np.asarray(cu_seqlens)
            doc_lens = np.diff(bounds)
            if len(doc_lens) > 0 and np.any(doc_lens <= 0):
                raise ValueError(f"cu_seqlens must be strictly monotonically increasing, got {cu_seqlens}.")
            doc_ids = np.repeat(np.arange(len(doc_lens), dtype=np.int32), doc_lens)
            seg_ids = jnp.broadcast_to(jnp.asarray(doc_ids)[None, :], (B, L))
        elif isinstance(cu_seqlens, np.ndarray) and not isinstance(cu_seqlens, jax.core.Tracer):
            if len(cu_seqlens) == 0 or cu_seqlens[0] != 0:
                raise ValueError(f"cu_seqlens[0] must be 0, got {cu_seqlens[0]}.")
            if cu_seqlens[-1] != orig_L:
                raise ValueError(f"cu_seqlens[-1] ({cu_seqlens[-1]}) must equal sequence length ({orig_L}).")
            doc_lens = np.diff(cu_seqlens)
            if len(doc_lens) > 0 and np.any(doc_lens <= 0):
                raise ValueError(f"cu_seqlens must be strictly monotonically increasing, got {cu_seqlens}.")
            doc_ids = np.repeat(np.arange(len(doc_lens), dtype=np.int32), doc_lens)
            seg_ids = jnp.broadcast_to(jnp.asarray(doc_ids)[None, :], (B, L))
        else:
            if not isinstance(cu_seqlens, jax.core.Tracer):
                try:
                    if int(cu_seqlens[0]) != 0:
                        raise ValueError(f"cu_seqlens[0] must be 0, got {int(cu_seqlens[0])}.")
                    if int(cu_seqlens[-1]) != orig_L:
                        raise ValueError(f"cu_seqlens[-1] ({int(cu_seqlens[-1])}) must equal sequence length ({orig_L}).")
                    bounds_np = np.asarray(cu_seqlens)
                    if len(bounds_np) > 1 and np.any(np.diff(bounds_np) <= 0):
                        raise ValueError(f"cu_seqlens must be strictly monotonically increasing, got {cu_seqlens}.")
                except (TypeError, jax.errors.ConcretizationTypeError):
                    pass
            bounds = jnp.asarray(cu_seqlens)
            doc_lens = jnp.diff(bounds)
            doc_ids = jnp.repeat(jnp.arange(doc_lens.shape[0], dtype=jnp.int32), doc_lens, total_repeat_length=orig_L)
            seg_ids = jnp.broadcast_to(doc_ids[None, :], (B, L))
    else:
        seg_ids = jnp.zeros((B, L), dtype=jnp.int32)

    # Canonicalize segment IDs: reset wherever the ID changes so reused IDs or pad gaps create distinct documents
    if orig_L > 1:
        seg_ids = jnp.concatenate([
            jnp.zeros((B, 1), dtype=jnp.int32),
            jnp.cumsum((seg_ids[:, 1:] != seg_ids[:, :-1]).astype(jnp.int32), axis=1)
        ], axis=1)
    else:
        seg_ids = jnp.zeros((B, orig_L), dtype=jnp.int32)

    # Dynamic sequence padding for arbitrary sequence lengths
    pad_len = (chunk_size - (L % chunk_size)) % chunk_size
    if pad_len > 0:
        q = jnp.pad(q, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        if fuse_prologue:
            g = jnp.pad(g, ((0, 0), (0, pad_len), (0, 0), (0, 0)), constant_values=-64.0)
            b = jnp.pad(b, ((0, 0), (0, pad_len), (0, 0), (0, 0)), constant_values=-64.0)
            w = jnp.pad(w, ((0, 0), (0, pad_len), (0, 0), (0, 0)), constant_values=-64.0)
        else:
            g = jnp.pad(g, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            b = jnp.pad(b, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            w = jnp.pad(w, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        seg_ids = jnp.pad(seg_ids, ((0, 0), (0, pad_len)), mode="edge")
        L = L + pad_len

    # In-kernel L2 norm
    if not fuse_prologue:
        if qk_l2norm:
            q_eps = jnp.array(1e-6, dtype=q.dtype)
            k_eps = jnp.array(1e-6, dtype=k.dtype)
            scale_val = jnp.array(scale, dtype=q.dtype)
            q_norm = (q / jnp.sqrt(jnp.sum(q ** 2, axis=-1, keepdims=True) + q_eps)) * scale_val
            k_norm = k / jnp.sqrt(jnp.sum(k ** 2, axis=-1, keepdims=True) + k_eps)
        else:
            q_norm = q * jnp.array(scale, dtype=q.dtype)
            k_norm = k
    else:
        q_norm = q
        k_norm = k

    has_init = (initial_state is not None)

    if is_inference:
        inf_op = _get_inference_custom_vjp(
            chunk_size,
            precision=precision,
            fallback_scan=fallback_scan,
            with_final_state=output_final_state,
            has_init_state=has_init,
            fuse_prologue=fuse_prologue,
            scale=scale,
        )
        if G > 1:
            v_fused = v.reshape(B, L, Hk, G * V)
            w_fused = w.reshape(B, L, Hk, G * V)
            if has_init:
                init_fused = initial_state.reshape(B, Hk, G, K, V).transpose(0, 1, 3, 2, 4).reshape(B, Hk, K, G * V)
            else:
                init_fused = None

            res = inf_op(q_norm, k_norm, v_fused, g, b, w_fused, seg_ids, init_fused)
            if output_final_state:
                out_fused, final_state_fused = res
                out = out_fused.reshape(B, L, Hv, V).astype(out_dtype)
                out = out[:, :orig_L]
                final_state = final_state_fused.reshape(B, Hk, K, G, V).transpose(0, 1, 3, 2, 4).reshape(B, Hv, K, V)
                return out, final_state
            else:
                out = res.reshape(B, L, Hv, V).astype(out_dtype)
                return out[:, :orig_L]
        else:
            res = inf_op(q_norm, k_norm, v, g, b, w, seg_ids, initial_state)
            if output_final_state:
                out, final_state = res
                out = out[:, :orig_L]
                return out.astype(out_dtype), final_state
            else:
                out = res.astype(out_dtype)
                return out[:, :orig_L]

    vjp_op = _get_chunked_recurrence_custom_vjp(
        chunk_size,
        precision=precision,
        fallback_scan=fallback_scan,
        with_final_state=output_final_state,
        has_init_state=has_init,
        fuse_prologue=fuse_prologue,
        scale=scale,
    )

    if G > 1:
        # Value head fusion: carry (K, G * V) state per key head
        v_fused = v.reshape(B, L, Hk, G * V)
        w_fused = w.reshape(B, L, Hk, G * V)
        if has_init:
            init_fused = initial_state.reshape(B, Hk, G, K, V).transpose(0, 1, 3, 2, 4).reshape(B, Hk, K, G * V)
        else:
            init_fused = None

        res = vjp_op(q_norm, k_norm, v_fused, g, b, w_fused, seg_ids, init_fused)
        if output_final_state:
            out_fused, final_state_fused = res
            out = out_fused.reshape(B, L, Hv, V).astype(out_dtype)
            out = out[:, :orig_L]
            final_state = final_state_fused.reshape(B, Hk, K, G, V).transpose(0, 1, 3, 2, 4).reshape(B, Hv, K, V)
            return out, final_state
        else:
            out = res.reshape(B, L, Hv, V).astype(out_dtype)
            return out[:, :orig_L]
    else:
        res = vjp_op(q_norm, k_norm, v, g, b, w, seg_ids, initial_state)
        if output_final_state:
            out, final_state = res
            out = out[:, :orig_L]
            return out.astype(out_dtype), final_state
        else:
            out = res.astype(out_dtype)
            return out[:, :orig_L]


def gdn2_decode_step(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    state: jax.Array,
    *,
    reset: bool | jax.Array | None = None,
    scale: float | None = None,
    qk_l2norm: bool = False,
    state_dtype: str | jnp.dtype | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Single token decode step for GDN-2 in vectorized JAX.

    Computes the recurrent state update and linear attention output for a single
    token, advancing the hidden state using vectorized JAX operations.

    Args:
        q: Query tensor of shape (B, Hk, K) or (B, 1, Hk, K).
        k: Key tensor of shape (B, Hk, K) or (B, 1, Hk, K).
        v: Value tensor of shape (B, Hv, V) or (B, 1, Hv, V).
        g: Log-decay tensor of shape (B, Hk, K), (B, Hk, 1), (B, Hv), or scalar per head.
            Values must satisfy g <= 0; positive values are automatically clamped to 0.0.
        b: Erase gate tensor of shape (B, Hk, K), (B, Hk, 1), (B, Hv), or scalar per head.
        w: Write gate tensor of shape (B, Hv, V), (B, Hv, 1), or scalar per head.
        state: Recurrent hidden state tensor of shape (B, Hv, K, V).
        reset: Optional document boundary reset flag of shape (B,) or scalar.
            When True for batch element b, the state is cleared before the step.
        scale: Query scaling factor. Defaults to 1.0 / sqrt(K).
        qk_l2norm: Whether to apply L2 normalization to query and key. Defaults to False.
        state_dtype: Target dtype for output state (e.g. bfloat16 or float32). Defaults to float32.
            Note: bf16 state validated for channel-wise GDN-2 only; scalar-decay decode not validated.

    Returns:
        tuple of (output, new_state) where:
            output has shape (B, Hv, V) or (B, 1, Hv, V) matching query dimension.
            new_state has shape (B, Hv, K, V) in state_dtype (float32 by default).
    """
    out_dtype = v.dtype
    orig_ndim = q.ndim
    if orig_ndim not in (3, 4):
        raise ValueError(f"Query q must have rank 3 (B, Hk, K) or rank 4 (B, 1, Hk, K), got shape {q.shape}.")
    if k.ndim != orig_ndim:
        raise ValueError(f"Key k rank ({k.ndim}) must match query rank ({orig_ndim}).")
    if v.ndim != orig_ndim:
        raise ValueError(f"Value v rank ({v.ndim}) must match query rank ({orig_ndim}).")

    if orig_ndim == 4:
        if q.shape[1] != 1 or k.shape[1] != 1 or v.shape[1] != 1:
            raise ValueError(f"Decode step sequence dimension must be 1, got q={q.shape}, k={k.shape}, v={v.shape}.")
        q = q[:, 0]
        k = k[:, 0]
        v = v[:, 0]
        if g.ndim in (3, 4) and g.shape[1] == 1:
            g = g[:, 0]
        if b.ndim in (3, 4) and b.shape[1] == 1:
            b = b[:, 0]
        if w.ndim in (3, 4) and w.shape[1] == 1:
            w = w[:, 0]

    B, Hk, K = q.shape
    if v.ndim != 3 or v.shape[0] != B:
        raise ValueError(f"Value tensor v batch dimension ({v.shape[0]}) must match query batch dimension ({B}).")
    Hv, V = v.shape[1], v.shape[2]

    if g.ndim in (3, 4) and g.shape[0] == B and g.shape[1] == 1 and g.shape[2] in (1, Hk, Hv):
        g = g[:, 0]
    if b.ndim in (3, 4) and b.shape[0] == B and b.shape[1] == 1 and b.shape[2] in (1, Hk, Hv):
        b = b[:, 0]
    if w.ndim in (3, 4) and w.shape[0] == B and w.shape[1] == 1 and w.shape[2] in (1, Hk, Hv):
        w = w[:, 0]
    if k.shape != (B, Hk, K):
        raise ValueError(f"Key tensor k shape {k.shape} must match query q shape ({B}, {Hk}, {K}).")

    if K != 128 or V != 128:
        raise ValueError(f"Unsupported head dimensions (K={K}, V={V}); requires K=128 and V=128.")
    if Hv % Hk != 0:
        raise ValueError(f"Value heads ({Hv}) must be divisible by key heads ({Hk}).")

    if state.shape != (B, Hv, K, V):
        raise ValueError(f"state shape {state.shape} must match ({B}, {Hv}, {K}, {V}).")

    if state_dtype is not None:
        target_state_dtype = jnp.dtype(state_dtype)
    else:
        target_state_dtype = jnp.float32

    if g.ndim not in (2, 3) or g.shape[0] != B or g.shape[1] not in (1, Hk, Hv) or (g.ndim == 3 and g.shape[2] not in (1, K)):
        raise ValueError(f"g shape {g.shape} invalid for batch={B}, Hk={Hk}, Hv={Hv}, K={K}.")
    if b.ndim not in (2, 3) or b.shape[0] != B or b.shape[1] not in (1, Hk, Hv) or (b.ndim == 3 and b.shape[2] not in (1, K)):
        raise ValueError(f"b shape {b.shape} invalid for batch={B}, Hk={Hk}, Hv={Hv}, K={K}.")
    if w.ndim not in (2, 3) or w.shape[0] != B or w.shape[1] not in (1, Hk, Hv) or (w.ndim == 3 and w.shape[2] not in (1, V)):
        raise ValueError(f"w shape {w.shape} invalid for batch={B}, Hk={Hk}, Hv={Hv}, V={V}.")

    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    g = _nonpositive_gate(g.astype(jnp.float32))
    b = b.astype(jnp.float32)
    w = w.astype(jnp.float32)
    state = state.astype(jnp.float32)

    if reset is not None:
        reset_arr = jnp.asarray(reset)
        if reset_arr.ndim > 0 and reset_arr.shape[0] != B:
            raise ValueError(f"reset batch dimension ({reset_arr.shape[0]}) must match input batch dimension ({B}).")
        if reset_arr.ndim > 4:
            raise ValueError(f"reset ndim ({reset_arr.ndim}) cannot exceed 4.")
        while reset_arr.ndim < 4:
            reset_arr = reset_arr[..., None]
        state = jnp.where(reset_arr, jnp.zeros_like(state), state)

    # If g or b has Hv heads and Hk < Hv (e.g. Qwen3.5-4B), repeat q and k to Hv heads
    # Whichever gate is still at Hk heads is also repeated to Hv heads.
    if ((g.ndim >= 2 and g.shape[1] == Hv) or (b.ndim >= 2 and b.shape[1] == Hv)) and Hk < Hv:
        repeat_factor = Hv // Hk
        q = jnp.repeat(q, repeat_factor, axis=1)
        k = jnp.repeat(k, repeat_factor, axis=1)
        if g.ndim >= 2 and g.shape[1] == Hk:
            g = jnp.repeat(g, repeat_factor, axis=1)
        if b.ndim >= 2 and b.shape[1] == Hk:
            b = jnp.repeat(b, repeat_factor, axis=1)
        if w.ndim >= 2 and w.shape[1] == Hk:
            w = jnp.repeat(w, repeat_factor, axis=1)
        Hk = Hv

    G = Hv // Hk

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    scale_c = jnp.array(scale, dtype=jnp.float32)

    if qk_l2norm:
        q = (q / jnp.sqrt(jnp.sum(q ** 2, axis=-1, keepdims=True) + 1e-6)) * scale_c
        k = k / jnp.sqrt(jnp.sum(k ** 2, axis=-1, keepdims=True) + 1e-6)
    else:
        q = q * scale_c

    if G > 1:
        q = jnp.repeat(q, G, axis=1)
        k = jnp.repeat(k, G, axis=1)
        if g.shape[1] == Hk:
            g = jnp.repeat(g, G, axis=1)
        if b.shape[1] == Hk:
            b = jnp.repeat(b, G, axis=1)
        if w.shape[1] == Hk:
            w = jnp.repeat(w, G, axis=1)

    if g.ndim == 2:
        g = g[:, :, None]
    if g.shape[1] == 1 and Hv > 1:
        g = jnp.broadcast_to(g, (B, Hv, g.shape[-1]))
    if g.shape[-1] == 1:
        g = jnp.broadcast_to(g, (B, Hv, K))

    if b.ndim == 2:
        b = b[:, :, None]
    if b.shape[1] == 1 and Hv > 1:
        b = jnp.broadcast_to(b, (B, Hv, b.shape[-1]))
    if b.shape[-1] == 1:
        b = jnp.broadcast_to(b, (B, Hv, K))

    if w.ndim == 2:
        w = w[:, :, None]
    if w.shape[1] == 1 and Hv > 1:
        w = jnp.broadcast_to(w, (B, Hv, w.shape[-1]))
    if w.shape[-1] == 1:
        w = jnp.broadcast_to(w, (B, Hv, V))

    decay = jnp.exp(g)
    S_decayed = state * decay[:, :, :, None]

    beta = b * k
    e = jnp.sum(beta[:, :, :, None] * S_decayed, axis=2)
    v_new = w * v - e
    S_next = S_decayed + k[:, :, :, None] * v_new[:, :, None, :]
    o = jnp.sum(S_next * q[:, :, :, None], axis=2)

    o = o.astype(out_dtype)
    if orig_ndim == 4:
        o = o[:, None, :, :]

    S_next = S_next.astype(target_state_dtype)
    return o, S_next
