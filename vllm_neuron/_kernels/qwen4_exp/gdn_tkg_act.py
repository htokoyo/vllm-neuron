# SPDX-License-Identifier: Apache-2.0
"""``gdn_tkg_act``: :func:`gdn_tkg.gdn_tkg` with the activation-engine work regrouped (one-token GDN decode).

Same contract, same per-element instructions, same outputs as ``gdn_tkg`` (bit-identical; see below); only WHERE
the two RMSNormGated activations run changes.

Why: in ``gdn_tkg`` every head of the (Python-unrolled) loop runs ``rsqrt`` (the RMSNorm scale, ``gdn_tkg.py:322``)
and the gate activation on ``z`` (``:335``) one after the other; they are in different activation tables, so the
activation engine reloads its table twice per head. A TP16 B=16 decode device profile: 3,522
ACT_TABLE_LOAD per step (4.52 ms of instructions), 1,660 / 1,675 of the 1,728 ACTIVATEs at :322 / :335 reload
(96 %); the loop is ACT-bound (285 µs per layer).

Here:
  * the gate activation runs ONCE on all of this core's ``z`` (``z_all``, in place), next to ``sigmoid(b)`` (the
    same table for the sigmoid gate); it does not depend on the recurrence;
  * the per-head recurrence ends at ``o`` (kept in the head's slice of ``v_all``, which the head no longer reads)
    and its sum of squares (column h of ``ssq``); after every ``rsqrt_group`` heads ONE ``rsqrt`` for the group (the
    table of the q / k l2norm rsqrt, the last one loaded), per head ``o * rsqrt * norm_weight`` (DVE, as before) and
    one ``o * act(z)`` multiply for the group into ``out_sb`` -- the group's RMSNorm runs on DVE while the next
    group's recurrence runs, and the loop uses only copy / square / rsqrt on the activation engine (one table);
  * the state-independent pre-ops keep their order on the activation engine (``no_reorder``):
    softplus -> exp -> sigmoid (b, z) -> square -> rsqrt, so a call loads 4 tables instead of 2 per head.

Equivalence: every output element is produced by the same instruction type with the same operands (scale, bias,
dtypes) as in ``gdn_tkg`` -- only the grouping of independent elements into instructions differs (the gate on
[1, BH_local * V] instead of [1, V] per head, rsqrt on [1, group] instead of [1, 1]) and the RMSNorm output is
written in place instead of into a fresh tile; the recurrence (states, matmuls, casts) is untouched. Checked
bit for bit against ``gdn_tkg`` in the NKI simulator and BIRSim.

Trace-time options (defaults = the deployed form): ``rsqrt_group`` (heads per batched rsqrt, default 8; 1 = per
head, >= the core's heads = one rsqrt after the loop), ``pre_no_reorder`` (False: let the scheduler order the pre-ops).
"""

import math
from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@nki.jit
def gdn_tkg_act(
    q: nl.NkiTensor,
    k: nl.NkiTensor,
    v: nl.NkiTensor,
    b: nl.NkiTensor,
    a: nl.NkiTensor,
    A_log: nl.NkiTensor,
    dt_bias: nl.NkiTensor,
    z: nl.NkiTensor,
    norm_weight: nl.NkiTensor,
    state_in: nl.NkiTensor,
    gate_activation: str = "silu",
    rsqrt_group: int = 8,
    pre_no_reorder: bool = True,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """Fused single-token gated delta-rule decode step (``gdn_tkg`` contract).

    Args (as ``gdn_tkg``):
        q, k (nl.NkiTensor): [BH, K] @ HBM, RAW (GQA-expanded, not l2-normed).
        v, z (nl.NkiTensor): [BH, V] @ HBM.
        b, a (nl.NkiTensor): [BH] @ HBM, RAW beta / gate inputs.
        A_log, dt_bias (nl.NkiTensor): [num_v_heads] @ HBM.
        norm_weight (nl.NkiTensor): [V] @ HBM.
        state_in (nl.NkiTensor): [BH, K, V] @ HBM.
        gate_activation (str): "silu" or "sigmoid" (trace-time constant).
        rsqrt_group (int): heads per RMSNorm rsqrt instruction (>= 1; trace-time constant).
        pre_no_reorder (bool): keep the pre-op order on the activation engine.

    Returns:
        out (nl.NkiTensor): [BH, V] @ HBM bf16; state_out (nl.NkiTensor): [BH, K, V] @ HBM fp32.
    """
    kernel_assert(gate_activation in ("silu", "sigmoid"), f"unsupported gate_activation {gate_activation!r}")
    gate_op = nl.sigmoid if gate_activation == "sigmoid" else nl.silu
    BH, K = q.shape
    _, V = v.shape
    num_v_heads = A_log.shape[0]
    B = BH // num_v_heads
    kernel_assert(
        state_in.shape == (BH, K, V),
        f"state_in must be [BH, K, V], got {state_in.shape=}, {BH=}, {K=}, {V=}",
    )
    kernel_assert(K <= 128 and V <= 128, f"K and V must each be <= 128, got {K=}, {V=}")
    kernel_assert(
        dt_bias.shape[0] == num_v_heads,
        f"dt_bias must be [num_v_heads], got {dt_bias.shape[0]=}, {num_v_heads=}",
    )
    kernel_assert(isinstance(rsqrt_group, int) and rsqrt_group >= 1, f"rsqrt_group must be >= 1, got {rsqrt_group!r}")
    kernel_assert(B * num_v_heads == BH, f"BH must equal B * num_v_heads, got {B=}, {num_v_heads=}, {BH=}")

    # ========== LNC Sharding (as gdn_tkg) ==========
    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)
    BH_local = BH // n_prgs
    h_start = prg_id * BH_local
    if prg_id == n_prgs - 1:
        BH_local = BH - h_start  # last core takes the remainder

    out = nl.ndarray((BH, V), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    state_out = nl.ndarray((BH, K, V), dtype=nl.float32, buffer=nl.shared_hbm)
    scale = 1.0 / math.sqrt(K)

    # ========== Pre-allocated Buffers (as gdn_tkg, plus ssq) ==========
    S_all = nl.ndarray((K, BH_local * V), dtype=nl.float32, buffer=nl.sbuf)
    S_bf_all = nl.ndarray((K, BH_local * V), dtype=nl.bfloat16, buffer=nl.sbuf)

    Q_p = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    K_p = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    v_all = nl.ndarray((1, BH_local * V), dtype=nl.float32, buffer=nl.sbuf)

    exp_g_K = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    beta_K = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)

    out_sb = nl.ndarray((1, BH_local * V), dtype=nl.bfloat16, buffer=nl.sbuf)
    ssq = nl.ndarray((1, BH_local), dtype=nl.float32, buffer=nl.sbuf)  # sum(o^2), then rsqrt(mean + eps)

    # ========== Bulk Preloads (as gdn_tkg, same order) ==========
    nisa.dma_copy(dst=Q_p, src=q.ap(pattern=[[1, K], [K, BH_local]], offset=h_start * K))
    nisa.dma_copy(dst=K_p, src=k.ap(pattern=[[1, K], [K, BH_local]], offset=h_start * K))

    nisa.dma_copy(dst=v_all, src=v.reshape((1, BH * V))[:, h_start * V : h_start * V + BH_local * V])
    # The head's o (then o * rsqrt * norm_weight) is kept in its v slice, which the head no longer reads once delta
    # is formed: no extra [1, BH_local * V] fp32 tile on partition 0 (SBUF as gdn_tkg).
    o_all = v_all

    z_all = nl.ndarray((1, BH_local * V), dtype=nl.float32, buffer=nl.sbuf)  # z, then act(z) in place
    nisa.dma_copy(dst=z_all, src=z.reshape((1, BH * V))[:, h_start * V : h_start * V + BH_local * V])
    nw_line = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nw_line, src=norm_weight.reshape((1, V)))

    k_f_all = nl.ndarray((1, BH_local * K), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_f_all, src=k.reshape((1, BH * K))[:, h_start * K : h_start * K + BH_local * K])

    g_line = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    b_line = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(g_line, 0.0)
    nisa.memset(b_line, 0.0)
    nisa.dma_copy(dst=g_line[0:1, :], src=a.reshape((1, BH))[:, h_start : h_start + BH_local])
    nisa.dma_copy(dst=b_line[0:1, :], src=b.reshape((1, BH))[:, h_start : h_start + BH_local])

    alog_full = nl.ndarray((1, BH), dtype=nl.float32, buffer=nl.sbuf)
    dt_full = nl.ndarray((1, BH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=alog_full, src=A_log.ap(pattern=[[0, B], [1, num_v_heads]], offset=0))
    nisa.dma_copy(dst=dt_full, src=dt_bias.ap(pattern=[[0, B], [1, num_v_heads]], offset=0))
    alog_h = alog_full[0:1, nl.ds(h_start, BH_local)]
    dt_h = dt_full[0:1, nl.ds(h_start, BH_local)]

    # ========== State-independent pre-ops (gdn_tkg's instructions + the gate activation) ==========
    # Activation-table order: softplus -> exp -> sigmoid (beta, and the sigmoid gate) -> square -> rsqrt.
    with nl.no_reorder() if pre_no_reorder else _NullCtx():
        gtmp = nl.ndarray((1, BH_local), dtype=nl.float32, buffer=nl.sbuf)
        expA = nl.ndarray((1, BH_local), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gtmp, data1=g_line[0:1, :], data2=dt_h, op=nl.add)
        nisa.activation(dst=gtmp, data=gtmp, op=nl.softplus)
        nisa.activation(dst=expA, data=alog_h, op=nl.exp)
        nisa.scalar_tensor_tensor(
            dst=g_line[0:1, :], data=gtmp, op0=nl.multiply, operand0=-1.0, op1=nl.multiply, operand1=expA
        )

        stream_shuffle_broadcast(src=g_line, dst=g_line)
        stream_shuffle_broadcast(src=b_line, dst=b_line)

        nisa.activation(dst=exp_g_K, data=g_line, op=nl.exp)
        nisa.activation(dst=beta_K, data=b_line, op=nl.sigmoid)
        # Hoisted out of the per-head loop (gdn_tkg.py:335): act(z) for all heads of this core, in place.
        nisa.activation(dst=z_all, data=z_all, op=gate_op)

        ones_K = nl.ndarray((K, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(ones_K, 1.0)
        q_sq = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
        k_sq = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=q_sq, data=Q_p, op=nl.square)
        nisa.activation(dst=k_sq, data=K_p, op=nl.square)
        q_ss = nl.ndarray((1, BH_local), dtype=nl.float32, buffer=nl.psum)
        k_ss = nl.ndarray((1, BH_local), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(q_ss, ones_K, q_sq)
        nisa.nc_matmul(k_ss, ones_K, k_sq)
        q_scale = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
        k_scale = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=q_scale[0:1, :], data=q_ss, op=nl.rsqrt, bias=1e-6)
        nisa.activation(dst=k_scale[0:1, :], data=k_ss, op=nl.rsqrt, bias=1e-6)
        stream_shuffle_broadcast(src=q_scale, dst=q_scale)
        stream_shuffle_broadcast(src=k_scale, dst=k_scale)

        nisa.tensor_tensor(dst=K_p, data1=K_p, data2=k_scale, op=nl.multiply)

        Q_scaled = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=Q_scaled, data=Q_p, op0=nl.multiply, operand0=scale, op1=nl.multiply, operand1=q_scale
        )

    # Bulk load this core's states in one transposed DMA (as gdn_tkg, after the pre-ops in program order)
    nisa.dma_copy(
        dst=S_all,
        src=state_in.ap(pattern=[[V, K], [K * V, BH_local], [1, V]], offset=h_start * K * V),
    )

    # ========== Per-head Recurrence (as gdn_tkg) ==========
    for head_idx in range(BH_local):
        S = S_all[:, nl.ds(head_idx * V, V)]
        S_bf = S_bf_all[:, nl.ds(head_idx * V, V)]
        k_p = K_p[:, head_idx : head_idx + 1]
        q_p = Q_scaled[:, head_idx : head_idx + 1]
        v_f = v_all[:, nl.ds(head_idx * V, V)]
        k_f = k_f_all[:, nl.ds(head_idx * K, K)]
        g_K_h = exp_g_K[:, head_idx : head_idx + 1]
        b_K_h = beta_K[:, head_idx : head_idx + 1]

        nisa.tensor_scalar(k_f, k_f, op0=nl.multiply, operand0=k_scale[0:1, head_idx : head_idx + 1])
        nisa.tensor_scalar(S, S, op0=nl.multiply, operand0=g_K_h)
        nisa.tensor_copy(dst=S_bf, src=S, engine=nisa.scalar_engine)
        kv_p = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(kv_p, k_p, S_bf)
        diff = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=diff, data1=v_f, data2=kv_p, op=nl.subtract)
        delta = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(delta, diff, op0=nl.multiply, operand0=b_K_h[0:1, 0:1])
        delta_bf = nl.ndarray((1, V), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=delta_bf, src=delta, engine=nisa.scalar_engine)
        outer_p = nl.ndarray((K, V), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(outer_p, k_f, delta_bf)
        nisa.tensor_tensor(dst=S, data1=S, data2=outer_p, op=nl.add)
        nisa.tensor_copy(dst=S_bf, src=S, engine=nisa.scalar_engine)
        out_p = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(out_p, q_p, S_bf)

        # RMSNormGated: o and sum(o^2) (copy / square: no table change).
        o_sb = o_all[:, nl.ds(head_idx * V, V)]
        nisa.tensor_copy(dst=o_sb, src=out_p, engine=nisa.scalar_engine)
        sq = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(
            sq, op=nl.square, data=o_sb, reduce_op=nl.add, reduce_res=ssq[0:1, head_idx : head_idx + 1]
        )

        # After the last head of a group: the group's RMSNorm scale and output.
        if (head_idx + 1) % rsqrt_group == 0 or head_idx == BH_local - 1:
            g0 = head_idx - head_idx % rsqrt_group
            gn = head_idx + 1 - g0
            # rsqrt(mean(o^2) + eps) = rsqrt(sum / V + eps) for the group's heads in one instruction.
            nisa.activation(dst=ssq[0:1, g0 : g0 + gn], data=ssq[0:1, g0 : g0 + gn], op=nl.rsqrt, scale=1.0 / V, bias=1e-6)
            for gh in range(g0, g0 + gn):
                o_g = o_all[:, nl.ds(gh * V, V)]
                nisa.scalar_tensor_tensor(
                    dst=o_g, data=o_g, op0=nl.multiply, operand0=ssq[0:1, gh : gh + 1], op1=nl.multiply, operand1=nw_line
                )
            nisa.tensor_tensor(
                dst=out_sb[:, nl.ds(g0 * V, gn * V)],
                data1=o_all[:, nl.ds(g0 * V, gn * V)],
                data2=z_all[:, nl.ds(g0 * V, gn * V)],
                op=nl.multiply,
            )

    # ========== Bulk Stores (as gdn_tkg) ==========
    nisa.dma_copy(dst=out.reshape((1, BH * V))[:, h_start * V : h_start * V + BH_local * V], src=out_sb)
    nisa.dma_copy(
        dst=state_out.ap(pattern=[[V, K], [K * V, BH_local], [1, V]], offset=h_start * K * V),
        src=S_all,
    )

    return out, state_out
