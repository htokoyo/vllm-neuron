# SPDX-License-Identifier: Apache-2.0
"""GDN TKG for T consecutive tokens of each row in ONE kernel call (MTP verify).

Same math, and per token the same instruction sequence, as ``gdn_tkg`` (vendored from
nkilib, l2norm + beta / g gates + gated delta rule + RMSNormGated): running this kernel on
T tokens equals running ``gdn_tkg`` T times, feeding each call the state the previous one
returned. What changes is what happens between the tokens:

  * the recurrent state stays in SBUF (fp32) from one token to the next; the state after
    every token is still written out (the verify step keeps it as a candidate), but as a
    store that overlaps the following tokens' work, not as a reload;
  * the per-token small ops that do not depend on the state (l2norm of q / k, beta, g,
    act(z), the final rsqrt of the RMSNorm) run once over all T tokens (and q with k, beta
    with act(z)), so each activation table is loaded about once per call instead of per token
    and head;
  * the inputs of the T tokens arrive in one DMA per tensor (except the transposed q / k
    loads, which the DGE still splits per token when the core holds part of a row's heads:
    19 DMAs per core at B=1, T=3, LNC 2, vs 3 x 13 for gdn_tkg).

Bit-identical to T x gdn_tkg in the NKI simulator (out and every state, T = 1..4, B = 1..3,
LNC 1 / 2; 2026-11-05). Static instruction counts per
core at B=1, T=3, LNC 2: 4 ACT_TABLE_LOAD (3 x 9 for gdn_tkg), 239 instructions (3 x 148).

Layout: rows-major, as the model produces them ([B, T, Hv, .]); the outputs are in the
layout the verify step consumes (``out`` [B, T, Hv, V] -> one out_proj for all tokens,
``state_out`` [B, T, Hv, K, V] -> the per-token recurrent-state candidates).

Inside the kernel the (token, head) pairs of this core are columns ``c = t * BH_local + h``
(token-major): when the core's heads are one run inside one row (B = 1, and B = 2 at LNC 2),
each input is one [T, n * F] DMA (with head-major columns the DGE split every load into T).
The recurrence walks token-major too (all heads of token t, then t + 1): the heads are
independent, so consecutive instructions of different heads can overlap.
"""

import math
from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast


def _head_runs(h_start: int, n_local: int, num_v_heads: int):
    """Split this core's flat heads [h_start, h_start + n_local) into runs inside one row.

    Returns (local head offset, row b, value head of the first head, run length) tuples.
    """
    runs = []
    h = 0
    while h < n_local:
        row, hv = divmod(h_start + h, num_v_heads)
        n = min(n_local - h, num_v_heads - hv)
        runs.append((h, row, hv, n))
        h += n
    return runs


def _load_rows(dst, src, runs, T, NV, BH_local, F, part_stride, transpose=False):
    """dst[:, (t * BH_local + h) * F + f] = src[row, t, hv, f] for this core's heads (``src``
    [B, T, Hv, F] in HBM). ``transpose``: dst is [F, T * BH_local] (partition = f) instead.

    One DMA per run of heads when the core's heads are one run (then each token's block is
    contiguous in dst and the heads' F-blocks are contiguous in src: a [T, n * F] pattern);
    one per run and token otherwise.
    """
    single = len(runs) == 1
    for h0, row, hv0, n in runs:
        base = (row * T * NV + hv0) * F
        if transpose:
            if single:
                nisa.dma_copy(dst=dst[:, 0 : T * n], src=src.ap(pattern=[[1, F], [NV * F, T], [F, n]], offset=base))
            else:
                for t in range(T):
                    nisa.dma_copy(
                        dst=dst[:, t * BH_local + h0 : t * BH_local + h0 + n],
                        src=src.ap(pattern=[[1, F], [F, n]], offset=base + t * NV * F),
                    )
        elif single:
            nisa.dma_copy(dst=dst[:, 0 : T * n * F], src=src.ap(pattern=[[part_stride, 1], [NV * F, T], [1, n * F]], offset=base))
        else:
            for t in range(T):
                c0 = (t * BH_local + h0) * F
                nisa.dma_copy(
                    dst=dst[:, c0 : c0 + n * F], src=src.ap(pattern=[[part_stride, 1], [1, n * F]], offset=base + t * NV * F)
                )


def _load_head_vector(dst, src, runs, T, NV, BH_local):
    """dst[:, t * BH_local + h] = src[hv(h)] (a per-value-head vector repeated over T)."""
    single = len(runs) == 1
    for h0, row, hv0, n in runs:
        if single:
            nisa.dma_copy(dst=dst[:, 0 : T * n], src=src.ap(pattern=[[NV, 1], [0, T], [1, n]], offset=hv0))
        else:
            for t in range(T):
                nisa.dma_copy(dst=dst[:, t * BH_local + h0 : t * BH_local + h0 + n], src=src.ap(pattern=[[NV, 1], [1, n]], offset=hv0))


@nki.jit
def gdn_tkg_multi(
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
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """T-token fused gated delta-rule decode (T one-token ``gdn_tkg`` steps in one call).

    Dimensions:
        B: rows, T: tokens per row (consecutive), Hv: value heads (q / k GQA-expanded),
        K, V: key / value head dims (<= 128). BH = B * Hv is sharded over the SPMD grid.

    Args:
        q, k (nl.NkiTensor): [B, T, Hv, K] @ HBM, RAW (not l2-normalized), bf16.
        v, z (nl.NkiTensor): [B, T, Hv, V] @ HBM, bf16.
        b, a (nl.NkiTensor): [B, T, Hv] @ HBM, RAW beta (pre-sigmoid) / gate input, fp32.
        A_log, dt_bias (nl.NkiTensor): [Hv] @ HBM, fp32.
        norm_weight (nl.NkiTensor): [V] @ HBM, fp32.
        state_in (nl.NkiTensor): [B, Hv, K, V] @ HBM, fp32, the state before token 0.
        gate_activation (str): "silu" or "sigmoid" (trace-time constant).

    Returns:
        out (nl.NkiTensor): [B, T, Hv, V] bf16, RMSNormGated output of every token.
        state_out (nl.NkiTensor): [B, T, Hv, K, V] fp32, the state after every token.
    """
    kernel_assert(gate_activation in ("silu", "sigmoid"), f"unsupported gate_activation {gate_activation!r}")
    gate_op = nl.sigmoid if gate_activation == "sigmoid" else nl.silu
    B, T, NV, K = q.shape
    V = v.shape[3]
    BH = B * NV
    kernel_assert(k.shape == (B, T, NV, K), f"k must be [B, T, Hv, K], got {k.shape=}")
    kernel_assert(v.shape == (B, T, NV, V) and z.shape == (B, T, NV, V), f"v / z must be [B, T, Hv, V], got {v.shape=} {z.shape=}")
    kernel_assert(b.shape == (B, T, NV) and a.shape == (B, T, NV), f"a / b must be [B, T, Hv], got {a.shape=} {b.shape=}")
    kernel_assert(A_log.shape[0] == NV and dt_bias.shape[0] == NV, f"A_log / dt_bias must be [Hv], got {A_log.shape=}")
    kernel_assert(state_in.shape == (B, NV, K, V), f"state_in must be [B, Hv, K, V], got {state_in.shape=}")
    kernel_assert(K <= 128 and V <= 128, f"K and V must each be <= 128, got {K=}, {V=}")
    kernel_assert(1 <= T <= 8, f"T must be in [1, 8], got {T=}")

    # ========== LNC Sharding (as gdn_tkg) ==========
    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)
    BH_local = BH // n_prgs
    h_start = prg_id * BH_local
    if prg_id == n_prgs - 1:
        BH_local = BH - h_start
    runs = _head_runs(h_start, BH_local, NV)
    C = BH_local * T  # (token, head) columns of this core, c = t * BH_local + h
    SW = BH_local * V  # free width of one token's states of this core

    out = nl.ndarray((B, T, NV, V), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    state_out = nl.ndarray((B, T, NV, K, V), dtype=nl.float32, buffer=nl.shared_hbm)
    scale = 1.0 / math.sqrt(K)

    # ========== Buffers ==========
    # S_hist[:, t * SW + h * V + v]: state after t tokens (t = 0: state_in).
    S_hist = nl.ndarray((K, (T + 1) * SW), dtype=nl.float32, buffer=nl.sbuf)
    S_bf_hist = nl.ndarray((K, T * SW), dtype=nl.bfloat16, buffer=nl.sbuf)
    # q | k side by side: one square, one sum-of-squares matmul, one rsqrt, one broadcast.
    QK_p = nl.ndarray((K, 2 * C), dtype=nl.bfloat16, buffer=nl.sbuf)
    Q_p = QK_p[:, 0:C]
    K_p = QK_p[:, C : 2 * C]
    v_all = nl.ndarray((1, C * V), dtype=nl.float32, buffer=nl.sbuf)
    # beta | z side by side: with the sigmoid gate one activation gives sigmoid(b) and act(z).
    bz = nl.ndarray((1, C + C * V), dtype=nl.float32, buffer=nl.sbuf)
    b_line = bz[:, 0:C]
    z_all = bz[:, C : C + C * V]
    k_f_all = nl.ndarray((1, C * K), dtype=nl.bfloat16, buffer=nl.sbuf)
    nw_line = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
    g_line = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
    alog_line = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
    dt_line = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
    exp_g_K = nl.ndarray((K, C), dtype=nl.float32, buffer=nl.sbuf)
    o_all = nl.ndarray((1, C * V), dtype=nl.float32, buffer=nl.sbuf)
    ssq = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
    out_sb = nl.ndarray((1, C * V), dtype=nl.bfloat16, buffer=nl.sbuf)

    # ========== Loads: one DMA per tensor (all T tokens) when the core's heads are one run ==========
    # Issue order = order of use: the gate inputs (softplus / exp / sigmoid), then q / k (rsqrt),
    # then the recurrence inputs.
    TOT = B * T * NV
    _load_rows(g_line, a, runs, T, NV, BH_local, 1, TOT)
    _load_rows(b_line, b, runs, T, NV, BH_local, 1, TOT)
    _load_head_vector(alog_line, A_log, runs, T, NV, BH_local)
    _load_head_vector(dt_line, dt_bias, runs, T, NV, BH_local)
    _load_rows(z_all, z, runs, T, NV, BH_local, V, TOT * V)
    # [K, (t, h)] transposed loads (partition = K), as gdn_tkg's Q_p / K_p.
    _load_rows(Q_p, q, runs, T, NV, BH_local, K, 0, transpose=True)
    _load_rows(K_p, k, runs, T, NV, BH_local, K, 0, transpose=True)
    # Row layout on partition 0: (t, h) blocks of K / V.
    _load_rows(k_f_all, k, runs, T, NV, BH_local, K, TOT * K)
    _load_rows(v_all, v, runs, T, NV, BH_local, V, TOT * V)
    nisa.dma_copy(dst=nw_line, src=norm_weight.reshape((1, V)))
    # State before token 0 (flat heads h_start.., contiguous in [B, Hv, K, V]).
    nisa.dma_copy(
        dst=S_hist[:, 0:SW],
        src=state_in.ap(pattern=[[V, K], [K * V, BH_local], [1, V]], offset=h_start * K * V),
    )

    # ========== State-independent pre-ops, once for all T tokens ==========
    # Elementwise the same instructions as gdn_tkg per token (bit-identical values); merged so the
    # activation-table order is softplus -> exp -> sigmoid -> rsqrt (the recurrence uses copy /
    # square only, which every table has, and the epilogue rsqrt reuses the last table). gdn_tkg
    # computes g, exp(g) and beta on all K partitions; exp(g) is computed on partition 0 and then
    # broadcast (same values), beta is read from partition 0 only.
    # no_reorder: keep this order on the activation engine (the scheduler otherwise puts the
    # q / k rsqrt between exp and sigmoid: 5 table loads instead of 4).
    with nl.no_reorder():
        gtmp = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
        expA = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gtmp, data1=g_line, data2=dt_line, op=nl.add)
        nisa.activation(dst=gtmp, data=gtmp, op=nl.softplus)
        nisa.activation(dst=expA, data=alog_line, op=nl.exp)
        nisa.scalar_tensor_tensor(dst=g_line, data=gtmp, op0=nl.multiply, operand0=-1.0, op1=nl.multiply, operand1=expA)
        nisa.activation(dst=gtmp, data=g_line, op=nl.exp)  # gtmp = exp(g)
        stream_shuffle_broadcast(src=gtmp, dst=exp_g_K)
        if gate_activation == "sigmoid":
            nisa.activation(dst=bz, data=bz, op=nl.sigmoid)  # beta = sigmoid(b), act(z)
        else:
            nisa.activation(dst=b_line, data=b_line, op=nl.sigmoid)
            nisa.activation(dst=z_all, data=z_all, op=gate_op)
        beta = b_line

        ones_K = nl.ndarray((K, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(ones_K, 1.0)
        qk_sq = nl.ndarray((K, 2 * C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=qk_sq, data=QK_p, op=nl.square)
        qk_ss = nl.ndarray((1, 2 * C), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(qk_ss, ones_K, qk_sq)
        qk_scale = nl.ndarray((K, 2 * C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=qk_scale[0:1, :], data=qk_ss, op=nl.rsqrt, bias=1e-6)
        stream_shuffle_broadcast(src=qk_scale, dst=qk_scale)
        q_scale = qk_scale[:, 0:C]
        k_scale = qk_scale[:, C : 2 * C]
        nisa.tensor_tensor(dst=K_p, data1=K_p, data2=k_scale, op=nl.multiply)
        Q_scaled = nl.ndarray((K, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=Q_scaled, data=Q_p, op0=nl.multiply, operand0=scale, op1=nl.multiply, operand1=q_scale
        )

    # ========== Recurrence: token-major, heads interleaved; state stays in SBUF ==========
    for t in range(T):
        for h in range(BH_local):
            c = t * BH_local + h
            S_prev = S_hist[:, nl.ds(t * SW + h * V, V)]
            S = S_hist[:, nl.ds((t + 1) * SW + h * V, V)]
            S_bf = S_bf_hist[:, nl.ds(t * SW + h * V, V)]
            k_p = K_p[:, c : c + 1]
            q_p = Q_scaled[:, c : c + 1]
            v_f = v_all[:, nl.ds(c * V, V)]
            k_f = k_f_all[:, nl.ds(c * K, K)]
            g_K_h = exp_g_K[:, c : c + 1]

            nisa.tensor_scalar(k_f, k_f, op0=nl.multiply, operand0=k_scale[0:1, c : c + 1])
            # Step 1: S = S_prev * exp(g)
            nisa.tensor_scalar(S, S_prev, op0=nl.multiply, operand0=g_K_h)
            nisa.tensor_copy(dst=S_bf, src=S, engine=nisa.scalar_engine)
            # Step 2: kv = k^T @ S
            kv_p = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(kv_p, k_p, S_bf)
            # Step 3: delta = (v - kv) * beta
            diff = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=diff, data1=v_f, data2=kv_p, op=nl.subtract)
            delta = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(delta, diff, op0=nl.multiply, operand0=beta[0:1, c : c + 1])
            # Step 4: S += outer(k, delta)
            delta_bf = nl.ndarray((1, V), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=delta_bf, src=delta, engine=nisa.scalar_engine)
            outer_p = nl.ndarray((K, V), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(outer_p, k_f, delta_bf)
            nisa.tensor_tensor(dst=S, data1=S, data2=outer_p, op=nl.add)
            # Step 5: o = q_scaled^T @ S
            nisa.tensor_copy(dst=S_bf, src=S, engine=nisa.scalar_engine)
            out_p = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(out_p, q_p, S_bf)
            o_sb = o_all[:, nl.ds(c * V, V)]
            nisa.tensor_copy(dst=o_sb, src=out_p, engine=nisa.scalar_engine)
            sq = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(sq, op=nl.square, data=o_sb, reduce_op=nl.add, reduce_res=ssq[0:1, c : c + 1])
        # The state after token t: stored while the next tokens run (S_hist slices are not reused).
        for h0, row, hv0, n in runs:
            nisa.dma_copy(
                dst=state_out.ap(pattern=[[V, K], [K * V, n], [1, V]], offset=((row * T + t) * NV + hv0) * K * V),
                src=S_hist[:, nl.ds((t + 1) * SW + h0 * V, n * V)],
            )

    # ========== RMSNormGated epilogue, once for all (head, token) columns ==========
    nisa.activation(dst=ssq, data=ssq, op=nl.rsqrt, scale=1.0 / V, bias=1e-6)
    for c in range(C):
        o_sb = o_all[:, nl.ds(c * V, V)]
        nisa.scalar_tensor_tensor(
            dst=o_sb, data=o_sb, op0=nl.multiply, operand0=ssq[0:1, c : c + 1], op1=nl.multiply, operand1=nw_line
        )
    nisa.tensor_tensor(dst=out_sb, data1=o_all, data2=z_all, op=nl.multiply)
    for h0, row, hv0, n in runs:
        base = (row * T * NV + hv0) * V
        if len(runs) == 1:
            nisa.dma_copy(dst=out.ap(pattern=[[TOT * V, 1], [NV * V, T], [1, n * V]], offset=base), src=out_sb)
        else:
            for t in range(T):
                c0 = (t * BH_local + h0) * V
                nisa.dma_copy(
                    dst=out.ap(pattern=[[TOT * V, 1], [1, n * V]], offset=base + t * NV * V), src=out_sb[:, c0 : c0 + n * V]
                )
    return out, state_out
