# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp hyper-connection read (``Qwen4ExpGatedResidual.forward``) in ONE kernel.

The op, per token (``hc`` = 4 streams, ``H`` = 2560, ``D = hc * H`` = 10240, ``R`` = hc_lowrank = 320)::

    normed = rmsnorm_per_H_group(streams) * (1 + w_norm)      fp32 math, rounded to bf16
    read   = sigmoid(up(silu(down(normed) / hc)))             [D]   (bf16 per linear, as torch)
    mixed  = mean_g(read[g] * normed[g])                      [H]
    inject = 2 * sigmoid(block_inject(normed) / hc)           [hc]

``hc_cte`` (prefill chunk, more than ``TKG_MAX_N`` = 32 rows; launched on the LNC grid): tokens are
split over the programs, each program does all features for its tokens in blocks of ``blk`` tokens
(weights streamed once per block). Up to 32 rows (decode / MTP verify) the model keeps the torch
forward.

Feature layout. Every on-chip tensor that is indexed by a feature uses ``f = H * g + S * p + j``
with ``p`` the partition, ``g`` the group, ``j`` in ``[0, S)``, ``S`` = 20. Every partition therefore
owns ``S`` CONSECUTIVE features per group, so the down weight ``wd_t`` = Wd^T ``[D, R]`` (row f
contiguous) is read in runs of ``S * R * 2`` bytes (12.8 KB). The price is a stride-S free access in
the stationary operand of the x transposes.

Rounding follows the torch path: bf16 normed, bf16 down / up / inject outputs before their
activation, bf16 ``read * normed`` products, fp32 sum of the hc products times 1/hc (exact) and one
final bf16 rounding. Differences to the CPU torch path: the summation order of the sum of squares
and of the matmuls, the hardware rsqrt / sigmoid / silu, and ``(x * w) * rstd`` instead of
``(x * rstd) * w`` in hc_cte (fp32, one ulp before the bf16 rounding).

Inputs (HBM): ``x`` [N, D] bf16 (the streams); ``wd_t`` [D, R] and ``wup_t`` [R, D] bf16; kernel-
layout copies built once at load (see ``kernel_layout`` below): ``w1`` (w_norm; the kernel adds 1 in fp32) and ``inj``
(block_inject^T, bf16; zeros for the final mixer). Returns ``(mixed [N, H] bf16, inject [N, hc] bf16)``.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

_P = 128
TKG_MAX_N = 32


def _ceil_div(a, b):
    return (a + b - 1) // b


def _identity(dst, n):
    """dst [n, n] (fp32 or bf16 SBUF) <- identity."""
    io = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[-1, n]], offset=0, channel_multiplier=1)  # p - i
    nisa.tensor_scalar(dst=dst, data=io, op0=nl.equal, operand0=0.0)


def _r_tiles(R):
    return [(rt, min(_P, R - rt * _P)) for rt in range(_ceil_div(R, _P))]


# ─────────────────────────────────────────────────────────────────────────────
# Prefill: tokens split over the programs, all features per program
# ─────────────────────────────────────────────────────────────────────────────


@nki.jit
def hc_cte(x, wd_t, wup_t, w1, inj, eps: float = 1e-6, hc: int = 4, blk: int = 512, round_up: bool = True):
    """See the module docstring. ``w1`` [128, hc, S] (hc_norm.weight), ``inj`` [128, hc, S, hc] bf16, S = H / 128."""
    N, D = x.shape
    R = wup_t.shape[0]
    H = D // hc
    _, n_prg, prg = get_verified_program_sharding_info("hc_cte", (0, 1), 2)
    kernel_assert(H % _P == 0, "hc_cte: H % 128")
    S = H // _P
    kernel_assert(tuple(w1.shape) == (_P, hc, S), f"hc_cte: w1 {w1.shape}")
    kernel_assert(tuple(inj.shape) == (_P, hc, S, hc), "hc_cte: inj layout")
    kernel_assert(blk % _P == 0 and blk <= 512, "hc_cte: blk")
    rts = _r_tiles(R)
    NRT = len(rts)
    RW = R + hc  # down | inject columns of the token-stationary down matmul

    n_per = _ceil_div(N, n_prg)
    t_lo = min(N, prg * n_per)
    t_hi = min(N, t_lo + n_per)

    mixed = nl.ndarray((N, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    w_out = nl.ndarray((N, hc), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    ident = nl.ndarray((_P, _P), dtype=nl.bfloat16, buffer=nl.sbuf)
    _identity(ident, _P)
    ident_f = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.sbuf)
    _identity(ident_f, _P)
    quarter = nl.ndarray((_P, _P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=quarter, data=ident, op0=nl.multiply, operand0=1.0 / hc)
    ones = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    eps_sb = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(eps_sb, eps)
    w1_raw = nl.ndarray((_P, hc, S), dtype=w1.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=w1_raw, src=w1)
    w1_sb = nl.ndarray((_P, hc, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=w1_sb, data=w1_raw, op0=nl.add, operand0=1.0)  # 1 + w in fp32
    inj_sb = nl.ndarray((_P, hc, S, hc), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=inj_sb, src=inj)

    for b0 in range(t_lo, t_hi, blk):
        nb = min(blk, t_hi - b0)
        tiles = [(t, b0 + t * _P, min(_P, nb - t * _P)) for t in range(_ceil_div(nb, _P))]

        # ---- per token tile: sum of squares (token layout), rstd broadcast, x^T, normed^T ---------
        nrm = nl.ndarray((_P, hc, S, blk), dtype=nl.bfloat16, buffer=nl.sbuf)  # becomes read * normed
        for t, r0, nt in tiles:
            x_sb = nl.ndarray((_P, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sb[0:nt, :], src=x[r0 : r0 + nt, :])
            ss = nl.ndarray((_P, hc), dtype=nl.float32, buffer=nl.sbuf)
            sq_junk = nl.ndarray((_P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            for g in range(hc):
                nisa.activation_reduce(dst=sq_junk[0:nt, :], op=nl.square, data=x_sb[0:nt, g * H : (g + 1) * H],
                                       reduce_op=nl.add, reduce_res=ss[0:nt, g : g + 1])
            rs = nl.ndarray((_P, hc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rs[0:nt, :], op=nl.rsqrt, data=ss[0:nt, :], scale=1.0 / H, bias=eps_sb[0:nt, :])
            # rstd_b[m, g, n] = rs[n, g] for every partition m: ones^T @ diag(rs[:, g]) (fp32, exact)
            dg = nl.ndarray((_P, hc, _P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dg[0:nt, :, 0:nt], data1=ident_f[0:nt, 0:nt].expand_dim(1).broadcast(1, hc),
                               data2=rs[0:nt, :].expand_dim(2).broadcast(2, nt), op=nl.multiply)
            rb_ps = nl.ndarray((_P, hc, _P), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=rb_ps[:, :, 0:nt], stationary=ones[0:nt, :], moving=dg[0:nt, :, 0:nt],
                           accumulate=False)
            rstd_b = nl.ndarray((_P, hc, _P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=rstd_b[:, :, 0:nt], src=rb_ps[:, :, 0:nt])

            xv = x_sb.reshape((_P, hc, _P, S))  # [n, g, p, j]
            for g in range(hc):
                for j0 in range(0, S, 4):
                    xT_ps = nl.ndarray((_P, 4, _P), dtype=nl.float32, buffer=nl.psum)
                    for jj in range(4):
                        nisa.nc_matmul(dst=xT_ps[:, jj, 0:nt], stationary=xv[0:nt, g, :, j0 + jj],
                                       moving=ident[0:nt, 0:nt], accumulate=False)
                    for jj in range(4):
                        nisa.scalar_tensor_tensor(
                            dst=nrm[:, g, j0 + jj, t * _P : t * _P + nt], data=xT_ps[:, jj, 0:nt],
                            op0=nl.multiply, operand0=w1_sb[:, g, j0 + jj : j0 + jj + 1],
                            op1=nl.multiply, operand1=rstd_b[:, g, 0:nt])

        # ---- down | inject (token stationary, weights moving), per token tile ------------------------
        # moving = [Wd^T rows | inject^T rows] of one slice in one SBUF tile: one matmul per (slice, tile)
        d_ps = [nl.ndarray((_P, RW), dtype=nl.float32, buffer=nl.psum) for _ in tiles]
        for g in range(hc):
            wd_sb = nl.ndarray((_P, S, RW), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=wd_sb[:, :, 0:R], src=wd_t.ap(pattern=[[S * R, _P], [R, S], [1, R]], offset=g * H * R))
            nisa.tensor_copy(dst=wd_sb[:, :, R:RW], src=inj_sb[:, g])
            for j in range(S):
                s = g * S + j
                for t, r0, nt in tiles:
                    nisa.nc_matmul(dst=d_ps[t][0:nt, :], stationary=nrm[:, g, j, t * _P : t * _P + nt],
                                   moving=wd_sb[:, j, :], accumulate=(s > 0))

        # ---- h = silu(bf16(d) / hc) -> h^T; inject = 2 sigmoid(bf16(logit) / hc) -> HBM ---------------
        hT = nl.ndarray((_P, NRT, blk), dtype=nl.bfloat16, buffer=nl.sbuf)
        for t, r0, nt in tiles:
            d_bf = nl.ndarray((_P, RW), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=d_bf[0:nt, :], src=d_ps[t][0:nt, :])
            h_tok = nl.ndarray((_P, R), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=h_tok[0:nt, :], op=nl.silu, data=d_bf[0:nt, 0:R], scale=1.0 / hc)
            wsig = nl.ndarray((_P, hc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=wsig[0:nt, :], op=nl.sigmoid, data=d_bf[0:nt, R:RW], scale=1.0 / hc)
            w_bf = nl.ndarray((_P, hc), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=w_bf[0:nt, :], data=wsig[0:nt, :], op0=nl.multiply, operand0=2.0)
            nisa.dma_copy(dst=w_out[r0 : r0 + nt, :], src=w_bf[0:nt, :])
            hT_ps = nl.ndarray((_P, NRT, _P), dtype=nl.float32, buffer=nl.psum)
            for rt, rows in rts:
                nisa.nc_matmul(dst=hT_ps[0:rows, rt, 0:nt], stationary=h_tok[0:nt, rt * _P : rt * _P + rows],
                               moving=ident[0:nt, 0:nt], accumulate=False)
            for rt, rows in rts:
                nisa.tensor_copy(dst=hT[0:rows, rt, t * _P : t * _P + nt], src=hT_ps[0:rows, rt, 0:nt])

        # ---- up (weights stationary, stride-S feature columns) -> read -> read * normed, in place -------
        for g in range(hc):
            wup_sb = nl.ndarray((_P, NRT, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            for rt, rows in rts:
                nisa.dma_copy(dst=wup_sb[0:rows, rt, :], src=wup_t[rt * _P : rt * _P + rows, g * H : (g + 1) * H])
            wv = wup_sb.reshape((_P, NRT, _P, S))
            for j in range(S):
                up_ps = nl.ndarray((_P, blk), dtype=nl.float32, buffer=nl.psum)
                for rt, rows in rts:
                    nisa.nc_matmul(dst=up_ps[:, 0:nb], stationary=wv[0:rows, rt, :, j], moving=hT[0:rows, rt, 0:nb],
                                   accumulate=(rt > 0))
                rd = nl.ndarray((_P, blk), dtype=nl.bfloat16, buffer=nl.sbuf)
                if round_up:  # bf16 up output before the sigmoid, as torch (one more PSUM pass)
                    nisa.tensor_copy(dst=rd[:, 0:nb], src=up_ps[:, 0:nb])
                    nisa.activation(dst=rd[:, 0:nb], op=nl.sigmoid, data=rd[:, 0:nb])
                else:
                    nisa.activation(dst=rd[:, 0:nb], op=nl.sigmoid, data=up_ps[:, 0:nb])
                # read * normed on GpSimd (SBUF operands): keeps the Vector engine for the PSUM-side work
                nisa.tensor_tensor(dst=nrm[:, g, j, 0:nb], data1=rd[:, 0:nb], data2=nrm[:, g, j, 0:nb], op=nl.multiply,
                                   engine=nisa.engine.gpsimd)

        # ---- mixed[n, S p + j] = sum_g prod / hc, token layout, per token tile ------------------------
        for t, r0, nt in tiles:
            mixed_sb = nl.ndarray((_P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            mv = mixed_sb.reshape((_P, _P, S))
            for j in range(S):
                # one PSUM tile per accumulation group (see hc_tkg's up: groups sharing a tile lose terms on device)
                mix_ps = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.psum)
                for g in range(hc):
                    nisa.nc_matmul(dst=mix_ps[0:nt, :], stationary=nrm[:, g, j, t * _P : t * _P + nt],
                                   moving=quarter, accumulate=(g > 0))
                nisa.tensor_copy(dst=mv[0:nt, :, j], src=mix_ps[0:nt, :])
            nisa.dma_copy(dst=mixed[r0 : r0 + nt, :], src=mixed_sb[0:nt, :])
    return mixed, w_out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel-layout copies (built once at load) and a torch model of the kernels
# ─────────────────────────────────────────────────────────────────────────────


def kernel_layout(norm_w, w_down, w_up, w_inj, hc: int):
    """Kernel inputs from the module weights (torch, CPU or device).

    ``norm_w`` [D] (hc_norm.weight), ``w_down`` [R, D], ``w_up`` [D, R], ``w_inj`` [hc, D] or None.
    Returns a dict (all in the weights' dtype): ``wd_t`` [D, R], ``wup_t`` [R, D], ``w1_cte`` [128, hc, S],
    ``inj_cte`` [128, hc, S, hc].
    """
    import torch

    D = norm_w.shape[0]
    H = D // hc
    s_cte = H // _P
    w1 = norm_w  # raw weight (bf16, like the other buffers): the kernels form 1 + w in fp32
    if w_inj is None:
        w_inj = torch.zeros(hc, D, dtype=w_down.dtype, device=w_down.device)
    inj_t = w_inj.t()  # [D, hc]
    return dict(
        wd_t=w_down.t().contiguous(),
        wup_t=w_up.t().contiguous(),
        w1_cte=w1.view(hc, _P, s_cte).permute(1, 0, 2).contiguous(),
        inj_cte=inj_t.reshape(hc, _P, s_cte, hc).permute(1, 0, 2, 3).contiguous(),
    )
